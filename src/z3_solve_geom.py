import argparse
import csv
import json
import time
import re
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для генерации Z3 кода
# ============================================================================
SYSTEM_PROMPT = """Ты переводишь геометрические задачи в код Python с использованием Z3 solver.

# МЕТОД РЕШЕНИЯ

Используй координатный метод:
1. Размести фигуру в системе координат
2. Переведи все условия в уравнения
3. Не повторяй одно и то же условие несколько раз
4. Реши систему через Z3
5. Вычисли искомую величину

# ПРАВИЛА РАЗМЕЩЕНИЯ

- Одну точку помести в начало координат (0, 0)
- Одну сторону направь вдоль оси X

# ФОРМУЛЫ для записи самых частых условий

## Как задать равенство отрезков AB и CD:
s.add((bx - ax)**2 + (by - ay)**2 == (cx - dx)**2 + (cy - dy)**2)

## Как задать что AB перпендикулярно CD:
s.add((bx - ax)*(dx - cx) + (by - ay)*(dy - cy) == 0)

## Как задать что точка P на прямой AB:
s.add((px - ax) * (by - ay) == (py - ay) * (bx - ax))

## Как задать что точка P на отрезке AB:
s.add((px - ax) * (by - ay) == (py - ay) * (bx - ax))
s.add(And((px - ax) * (px - bx) <= 0, (py - ay) * (py - by) <= 0))

## Как задать что точка P середина AB:
s.add((px - ax) == (bx - px))
s.add((py - ay) == (by - py))

## Как задать что угол ABC равен углу DEF:
s.add(((ax - bx) * (cx - bx) + (ay - by) * (cy - by)) ** 2 * ((dx - ex) ** 2 + (dy - ey) ** 2) * ((fx - ex) ** 2 + (fy - ey) ** 2) == ((dx - ex) * (fx - ex) + (dy - ey) * (fy - ey)) ** 2 * ((ax - bx) ** 2 + (ay - by) ** 2) * ((cx - bx) ** 2 + (cy - by) ** 2))

## Как задать что угол ABC равен a градусов:
cos_angle = math.cos(a * math.pi / 180)
s.add(cos_angle ** 2 * ((ax - bx) ** 2 + (ay - by) ** 2) * ((cx - bx) ** 2 + (cy - by) ** 2) == ((ax - bx) * (cx - bx) + (ay - by) * (cy - by)) ** 2)

## Биссектриса угла BAC:
Точка X на биссектрисе:
Аналогично условию для равенства углов

# ШАБЛОН КОДА

```python
from z3 import *
import math

# Создаём solver с таймаутом
s = Solver()
s.set("timeout", 10000)  # 10 секунд таймаут

# Объявляем координаты точек как вещественные переменные
ax, ay = Reals('ax ay')
bx, by = Reals('bx by')
# ... другие точки

# Задаём условие что точки попарно не совпадают:
s.add(Or(ax != bx, ay != by))
s.add(Or(ax != cx, ay != cy))
# ... другие пары

# Фиксируем систему координат
s.add(ax == 0, ay == 0)  # A в начале
s.add(bx > 0, by == 0)   # B на положительной полуоси X

# Добавляем условия задачи
# ...

# Вспомогательные переменные для искомой величины
answer = Real('answer')
s.add(...)  # условие для answer, если answer это угол, то вместо него ищешь квадрат его косинуса

# Решаем
if s.check() == sat:
    m = s.model()
    # Извлекаем ответ
    result = m[answer]
    # Преобразуем в число
    if result is not None:
        # Для дробей
        if hasattr(result, 'numerator_as_long') and hasattr(result, 'denominator_as_long'):
            num = result.numerator_as_long()
            den = result.denominator_as_long()
            res = num / den
        else:
            res = result
        # Если в качестве ответа просили найти угол считаешь арккосинус корня из ответа
        res = ...
        print(f"Ответ: {res}")
else:
    print("Решение не найдено")
```

ВАЖНО:
1. Всегда добавляй s.set("timeout", 10000) после создания Solver()
2. Код должен быть полностью рабочим и выводить ответ в формате "Ответ: число"
3. Не используй внешние библиотеки кроме z3 и math
4. Ответ должен быть числом (целым или десятичной дробью)
"""

USER_PROMPT = """Переведи эту геометрическую задачу в код Python с Z3 solver.

Задача: {question}

Напиши только код Python, без объяснений. НЕ ПОВТОРЯЙ условия. Напиши код ОДИН раз и закончи. Код должен выводить ответ в формате "Ответ: число".
```python
"""

# ============================================================================
# Параметры
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
MAX_TOKENS = 2000
Z3_TASKS_DIR = Path("src/z3_tasks")


def extract_code(response: str) -> str:
    """Извлекает Python код из ответа модели."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.replace('<|im_end|>', '')
    response = response.replace('<|eot_id|>', '')
    response = response.strip()

    # Ищем код между ```python и ```
    code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Если нет маркеров, пробуем взять весь ответ как код
    # Убираем возможные маркеры в начале
    code = response.strip()
    if code.startswith('```'):
        code = code[3:]
    if code.endswith('```'):
        code = code[:-3]
    
    return code.strip()


def extract_answer_from_output(output: str) -> float | None:
    """Извлекает числовой ответ из вывода программы."""
    # Ищем "Ответ: число"
    match = re.search(r'Ответ:\s*(-?\d+\.?\d*)', output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Ищем любое число в последней строке
    lines = output.strip().split('\n')
    if lines:
        last_line = lines[-1]
        num_match = re.search(r'-?\d+\.?\d*', last_line)
        if num_match:
            try:
                return float(num_match.group())
            except ValueError:
                pass
    
    return None


def run_z3_code(code: str, task_index: int, timeout: int = 30) -> tuple[str, float | None, str]:
    """
    Выполняет Z3 код и возвращает (вывод, ответ, ошибка).
    """
    # Сохраняем код в файл
    task_file = Z3_TASKS_DIR / f"task_{task_index}.py"
    with open(task_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Запускаем код с таймаутом
    try:
        result = subprocess.run(
            [sys.executable, str(task_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd())
        )
        
        output = result.stdout
        error = result.stderr
        
        if result.returncode != 0:
            return output, None, error
        
        answer = extract_answer_from_output(output)
        return output, answer, error
        
    except subprocess.TimeoutExpired:
        return "", None, "Timeout: выполнение превысило лимит времени"
    except Exception as e:
        return "", None, str(e)


def compare_answers(predicted: float | None, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Сравнивает предсказанный ответ с правильным."""
    if predicted is None:
        return False
    
    # Парсим ground_truth
    gt_values = []
    for part in ground_truth.replace(';', ',').split(','):
        part = part.strip()
        if '/' in part:
            try:
                num, den = part.split('/')
                gt_values.append(float(num) / float(den))
            except:
                pass
        else:
            try:
                gt_values.append(float(part))
            except:
                pass
    
    # Проверяем совпадение с любым из правильных ответов
    for gt in gt_values:
        if abs(predicted - gt) <= tolerance:
            return True
    
    return False


def load_dataset(csv_path: str) -> list[dict]:
    """Загружает датасет из CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def run_evaluation(dataset_path: str, limit: int | None = None):
    """Запускает оценку на датасете."""
    
    # Создаём директорию для задач
    Z3_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Загружаем датасет
    print(f"📂 Загрузка датасета: {dataset_path}")
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]
    print(f"   Задач: {len(dataset)}")
    
    # Загружаем модель
    print(f"\n🤖 Загрузка модели: {MODEL_NAME}")
    start = time.time()
    model, tokenizer = load(MODEL_NAME)
    print(f"   Загружена за {time.time() - start:.1f} сек")
    
    # Результаты
    results = []
    correct_count = 0
    total_time = 0
    
    print(f"\n{'='*60}")
    print("ЗАПУСК ОЦЕНКИ (Z3 Solver)")
    print(f"{'='*60}")
    
    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["verifiable_answer"]
        
        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")
        
        task_start = time.time()
        
        # Формируем промпт через chat template, как в preprocessing3
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(question=question)},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        # Генерируем код
        response = generate(
            model, tokenizer,
            prompt=full_prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=0.0),
        )
        
        # Извлекаем код
        code = extract_code(response)
        
        # Выполняем код
        output, predicted, error = run_z3_code(code, i)
        
        task_time = time.time() - task_start
        total_time += task_time
        
        # Проверяем ответ
        is_correct = compare_answers(predicted, ground_truth)
        
        if is_correct:
            correct_count += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"   Предсказано: {predicted}, Правильно: {ground_truth} {status}")
        if error:
            print(f"   Ошибка: {error[:100]}...")
        print(f"   Время: {task_time:.1f} сек")
        
        # Сохраняем результат (без кода, как указано в задании)
        results.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "time": task_time,
            "error": error if error else None
        })
    
    # Итоговая статистика
    accuracy = correct_count / len(dataset) if dataset else 0
    
    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"Всего задач:   {len(dataset)}")
    print(f"Правильных:    {correct_count}")
    print(f"Accuracy:      {accuracy*100:.1f}%")
    print(f"Общее время:   {total_time:.1f} сек")
    print(f"Среднее время: {total_time/len(dataset):.1f} сек/задача")
    print(f"{'='*60}")
    
    # Сохраняем результаты
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(dataset_path).stem
    output_file = output_dir / f"z3_{dataset_name}_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "dataset": dataset_path,
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "total": len(dataset),
        "correct": correct_count,
        "accuracy": accuracy,
        "total_time": total_time,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Решение геометрических задач с помощью Z3 solver")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")
    
    args = parser.parse_args()
    run_evaluation(args.dataset, args.limit)


if __name__ == "__main__":
    main()