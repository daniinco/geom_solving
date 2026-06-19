import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения того, что нужно найти (из preprocessing3/get_statement.py)
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только требуемое строго по формату, который я тебе скажу. Напиши "Найти: " и сразу пиши условия и после этого ничего не пиши.
Определи, что нужно найти в геометрической задаче.

ТИПЫ ОТВЕТОВ:
1 XY — длина (отрезок, сторона, диагональ, медиана, высота) → ровно 2 буквы
2 XYZ — угол → ровно 3 буквы, вершина угла ВСЕГДА в середине
3 XY ZT - угол между прямыми XY и ZT → ровно 2 раза по 2 буквы

ГЛАВНОЕ ПРАВИЛО:
Если угол задан одной буквой ("угол A в треугольнике ABC"), раскрой в 3 буквы:
вершину ставь в середину, соседние вершины фигуры — по бокам.
- Треугольник ABC: угол A → BAC, угол B → ABC, угол C → BCA
- Четырёхугольник ABCD: угол A → DAB, угол B → ABC, угол C → BCD, угол D → CDA

ФОРМАТ: строго одна строка — цифра, пробел, условие. Ничего больше.

ПРИМЕРЫ:

Задача: В треугольнике ABC найдите BC.
Найти: 1 BC

Задача: Найдите угол ABC.
Найти: 2 ABC

Задача: Чему равна сторона MN?
Найти: 1 MN

Задача: Найдите угол A в треугольнике ABC.
Найти: 2 BAC

Задача: Найдите угол B в треугольнике ABC.
Найти: 2 ABC

Задача: Найдите в треугольнике ABC угол между прямой AC и медианой BM.
Найти: 3 AC BM

Задача: Вычислите длину диагонали BD.
Найти: 1 BD

Задача: Сколько градусов составляет угол ACB?
Найти: 2 ACB

Задача: В параллелограмме ABCD найдите угол D.
Найти: 2 CDA

Задача: В параллелограмме ABCD найдите угол между диагоналями.
Найти: 3 AC BD

Задача: Найдите медиану AM треугольника ABC.
Найти: 1 AM

Задача: Чему равен угол при вершине C треугольника BCD?
Найти: 2 BCD"""

THINK_INSTRUCTION = "Сначала порассуждай вслух: что именно нужно найти в задаче — длину или угол, как правильно записать по формату (1 XY или 2 XYZ или 3 XY ZT). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по определению того, что нужно найти, и уже извлечённый ответ.
Порассуждай строго по чеклисту:

1. ЧТО ИЩЕМ: Определи, что нужно найти в задаче:
   - Длина отрезка, стороны, диагонали, медианы, высоты → тип 1
   - Угол (в градусах или как величина) → тип 2
   - Угол между двумя прямыми/отрезками → тип 3

2. ФОРМАТ по типу:
   - Тип 1 (длина): "1 XY" — ровно 2 буквы
   - Тип 2 (угол): "2 XYZ" — ровно 3 буквы, вершина угла ВСЕГДА в середине
   - Тип 3 (угол между прямыми): "3 XY ZT" — две пары по 2 буквы

3. РАСКРЫТИЕ УГЛА (тип 2): Если угол задан одной буквой — раскрой по фигуре:
   - Треугольник ABC: угол A → BAC, угол B → ABC, угол C → BCA
   - Четырёхугольник ABCD: угол A → DAB, угол B → ABC, угол C → BCD, угол D → CDA
   - Вершина угла — СРЕДНЯЯ буква (не первая, не последняя)

4. РАЗЛИЧИЕ ТИПОВ 2 и 3:
   - "Найдите угол ABC" → тип 2: "2 ABC"
   - "Найдите угол между диагоналями AC и BD" → тип 3: "3 AC BD"
   - "Найдите угол между прямой MN и медианой BK" → тип 3: "3 MN BK"

5. ИТОГ: Совпадает ли извлечённое с тем, что должно быть?

ТИПИЧНЫЕ ОШИБКИ:
- Угол между прямыми записан как тип 2 (2 XYZ) вместо типа 3 (3 XY ZT)
- Вершина угла стоит не в середине: записано "2 ABС" когда нужно "2 BAC"
- Угол A в треугольнике ABC записан как "2 ABC" вместо "2 BAC"
- Длина записана как угол или наоборот

Рассуждение пиши свободно, не более 400-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по определению того, что нужно найти, извлечённый ответ и рассуждение о правильности.
На основе рассуждения вынеси вердикт: если извлечённый ответ ПОЛНОСТЬЮ совпадает с тем, что должно быть по правилам (правильный тип, правильный формат, правильное раскрытие угла) — ответь "Правильно". Если есть ХОТЬ ОДНО отличие — ответь "Неправильно".
Ответь СТРОГО одним словом — ничего больше, никаких знаков препинания, никаких пояснений:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по определению того, что нужно найти, извлечённый ответ, рассуждение о правильности и вердикт о том, что ответ неправильный.
Определи заново, что нужно найти, строго по правилам из инструкции.
Напиши ТОЛЬКО: Найти: <ответ в виде "1 XY", "2 XYZ" или "3 XY ZT">
Никаких пояснений, никакого рассуждения, только одна строка."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 30   # Токены для исправленного условия

CONDITION_COLUMN = "statement_condition"


def load_dataset(csv_path: str) -> list[dict]:
    """Загружает датасет из CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def clean_response(response: str) -> str:
    """Очищает ответ от служебных токенов."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.replace('<|im_end|>', '')
    response = response.replace('<|eot_id|>', '')
    return response.strip()


def extract_condition(response: str) -> str:
    """Извлекает условие из ответа модели."""
    response = clean_response(response)
    response = response.replace('<|im_end|>', '')
    response = response.replace('<|eot_id|>', '')
    response = response.replace('Найти:', '')
    response = response.strip()

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return response


def is_correct_verdict(verdict: str) -> bool:
    """Определяет, признала ли модель условие правильным."""
    verdict_clean = clean_response(verdict).strip().lower()
    return verdict_clean.startswith('правильно') and 'не' not in verdict_clean


def run_validation(dataset_path: str, limit: int | None = None):
    """Запускает валидацию того, что нужно найти, на датасете."""

    print(f"📂 Загрузка датасета: {dataset_path}")
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]
    print(f"   Задач: {len(dataset)}")

    print(f"\n🤖 Загрузка модели: {MODEL_NAME}")
    start = time.time()
    model, tokenizer = load(MODEL_NAME)
    print(f"   Загружена за {time.time() - start:.1f} сек")

    results = []
    total_time = 0
    corrected_count = 0

    print(f"\n{'='*60}")
    print("ВАЛИДАЦИЯ: Что нужно найти")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        verifiable_answer = item["verifiable_answer"]
        extracted_condition = item[CONDITION_COLUMN]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")
        print(f"   Извлечённое условие: {extracted_condition}")

        task_start = time.time()

        # --- Вызов 1: просим модель порассуждать о правильности ---
        think_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Задача: {question}\n\n"
                f"Извлечённый ответ (что нужно найти): {extracted_condition}\n\n"
                f"{VALIDATE_THINK_INSTRUCTION}"
            )},
        ]
        prompt_think = tokenizer.apply_chat_template(
            think_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        thinking = generate(
            model, tokenizer,
            prompt=prompt_think,
            max_tokens=THINK_MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=0.6),
        )
        thinking_clean = clean_response(thinking)

        print(f"   💭 Рассуждение: {thinking_clean[:120]}...")

        # --- Вызов 2: просим вынести вердикт ---
        verdict_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Задача: {question}\n\n"
                f"Извлечённый ответ (что нужно найти): {extracted_condition}\n\n"
                f"{VALIDATE_THINK_INSTRUCTION}"
            )},
            {"role": "assistant", "content": thinking_clean},
            {"role": "user", "content": (
                f"{VALIDATE_VERDICT_INSTRUCTION}"
            )},
        ]
        prompt_verdict = tokenizer.apply_chat_template(
            verdict_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        verdict_response = generate(
            model, tokenizer,
            prompt=prompt_verdict,
            max_tokens=VERDICT_MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=0.0),
        )
        verdict_clean = clean_response(verdict_response)
        correct = is_correct_verdict(verdict_clean)

        print(f"   🔍 Вердикт: {verdict_clean}")

        final_condition = extracted_condition

        if not correct:
            # --- Вызов 3: просим исправить условие ---
            correct_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Задача: {question}\n\n"
                    f"Извлечённый ответ (что нужно найти): {extracted_condition}\n\n"
                    f"{VALIDATE_THINK_INSTRUCTION}"
                )},
                {"role": "assistant", "content": thinking_clean},
                {"role": "user", "content": (
                    f"{VALIDATE_VERDICT_INSTRUCTION}"
                )},
                {"role": "assistant", "content": verdict_clean},
                {"role": "user", "content": (
                    f"{VALIDATE_CORRECT_INSTRUCTION}"
                )},
            ]
            prompt_correct = tokenizer.apply_chat_template(
                correct_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            correct_response = generate(
                model, tokenizer,
                prompt=prompt_correct,
                max_tokens=CORRECT_MAX_TOKENS,
                verbose=False,
                sampler=make_sampler(temp=0.0),
            )
            final_condition = extract_condition(correct_response)
            corrected_count += 1
            print(f"   ✏️  Исправлено: {final_condition}")
        else:
            print(f"   ✅ Условие верно: {final_condition}")

        task_time = time.time() - task_start
        total_time += task_time
        print(f"   Время: {task_time:.1f} сек")

        results.append({
            "question": question,
            CONDITION_COLUMN: final_condition,
            "verifiable_answer": verifiable_answer
        })

    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"Всего задач:    {len(dataset)}")
    print(f"Исправлено:     {corrected_count}")
    print(f"Общее время:    {total_time:.1f} сек")
    print(f"Среднее время:  {total_time/len(dataset):.1f} сек/задача")
    print(f"{'='*60}")

    dataset_name = Path(dataset_path).stem
    output_file = f"./data/validation/statements_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация того, что нужно найти, из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
