import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения того, что нужно найти
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

# Промпт для первого вызова — просим порассуждать
THINK_INSTRUCTION = "Сначала порассуждай вслух: что именно нужно найти в задаче — длину или угол, как правильно записать по формату (1 XY или 2 XYZ или 3 XY ZT). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 400  # Токены для рассуждения (300-400)
ANSWER_MAX_TOKENS = 30  # Токены для финального ответа (формат строго: "1 XY" или "2 XYZ")


def load_dataset(csv_path: str) -> list[dict]:
    """Загружает датасет из CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def extract_condition(response: str) -> str:
    """Извлекает условие из ответа модели."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.replace('<|im_end|>', '')
    response = response.replace('<|eot_id|>', '')
    response = response.replace('Найти:', '')
    response = response.strip()

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return response


def run_naming(dataset_path: str, limit: int | None = None):
    """Запускает извлечение того, что нужно найти, на датасете."""

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
    thinkings = []
    total_time = 0

    print(f"\n{'='*60}")
    print("ЗАПУСК надо доказать")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        verifiable_answer = item["verifiable_answer"]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")

        task_start = time.time()

        # --- Вызов 1: просим модель порассуждать ---
        think_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Задача: {question}\n\n{THINK_INSTRUCTION}"},
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
        thinking_clean = re.sub(r'<think>.*?</think>', '', thinking, flags=re.DOTALL)
        thinking_clean = thinking_clean.replace('<|im_end|>', '').replace('<|eot_id|>', '').strip()

        print(f"   💭 Рассуждение: {thinking_clean[:120]}...")

        # --- Вызов 2: финальный ответ, передаём рассуждение в контекст ---
        answer_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Задача: {question}\n\n{THINK_INSTRUCTION}"},
            {"role": "assistant", "content": thinking_clean},
            {"role": "user", "content": "Теперь дай финальный ответ строго по формату: напиши \"Найти: \" и сразу ответ. Ничего больше."},
        ]
        prompt_answer = tokenizer.apply_chat_template(
            answer_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer_response = generate(
            model, tokenizer,
            prompt=prompt_answer,
            max_tokens=ANSWER_MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=0.0),
        )
        condition = extract_condition(answer_response)

        task_time = time.time() - task_start
        total_time += task_time

        print(f"   ✅ Условие: {condition}...")
        print(f"   Время: {task_time:.1f} сек")

        results.append({
            "question": question,
            "condition": condition,
            "verifiable_answer": verifiable_answer
        })
        thinkings.append({
            "question": question,
            "thinking": thinking_clean,
            "verifiable_answer": verifiable_answer
        })

    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"Всего задач:   {len(dataset)}")
    print(f"Общее время:   {total_time:.1f} сек")
    print(f"Среднее время: {total_time/len(dataset):.1f} сек/задача")
    print(f"{'='*60}")

    dataset_name = Path(dataset_path).stem

    output_file = f"./data/statements_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/statements_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Извлечение того, что нужно найти, из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
