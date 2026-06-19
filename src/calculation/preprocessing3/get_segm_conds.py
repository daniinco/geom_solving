import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения условий на длины отрезков
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только условия строго по формату, который я тебе скажу. Напиши "Условия: " и сразу пиши условия и после этого ничего не пиши.
Извлеки из текста ВСЕ условия на длины отрезков.

ТРИ ДОПУСТИМЫХ ФОРМАТА:
- отрезок = отрезок → AB=CD
- отрезок = число → MN=5
- отрезок = коэффициент·отрезок → KL=2BC (число слитно перед отрезком)

ПРАВИЛА ИЗВЛЕЧЕНИЯ:
- "Стороны X, Y, Z равны" → запиши все пары: X=Y, Y=Z, X=Z
- "M — середина AB" → AM=MB
- "X в N раз больше Y" или "X в N раз длиннее Y" → X=NY
- "X вдвое/втрое больше Y" → X=2Y / X=3Y
- "X составляет половину Y" → X=0.5Y
- Десятичные дроби допустимы: AB=4.5CD

ИГНОРИРУЙ: углы, перпендикулярность, параллельность, касания, вписанность — всё, что не про длины.

ФОРМАТ ОТВЕТА: условия через запятую без пробелов. Если условий на длины нет — напиши: -

Пример 1:
Задача: В выпуклом четырёхугольнике ABCD стороны AB, BC и CD равны, M — середина AD. Известно, что угол BMC равен 90°. Найдите угол между диагоналями четырёхугольника ABCD.
Условия: AB=BC,BC=CD,AB=CD,AM=MD

Пример 2:
Задача: Окружность касается сторон AD, AB и BC параллелограмма ABCD, в котором угол A меньше 90°. Диагональ AC пересекает окружность в точках P и Q, причём P лежит между A и Q. Найдите площадь параллелограмма ABCD, если AP = 3, PQ = 9, QC = 16.
Условия: AP=3,PQ=9,QC=16

Пример 3:
Задача: Точка M — середина стороны BC треугольника ABC, в котором AB = 19, AC = 36, BC = 21.
Условия: BM=MC,AB=19,AC=36,BC=21

Пример 4:
Задача: В треугольнике KBC сторона KC в два раза больше чем BC. Угол B=90°. Чему равен угол K?
Условия: KC=2BC

Пример 5:
Задача: В треугольнике ABC проведена биссектриса AL. Найдите угол ALC, если углы ABC и BAC равны 32° и 120° соответственно.
Условия: -"""

# Промпт для первого вызова — просим порассуждать
THINK_INSTRUCTION = "Сначала порассуждай вслух: какие условия на длины отрезков есть в задаче, включая неявные (середины, равные стороны, кратные соотношения). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 400   # Токены для рассуждения (300-400)
ANSWER_MAX_TOKENS = 140  # Токены для финального ответа


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
    response = response.replace('Условия:', '')
    response = response.strip()

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return response


def run_naming(dataset_path: str, limit: int | None = None):
    """Запускает извлечение условий на длины отрезков на датасете."""

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
    print("ЗАПУСК отрезки")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        verifiable_answer = item["verifiable_answer"]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")

        task_start = time.time()

        # --- Вызов 1: просим модель порассуждать ---
        think_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Задача:\n{question}\n\n{THINK_INSTRUCTION}"},
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
            {"role": "user", "content": f"Задача:\n{question}\n\n{THINK_INSTRUCTION}"},
            {"role": "assistant", "content": thinking_clean},
            {"role": "user", "content": "Теперь дай финальный ответ строго по формату: напиши \"Условия: \" и сразу условия. Ничего больше."},
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

        print(f"   ✅ Условие: {condition[:80]}...")
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

    output_file = f"./data/segm_conds_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/segm_conds_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Извлечение условий на длины отрезков из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
