import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для поиска концикличных точек
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только условия строго по формату, который я тебе скажу. Напиши "Условия: " и сразу пиши условия и после этого ничего не пиши.
Найди все четвёрки точек, лежащих на одной окружности.

КОГДА 4 ТОЧКИ НА ОДНОЙ ОКРУЖНОСТИ:
1. "Вписан в окружность": ABCD вписан → ABCD
2. "Описана окружность" + точка на ней: около треугольника ABC описана, D на окружности → ABCD
3. Хорды одной окружности: хорды AB и CD → ABCD
4. Прямоугольник: всегда вписан → вершины
5. Равнобедренная трапеция: всегда вписана → вершины
6. Два прямых угла на один отрезок: угол AXB=90 и угол AYB=90 → AXYB (окружность на диаметре AB)
7. Две высоты треугольника: высоты BH и CK в треугольнике ABC → BKHC
8. Сумма противоположных углов = 180°

НЕ ЯВЛЯЮТСЯ КОНЦИКЛИЧНЫМИ:
- Произвольный параллелограмм, ромб (не вписаны)
- Произвольная трапеция (только равнобедренная)
- Центр окружности + точки на окружности (центр не на окружности)
- Середина диаметра = центр, не на окружности
- Прямой угол + середина гипотенузы: середина = центр, не на окружности
- Только 3 точки на окружности без четвёртой

ФОРМАТ:
- Четвёрки: 4 буквы слитно, через запятую если несколько
- Нет четвёрок → прочерк: -
- Без пояснений

ПРИМЕРЫ:

Задача: Четырёхугольник ABCD вписан в окружность. Найдите угол A.
Условия: ABCD

Задача: Около треугольника ABC описана окружность. Прямая через A пересекает окружность в точке D.
Условия: ABCD

Задача: В прямоугольнике ABCD диагонали пересекаются в точке O.
Условия: ABCD

Задача: В равнобедренной трапеции ABCD (AD параллельна BC) проведена диагональ AC.
Условия: ABCD

Задача: Хорды AB и CD окружности пересекаются в точке E.
Условия: ABCD

Задача: В треугольнике ABC проведены высоты BH и CK.
Условия: BKHC

Задача: В окружности проведены хорды AB и CD. На дуге AB отмечена точка E.
Условия: ABCD,ABEC

Задача: В треугольнике ABC угол C равен 90°, M — середина AB.
Условия: -

Задача: В параллелограмме ABCD угол A равен 60°.
Условия: -

Задача: В треугольнике ABC проведена высота BH.
Условия: -

Задача: В треугольнике ABC проведена медиана AM и биссектриса BK.
Условия: -

Задача: В ромбе ABCD диагонали пересекаются в точке O.
Условия: -"""

# Промпт для первого вызова — просим порассуждать
THINK_INSTRUCTION = "Сначала порассуждай вслух: есть ли в задаче четвёрки концикличных точек и почему. Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 400   # Токены для рассуждения (300-400)
ANSWER_MAX_TOKENS = 100  # Токены для финального ответа


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
    """Запускает поиск концикличных точек на датасете."""

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
    print("ЗАПУСК Окружности")
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

    output_file = f"./data/cols_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/cols_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Поиск концикличных точек в геометрических задачах")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
