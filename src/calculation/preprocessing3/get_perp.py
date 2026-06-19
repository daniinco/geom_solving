import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения перпендикулярности и параллельности
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только условия строго по формату, который я тебе скажу. Напиши "Условия: " и сразу пиши условия и после этого ничего не пиши.
Извлеки из задачи ВСЕ условия перпендикулярности и параллельности.

ФОРМАТ ВЫВОДА:
- Перпендикулярность: AB perp CD
- Параллельность: AB parallel CD
- Несколько условий — через запятую
- Нет условий — выведи ТОЛЬКО: ---

ПРАВИЛА:

[1] ПРЯМЫЕ УКАЗАНИЯ В ТЕКСТЕ:
- Символы перп, ⊥, "перпендикулярен", "перпендикулярно" → perp
- Символы парал, ||, "параллелен", "параллельно" → parallel
- "QC перп BC" → QC perp BC
- "MN парал AB" → MN parallel AB

[2] ПРЯМОЙ УГОЛ (90°):
- Угол при вершине X = 90° → две стороны, сходящиеся в X, перпендикулярны
- "угол B = 90°" в треугольнике ABC → AB perp BC
- "угол A = 90°" в четырёхугольнике ABCD → DA perp AB
- "угол BMC = 90°" → BM perp MC

[3] ВЫСОТА:
- "BH — высота треугольника ABC" → BH perp AC
- "CH — высота треугольника ABC" → CH perp AB
- "AH — высота треугольника ABC" → AH perp BC
- Общее правило: высота из вершины X на противолежащую сторону YZ → XH perp YZ

[4] ПЕРПЕНДИКУЛЯР / ПРОЕКЦИЯ:
- "Перпендикуляр из A к BC, основание H" → AH perp BC
- "Проекция точки A на прямую BC — точка H" → AH perp BC

[5] ФИГУРЫ — ПАРАЛЛЕЛЬНОСТЬ:
- Параллелограмм ABCD → AB parallel CD, BC parallel AD
- Прямоугольник ABCD → AB parallel CD, BC parallel AD
- Ромб ABCD → AB parallel CD, BC parallel AD
- Квадрат ABCD → AB parallel CD, BC parallel AD
- Трапеция ABCD → BC parallel AD (если не указано иначе)

[6] ФИГУРЫ — ПЕРПЕНДИКУЛЯРНОСТЬ (дополнительно к параллельности):
- Прямоугольник ABCD → добавить AB perp BC
- Квадрат ABCD → добавить AB perp BC
- Прямоугольная трапеция ABCD (угол A = 90°) → добавить AB perp AD
- Ромб: перпендикулярность диагоналей ТОЛЬКО если сказано в условии явно

[7] ПРЯМОУГОЛЬНЫЙ ТРЕУГОЛЬНИК:
- "Прямоугольный треугольник ABC, угол C = 90°" → AC perp BC
- "Прямоугольный треугольник ABC с гипотенузой AB" → прямой угол при C → AC perp BC
- Если прямой угол не указан явно, но указана гипотенуза XY — прямой угол при оставшейся вершине

[8] СРЕДНЯЯ ЛИНИЯ:
- "MN — средняя линия треугольника ABC, параллельная BC" → MN parallel BC
- "Средняя линия трапеции" — параллельна основаниям

[9] 3D — ПЕРПЕНДИКУЛЯРНОСТЬ К ПЛОСКОСТИ:
- "SA перп плоскости ABC" (A в плоскости) → SA perp AB, SA perp AC
- "SO перп (ABCD)" → SO perp OA, SO perp OB, SO perp OC, SO perp OD

ПРИМЕРЫ:

Задача: В выпуклом четырёхугольнике ABCD угол BMC равен 90°.
Условия: BM perp MC

Задача: В параллелограмме ABCD диагонали AC и BD перпендикулярны.
Условия: AB parallel CD, BC parallel AD, AC perp BD

Задача: Точка M — середина стороны BC треугольника ABC, AB = 19, AC = 36.
Условия: ---

Задача: В треугольнике ABC угол B = 90°.
Условия: AB perp BC

Задача: На отрезке BP точка Q такая, что QC перпендикулярна BC.
Условия: QC perp BC

Задача: Дан параллелограмм ABCD, угол D = 100°, BC = 24.
Условия: AB parallel CD, BC parallel AD

Задача: BH — высота треугольника ABC.
Условия: BH perp AC

Задача: В прямоугольнике ABCD диагонали пересекаются в точке O.
Условия: AB parallel CD, BC parallel AD, AB perp BC

Задача: В прямоугольном треугольнике ABC с гипотенузой AB проведена высота CH.
Условия: AC perp BC, CH perp AB

Задача: Трапеция ABCD с основаниями BC и AD, AB перпендикулярна AD.
Условия: BC parallel AD, AB perp AD

Задача: MN — средняя линия треугольника ABC, параллельная AC.
Условия: MN parallel AC

Задача: В пирамиде SABC ребро SA перпендикулярно плоскости ABC, AB = 5.
Условия: SA perp AB, SA perp AC

Задача: В ромбе ABCD сторона AB = 10, угол A = 60°.
Условия: AB parallel CD, BC parallel AD

Задача: В квадрате ABCD точка E — середина BC.
Условия: AB parallel CD, BC parallel AD, AB perp BC"""

# Промпт для первого вызова — просим порассуждать
THINK_INSTRUCTION = "Сначала порассуждай вслух: какие условия перпендикулярности и параллельности есть в задаче, явные и неявные (из фигур, углов, высот). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 400   # Токены для рассуждения (300-400)
ANSWER_MAX_TOKENS = 150  # Токены для финального ответа


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
    """Запускает извлечение перпендикулярности/параллельности на датасете."""

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
    print("ЗАПУСК перпендикулярность и параллельность")
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

    output_file = f"./data/perp_conds_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/perp_conds_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Извлечение перпендикулярности/параллельности из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
