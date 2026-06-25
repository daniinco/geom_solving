import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения перпендикулярности и параллельности (из preprocessing3/get_perp.py)
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

THINK_INSTRUCTION = "Сначала порассуждай вслух: какие условия перпендикулярности и параллельности есть в задаче, явные и неявные (из фигур, углов, высот). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий перпендикулярности и параллельности и уже извлечённые условия.
Порассуждай по чеклисту:
1. Перечисли все условия перп/парал из задачи (явные + из фигур + из прямых углов + из высот)
2. Проверь каждое: правильный формат "AB perp CD" / "AB parallel CD"?
3. Нет ли лишних условий?
4. Не пропущены ли: параллельность из фигур (параллелограмм, прямоугольник, трапеция), перпендикулярность из прямых углов, высот?
Рассуждение пиши свободно, не более 350-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий перпендикулярности и параллельности, извлечённые условия и рассуждение о правильности.
Ответь СТРОГО одним словом или двумя словами — ничего больше, никаких знаков препинания:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий перпендикулярности и параллельности, извлечённые условия, рассуждение о правильности и вердикт о том, что условия извлечены неправильно.
Извлеки условия правильно. Сразу напиши условия строго по формату (напиши "Условия: " и сразу условия). Если условий нет — выведи только: ---. Не рассуждай."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 150  # Токены для исправленного условия

CONDITION_COLUMN = "perp_condition"


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
    response = response.replace('Условия:', '')
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
    """Запускает валидацию перпендикулярности/параллельности на датасете."""

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
    print("ВАЛИДАЦИЯ: Перпендикулярность и параллельность")
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
                f"Извлечённые условия перпендикулярности/параллельности: {extracted_condition}\n\n"
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
                f"Извлечённые условия перпендикулярности/параллельности: {extracted_condition}\n\n"
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
                    f"Извлечённые условия перпендикулярности/параллельности: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/perp_conds_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация условий перпендикулярности/параллельности из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
