import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для поиска точек на одной прямой (из preprocessing3/get_lines.py)
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать условия строго по формату, который я тебе скажу. Напиши "Условия: " и сразу пиши условия и после этого ничего не пиши.
Извлеки из условия задачи все тройки лежащих на одной прямой точек.

ПРАВИЛО: если из условия следует, что некая точка X лежит на прямой через некие P и Q — запиши PXQ.

Типичные случаи:
- "X на стороне/отрезке PQ" → PXQ
- "X — середина PQ" → PXQ
- "X делит PQ в отношении ..." → PXQ
- "PQ и RS пересекаются в точке X" → PXQ,RXS
- "диагонали AC и BD пересекаются в O" → AOC,BOD
- "высота/биссектриса/медиана из A на BC, основание H" → BHC
- "X на продолжении PQ за точку Q" → PQX

Когда тройки точно НЕ на одной прямой:
- "Угол RKL равен 90" → точки RKL точно не на одной прямой
- "Четырехугольник KLBH вписан в окружность ..." → точки KLBH точно не на одной прямой
- "Дан треугольник ABC ..." → точки ABC точно не на одной прямой

НЕ добавляй тройку, если в условии нет явного указания, что точка лежит на прямой/стороне/отрезке через эти точки

ФОРМАТ:
- Тройка = ровно 3 буквы
- Несколько троек через запятую без пробелов
- Нет троек → -
- Никаких пояснений, только ответ

Задача: В параллелограмме ABCD точка L на стороне AD.
Условия: ALD

Задача: В треугольнике ABC точка P на AC, точка Q на BP.
Условия: APC,BQP

Задача: В треугольнике ABC угол A равен 45°, угол ABC равен 90.
Условия: -

Задача: Диагонали AC и BD параллелограмма ABCD пересекаются в точке O.
Условия: AOC,BOD

Задача: В треугольнике ABC M — середина BC, K — точка пересечения медианы AM с биссектрисой BL, L на AC.
Условия: BMC,ALC,AKM,BKL"""

THINK_INSTRUCTION = "Сначала порассуждай вслух: какие точки лежат на одной прямой в этой задаче и почему. Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску точек на одной прямой и уже найденные тройки.
Порассуждай строго по чеклисту:

1. ПОЛНОТА: Перечисли ВСЕ тройки точек, которые явно лежат на одной прямой:
   - "X на стороне/отрезке PQ" → PXQ
   - "X — середина PQ" → PXQ
   - "X делит PQ в отношении ..." → PXQ
   - "PQ и RS пересекаются в точке X" → PXQ и RXS
   - "диагонали AC и BD пересекаются в O" → AOC и BOD
   - "высота/биссектриса/медиана из A на BC, основание H" → BHC
   - "X на продолжении PQ за точку Q" → PQX

2. ФОРМАТ: Каждая тройка — ровно 3 буквы. Средняя буква — точка, лежащая МЕЖДУ двумя другими (или на прямой через них).

3. ЛОЖНЫЕ ТРОЙКИ — проверь, нет ли в найденных:
   - Вершины треугольника (ABC, BCD и т.п.) — они точно НЕ на одной прямой
   - Вершины вписанного четырёхугольника — НЕ на одной прямой
   - Точки, для которых нет явного указания в условии, что они коллинеарны
   - Угол XYZ=90 → точки X,Y,Z точно НЕ на одной прямой

4. ПРОПУСКИ: Сравни свой список из п.1 с найденными тройками — что пропущено?

5. ИТОГ: Совпадает ли найденное с тем, что должно быть?

ТИПИЧНЫЕ ОШИБКИ:
- Добавлены вершины треугольника как коллинеарные
- Пропущена тройка для точки пересечения двух отрезков (нужны ДВЕ тройки)
- Средняя буква в тройке — не та точка, которая лежит между двумя другими
- Пропущена тройка для середины отрезка

Рассуждение пиши свободно, не более 400-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску точек на одной прямой, найденные тройки и рассуждение о правильности.
На основе рассуждения вынеси вердикт: если найденные тройки ПОЛНОСТЬЮ совпадают с тем, что должно быть по правилам (ничего не пропущено, ничего лишнего, формат верен) — ответь "Правильно". Если есть ХОТЬ ОДНО отличие — ответь "Неправильно".
Ответь СТРОГО одним словом — ничего больше, никаких знаков препинания, никаких пояснений:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску точек на одной прямой, найденные тройки, рассуждение о правильности и вердикт о том, что условия извлечены неправильно.
Найди тройки точек на одной прямой заново, строго по правилам из инструкции.
Напиши ТОЛЬКО: Условия: <тройки через запятую без пробелов>
Если троек нет — напиши ТОЛЬКО: Условия: -
Никаких пояснений, никакого рассуждения, только одна строка."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 150  # Токены для исправленного условия

CONDITION_COLUMN = "lines_condition"


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
    response = re.sub(r'.*Условия:', '', response, flags=re.DOTALL)
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
    """Запускает валидацию точек на одной прямой на датасете."""

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
    print("ВАЛИДАЦИЯ: Точки на одной прямой")
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
                f"Найденные тройки точек на одной прямой: {extracted_condition}\n\n"
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
                f"Найденные тройки точек на одной прямой: {extracted_condition}\n\n"
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
                    f"Найденные тройки точек на одной прямой: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/lines_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация точек на одной прямой из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
