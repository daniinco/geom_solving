import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения условий на длины отрезков (из preprocessing3/get_segm_conds.py)
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

THINK_INSTRUCTION = "Сначала порассуждай вслух: какие условия на длины отрезков есть в задаче, включая неявные (середины, равные стороны, кратные соотношения). Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий на длины отрезков и уже извлечённые условия.
Порассуждай строго по чеклисту:

1. ПОЛНОТА: Перечисли ВСЕ условия на длины из задачи — явные и неявные:
   - Явные числа: "AB = 5" → AB=5
   - Равенства отрезков: "AB = CD" → AB=CD
   - Середина: "M — середина AB" → AM=MB
   - Равные стороны: "стороны AB, BC, CD равны" → AB=BC, BC=CD, AB=CD (все три пары!)
   - Кратные соотношения: "X вдвое/втрое больше Y" → X=2Y / X=3Y
   - "X в N раз больше Y" → X=NY
   - "X составляет половину Y" → X=0.5Y
   - Отношения: "AP:PB = 1:2" → AP=0.5PB (или аналогично)

2. ФОРМАТ: Только три допустимых формата — AB=CD, AB=5, AB=2CD. Коэффициент слитно перед отрезком.

3. ЛИШНИЕ УСЛОВИЯ — проверь, нет ли в извлечённых:
   - Угловых условий (ABC=60)
   - Условий перпендикулярности или параллельности
   - Площадей, радиусов, координат

4. ПРОПУСКИ: Сравни свой список из п.1 с извлечёнными условиями — что пропущено?
   Особое внимание: для "стороны X, Y, Z равны" нужны ВСЕ три пары (X=Y, Y=Z, X=Z).

5. ИТОГ: Совпадает ли извлечённое с тем, что должно быть?

ТИПИЧНЫЕ ОШИБКИ:
- Для "стороны AB, BC, CD равны" извлечены только AB=BC и BC=CD, но пропущено AB=CD
- Пропущено условие AM=MB для середины M отрезка AB
- "Вдвое больше" записано как X=2*Y вместо X=2Y
- Добавлены угловые условия или перпендикулярность

Рассуждение пиши свободно, не более 400-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий на длины отрезков, извлечённые условия и рассуждение о правильности.
На основе рассуждения вынеси вердикт: если извлечённые условия ПОЛНОСТЬЮ совпадают с тем, что должно быть по правилам (ничего не пропущено, ничего лишнего, формат верен) — ответь "Правильно". Если есть ХОТЬ ОДНО отличие — ответь "Неправильно".
Ответь СТРОГО одним словом — ничего больше, никаких знаков препинания, никаких пояснений:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению условий на длины отрезков, извлечённые условия, рассуждение о правильности и вердикт о том, что условия извлечены неправильно.
Извлеки условия на длины заново, строго по правилам из инструкции.
Напиши ТОЛЬКО: Условия: <условия через запятую без пробелов>
Если условий на длины нет — напиши ТОЛЬКО: Условия: -
Никаких пояснений, никакого рассуждения, только одна строка."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 140  # Токены для исправленного условия

CONDITION_COLUMN = "segm_condition"


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
    """Запускает валидацию условий на длины отрезков на датасете."""

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
    print("ВАЛИДАЦИЯ: Условия на длины отрезков")
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
                f"Задача:\n{question}\n\n"
                f"Извлечённые условия на длины отрезков: {extracted_condition}\n\n"
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
                f"Задача:\n{question}\n\n"
                f"Извлечённые условия на длины отрезков: {extracted_condition}\n\n"
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
                    f"Задача:\n{question}\n\n"
                    f"Извлечённые условия на длины отрезков: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/segm_conds_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация условий на длины отрезков из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
