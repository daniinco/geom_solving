import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения угловых условий (из preprocessing3/get_angles.py)
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только условия строго по формату, который я тебе скажу. Напиши "Условия: " и сразу пиши условия и после этого ничего не пиши.

ФОРМАТ ВЫВОДА:
Условия через запятую. Если угловых условий нет — выведи только: -

ТИПЫ УСЛОВИЙ:
- Угол равен градусам: ABC=60
- Угол равен углу: ABC=DEF
- Кратное соотношение: ABC=2DEF

ИМЕНОВАНИЕ УГЛОВ:
Угол ВСЕГДА — три буквы, вершина — СРЕДНЯЯ буква.
В многоугольнике вершину окружают соседние вершины по порядку обхода:
- "Угол A" в треугольнике ABC → BAC
- "Угол B" в треугольнике ABC → ABC
- "Угол C" в треугольнике ABC → BCA
- "Угол K" в ромбе KLMN → NKL
- "Угол L" в ромбе KLMN → KLM
- "Угол P" в четырёхугольнике PQRS → SPQ
- "Угол R" в четырёхугольнике PQRS → QRS
- "Угол S" в четырёхугольнике PQRS → RSP

КОНВЕРТАЦИЯ ГЕОМЕТРИЧЕСКИХ ТЕРМИНОВ:

Перпендикулярность:
- "QS перп PS" или "QS ⊥ PS" → QSP=90 (вершина — общая точка)
- "высота BH на AC" → BHA=90 или BHC=90

Прямоугольный треугольник:
- "прямоугольный треугольник XYZ с гипотенузой XZ" → XYZ=90
- "прямой угол при Y" → XYZ=90

Равнобедренный треугольник:
- "равнобедренный треугольник XYZ, XY=XZ" → XYZ=XZY
- "равнобедренный треугольник XYZ с основанием YZ" → XYZ=XZY

Равнобедренный прямоугольный треугольник:
- "равнобедренный прямоугольный треугольник BCM с гипотенузой BC" → BMC=90,MBC=MCB

Равносторонний треугольник:
- "равносторонний треугольник ABC" → BAC=60,ABC=60,BCA=60

Биссектриса:
- "BD — биссектриса угла B в треугольнике ABC" → ABD=DBC

Равные стороны в треугольнике:
- "в треугольнике ABC AB=BC" → BAC=BCA

НЕ ИЗВЛЕКАЙ:
Длины, площади, радиусы, координаты, параллельность, касательность, равенство сторон вне треугольников.

ПРИМЕРЫ:

Задача: В выпуклом четырёхугольнике ABCD стороны AB, BC и CD равны, M — середина AD. Известно, что угол BMC равен 90°. Найдите угол между диагоналями.
Условия: BMC=90

Задача: Окружность ω касается сторон AD, AB и BC параллелограмма ABCD. Диагональ AC пересекает ω в точках P и Q. Найдите площадь, если AP=3, PQ=9, QC=16.
Условия: -

Задача: В треугольнике ABC угол C равен 75°, а угол B равен 60°. Вершина M равнобедренного прямоугольного треугольника BCM с гипотенузой BC лежит внутри ABC. Найдите угол MAC.
Условия: BCA=75,ABC=60,BMC=90,MBC=MCB

Задача: В ромбе KLMN угол K равен 110°. Диагонали пересекаются в точке O. Найдите угол LOM.
Условия: NKL=110

Задача: В четырёхугольнике PQRS угол P равен 80°, угол R вдвое больше угла S. Диагональ QS перпендикулярна стороне PS. Найдите угол Q.
Условия: SPQ=80,QRS=2RSP,QSP=90

Задача: В треугольнике ABC проведена биссектриса BD. Угол A равен 50°, угол C равен 70°. Найдите угол BDC.
Условия: BAC=50,BCA=70,ABD=DBC

Задача: В равностороннем треугольнике ABC точка D — середина BC. Найдите угол ADC.
Условия: BAC=60,ABC=60,BCA=60

Задача: В треугольнике ABC AB=BC, угол B равен 40°. Найдите угол A.
Условия: ABC=40,BAC=BCA"""

THINK_INSTRUCTION = "Сначала порассуждай вслух: какие угловые условия есть в задаче, как их правильно записать по формату. Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению угловых условий и уже извлечённые условия.
Порассуждай строго по чеклисту:

1. ПОЛНОТА: Перечисли ВСЕ угловые условия из задачи — явные (угол ABC=60) и неявные:
   - Биссектриса BD угла B → ABD=DBC
   - Равнобедренный треугольник XYZ с основанием YZ → XYZ=XZY
   - Равносторонний треугольник ABC → BAC=60,ABC=60,BCA=60
   - Равнобедренный прямоугольный треугольник BCM с гипотенузой BC → BMC=90,MBC=MCB
   - Перпендикулярность QS⊥PS → QSP=90 (вершина — общая точка)
   - Высота BH на AC → BHA=90 (или BHC=90)
   - Равные стороны в треугольнике AB=BC → BAC=BCA

2. ФОРМАТ: Каждое условие — ровно 3 буквы, вершина угла ВСЕГДА в середине.
   Угол A в треугольнике ABC → BAC (не ABC, не ACB).
   Угол K в ромбе KLMN → NKL (соседние вершины по порядку обхода).

3. ЛИШНИЕ УСЛОВИЯ: Нет ли в извлечённых условиях того, чего НЕ должно быть:
   - длин отрезков (AB=5), площадей, радиусов
   - параллельности, касательности
   - равенства сторон вне треугольников

4. ПРОПУСКИ: Сравни свой список из п.1 с извлечёнными условиями — что пропущено?

5. ИТОГ: Совпадает ли извлечённое с тем, что должно быть?

ТИПИЧНЫЕ ОШИБКИ:
- Угол A в треугольнике ABC записан как ABC вместо BAC
- Для равностороннего треугольника извлечён только один угол вместо трёх
- Пропущено условие равенства углов из равных сторон (AB=BC → BAC=BCA)
- Добавлены условия на длины или параллельность

Рассуждение пиши свободно, не более 400-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению угловых условий, извлечённые условия и рассуждение о правильности.
На основе рассуждения вынеси вердикт: если извлечённые условия ПОЛНОСТЬЮ совпадают с тем, что должно быть по правилам (ничего не пропущено, ничего лишнего, формат верен) — ответь "Правильно". Если есть ХОТЬ ОДНО отличие — ответь "Неправильно".
Ответь СТРОГО одним словом — ничего больше, никаких знаков препинания, никаких пояснений:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению угловых условий, извлечённые условия, рассуждение о правильности и вердикт о том, что условия извлечены неправильно.
Извлеки угловые условия заново, строго по правилам из инструкции.
Напиши ТОЛЬКО: Условия: <условия через запятую без пробелов>
Если угловых условий нет — напиши ТОЛЬКО: Условия: -
Никаких пояснений, никакого рассуждения, только одна строка."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 120  # Токены для исправленного условия

CONDITION_COLUMN = "angles_condition"


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
    """Запускает валидацию угловых условий на датасете."""

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
    print("ВАЛИДАЦИЯ: Углы")
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
                f"Извлечённые угловые условия: {extracted_condition}\n\n"
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
                f"Извлечённые угловые условия: {extracted_condition}\n\n"
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
                    f"Извлечённые угловые условия: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/angles_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация угловых условий из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
