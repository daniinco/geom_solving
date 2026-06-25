import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для поиска концикличных точек (из preprocessing3/get_col.py)
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

THINK_INSTRUCTION = "Сначала порассуждай вслух: есть ли в задаче четвёрки концикличных точек и почему. Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску концикличных точек и уже найденные четвёрки.
Порассуждай по чеклисту:
1. Перечисли все четвёрки концикличных точек из задачи (вписанные, прямоугольники, равнобедренные трапеции, два прямых угла)
2. Проверь каждую четвёрку: ровно 4 буквы? Все точки действительно на одной окружности?
3. Нет ли ложных четвёрок (параллелограмм, ромб — не вписаны; центр окружности не на окружности)?
4. Не пропущены ли четвёрки (прямоугольник, равнобедренная трапеция, два прямых угла на один отрезок)?
Рассуждение пиши свободно, не более 350-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску концикличных точек, найденные четвёрки и рассуждение о правильности.
Ответь СТРОГО одним словом или двумя словами — ничего больше, никаких знаков препинания:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по поиску концикличных точек, найденные четвёрки, рассуждение о правильности и вердикт о том, что условия извлечены неправильно.
Найди концикличные точки правильно. Сразу напиши условия строго по формату (напиши "Условия: " и сразу условия). Если четвёрок нет — выведи только: -. Не рассуждай."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 100  # Токены для исправленного условия

CONDITION_COLUMN = "cols_condition"


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
    """Запускает валидацию концикличных точек на датасете."""

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
    print("ВАЛИДАЦИЯ: Концикличные точки")
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
                f"Найденные четвёрки концикличных точек: {extracted_condition}\n\n"
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
                f"Найденные четвёрки концикличных точек: {extracted_condition}\n\n"
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
                    f"Найденные четвёрки концикличных точек: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/cols_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация концикличных точек из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
