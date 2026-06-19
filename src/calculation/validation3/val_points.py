import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения именованных точек (из preprocessing3/get_points.py)
# ============================================================================
SYSTEM_PROMPT = """Ты — анализатор геометрических задач.
Твоя единственная задача — прочитать текст геометрической задачи и выписать только точки строго по формату, который я тебе скажу. Напиши "Точки: " и сразу пиши условия и после этого ничего не пиши.
Извлеки из геометрической задачи все именованные точки (вершины, середины, центры, точки пересечения и любые другие точки, обозначенные заглавными латинскими буквами). Перечисли их через запятую без повторений. Не добавляй точки, которых нет в условии.

Пример 1:
Условие: В выпуклом четырёхугольнике ABCD стороны AB, BC и CD равны, M — середина AD. Известно, что угол BMC равен 90°. Найдите угол между диагоналями четырёхугольника ABCD.
Точки: A,B,C,D,M

Пример 2:
Условие: На сторонах единичного квадрата отметили точки K, L, M, N так, что KM параллельна двум сторонам квадрата, а LN — двум другим. KL отсекает треугольник периметра 1. Какова площадь треугольника, отсекаемого MN?
Точки: K,L,M,N

Пример 3:
Условие: На стороне AC треугольника ABC взяли такую точку D, что угол BDC равен углу ABC. Чему равно наименьшее возможное расстояние между центрами окружностей, описанных около треугольников ABC и ABD, если BC = 1?
Точки: A,B,C,D

Пример 4:
Условие: В треугольнике ABC проведены биссектрисы AA1 и BB1, пересекающиеся в точке I. Найдите угол ACB, если AI:IA1 = BB1:BI.
Точки: A,B,C,A1,B1,I

Пример 5:
Условие: Окружности с центрами O1 и O2 пересекаются в точках P и Q. Прямая через P пересекает первую окружность в точке A и вторую в точке B. Найдите угол AQB.
Точки: O1,O2,P,Q,A,B"""

THINK_INSTRUCTION = "Сначала порассуждай вслух: перечисли все именованные точки, которые встречаются в задаче, и убедись, что не пропустил ни одну. Рассуждение пиши свободно, не более 300-400 токенов."

# ============================================================================
# Промпты для валидации
# ============================================================================
VALIDATE_THINK_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению именованных точек и уже извлечённый список точек.
Порассуждай строго по чеклисту:

1. ПОЛНОТА: Прочитай задачу и выпиши ВСЕ обозначения точек — заглавные латинские буквы и буквы с индексами:
   - Простые: A, B, C, D, H, K, L, M, N, O, P, Q, R, S, T и т.д.
   - С цифрами: A1, B1, O1, O2, P1, Q2 и т.д.
   - Точки из ВОПРОСА (что нужно найти) тоже считаются — например, если ищем угол BDC, то D должна быть в списке
   - Центры окружностей, точки пересечения, середины — всё включается

2. ЛИШНИЕ ТОЧКИ: Нет ли в списке точек, которых нет в условии задачи?

3. ПРОПУСКИ: Сравни свой список из п.1 с извлечёнными точками — что пропущено?
   Особое внимание: точки с индексами (A1, B1), точки из вопроса задачи.

4. ФОРМАТ: Точки через запятую без пробелов, без повторений.

5. ИТОГ: Совпадает ли извлечённое с тем, что должно быть?

ТИПИЧНЫЕ ОШИБКИ:
- Пропущены точки с числовыми индексами (O1, A1, B1)
- Пропущены точки, упомянутые только в вопросе задачи (что нужно найти)
- Добавлены точки, которых нет в условии
- Повторения в списке

Рассуждение пиши свободно, не более 400-500 токенов."""

VALIDATE_VERDICT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению именованных точек, извлечённый список точек и рассуждение о правильности.
На основе рассуждения вынеси вердикт: если список точек ПОЛНОСТЬЮ совпадает с тем, что должно быть (ничего не пропущено, ничего лишнего, нет повторений) — ответь "Правильно". Если есть ХОТЬ ОДНО отличие — ответь "Неправильно".
Ответь СТРОГО одним словом — ничего больше, никаких знаков препинания, никаких пояснений:
Правильно
Неправильно"""

VALIDATE_CORRECT_INSTRUCTION = """Тебе дана геометрическая задача, инструкции по извлечению именованных точек, извлечённый список точек, рассуждение о правильности и вердикт о том, что список неправильный.
Извлеки точки заново, строго по правилам из инструкции.
Напиши ТОЛЬКО: Точки: <точки через запятую без пробелов>
Никаких пояснений, никакого рассуждения, только одна строка."""

# ============================================================================
# Параметры генерации
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"
THINK_MAX_TOKENS = 500    # Токены для рассуждения валидации (350-500)
VERDICT_MAX_TOKENS = 10   # Токены для вердикта
CORRECT_MAX_TOKENS = 100  # Токены для исправленного условия

CONDITION_COLUMN = "points_condition"


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
    response = response.replace('Точки:', '')
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
    """Запускает валидацию именованных точек на датасете."""

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
    print("ВАЛИДАЦИЯ: Именованные точки")
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
                f"Условие:\n{question}\n\n"
                f"Извлечённые точки: {extracted_condition}\n\n"
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
                f"Условие:\n{question}\n\n"
                f"Извлечённые точки: {extracted_condition}\n\n"
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
                    f"Условие:\n{question}\n\n"
                    f"Извлечённые точки: {extracted_condition}\n\n"
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
    output_file = f"./data/validation/points_{dataset_name}_validated.csv"
    Path("./data/validation").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", CONDITION_COLUMN, "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Результаты сохранены: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Валидация именованных точек из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу (all_conditions_*.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_validation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
