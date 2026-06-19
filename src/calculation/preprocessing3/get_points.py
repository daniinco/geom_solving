import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Промпт для извлечения именованных точек
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

# Промпт для первого вызова — просим порассуждать
THINK_INSTRUCTION = "Сначала порассуждай вслух: перечисли все именованные точки, которые встречаются в задаче, и убедись, что не пропустил ни одну. Рассуждение пиши свободно, не более 300-400 токенов."

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
    response = response.replace('Точки:', '')
    response = response.strip()

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return response


def run_naming(dataset_path: str, limit: int | None = None):
    """Запускает извлечение точек на датасете."""

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
    print("ЗАПУСК точки")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        verifiable_answer = item["verifiable_answer"]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")

        task_start = time.time()

        # --- Вызов 1: просим модель порассуждать ---
        think_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Условие:\n{question}\n\n{THINK_INSTRUCTION}"},
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
            {"role": "user", "content": f"Условие:\n{question}\n\n{THINK_INSTRUCTION}"},
            {"role": "assistant", "content": thinking_clean},
            {"role": "user", "content": "Теперь дай финальный ответ строго по формату: напиши \"Точки: \" и сразу список точек. Ничего больше."},
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

    output_file = f"./data/points_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/points_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Извлечение именованных точек из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
