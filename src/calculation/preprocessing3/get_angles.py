import argparse
import csv
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

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


MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"

THINK_MAX_TOKENS = 400   # Токены для рассуждения (300-400)
ANSWER_MAX_TOKENS = 120  # Токены для финального ответа


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
    """Запускает извлечение угловых условий на датасете."""

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
    print("ЗАПУСК Углы")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        verifiable_answer = item["verifiable_answer"]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")

        task_start = time.time()

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

    output_file = f"./data/angles_{dataset_name}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "condition", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Условия сохранены: {output_file}")

    Path("./data/thinkings").mkdir(parents=True, exist_ok=True)
    thinking_file = f"./data/thinkings/angles_{dataset_name}.csv"
    with open(thinking_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "thinking", "verifiable_answer"])
        writer.writeheader()
        writer.writerows(thinkings)
    print(f"💭 Рассуждения сохранены: {thinking_file}")


def main():
    parser = argparse.ArgumentParser(description="Извлечение угловых условий из геометрических задач")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к CSV файлу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число задач")

    args = parser.parse_args()
    run_naming(args.dataset, args.limit)


if __name__ == "__main__":
    main()
