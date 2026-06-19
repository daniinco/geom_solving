#!/usr/bin/env python3
"""
Baseline LLM Evaluator

Evaluates geometry problems using TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8
via a two-step pipeline:
  1. Ask the model to solve the problem (thinking/solution step)
  2. Ask the model (with the solution in context) to give only the final numeric answer

Usage:
    python3 baseline.py --dataset data/t_geom_lettered.csv --limit 5
    python3 baseline.py --dataset data/t_geom_lettered.csv
"""

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Model
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"

# ============================================================================
# Generation parameters
# ============================================================================
SOLVE_MAX_TOKENS = 16384  # tokens for the solution step
ANSWER_MAX_TOKENS = 64   # tokens for the answer extraction step

# ============================================================================
# Prompts
# ============================================================================
SYSTEM_PROMPT = """Ты — решатель геометрических задач.
Твоя задача — решать геометрические задачи и давать точный числовой ответ."""

SOLVE_INSTRUCTION = "Реши следующую геометрическую задачу. Покажи полное решение с пояснениями."

ANSWER_INSTRUCTION = "Теперь дай финальный числовой ответ строго в формате: \"Ответ: число\". Ответ должен быть целым числом или десятичной дробью. Никаких пояснений, только строка с ответом."


# ============================================================================
# Helpers
# ============================================================================

def extract_number(text: str) -> float | None:
    """Extract a numeric answer from model output."""
    # Strip special tokens
    text = text.replace('<|im_end|>', '').replace('<|eot_id|>', '').strip()

    # Remove everything after </think> if present
    if '</think>' in text:
        text = text.split('</think>')[0]

    # Try "Ответ: <number>"
    answer_match = re.search(r'[Оо]твет[:\s]+(-?\d+[\.,]?\d*)', text)
    if answer_match:
        return float(answer_match.group(1).replace(',', '.'))

    # Try \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        content = boxed_match.group(1)
        frac_match = re.search(r'(-?)\\d?frac\{(-?\d+)\}\{(\d+)\}', content)
        if frac_match:
            sign = -1 if frac_match.group(1) == '-' else 1
            num = int(frac_match.group(2))
            den = int(frac_match.group(3))
            if den != 0:
                return sign * num / den
        num_match = re.search(r'-?\d+[\.,]?\d*', content)
        if num_match:
            return float(num_match.group().replace(',', '.'))

    # Try sqrt(n)/d
    sqrt_frac_match = re.search(r'sqrt\((\d+)\)\s*/\s*(\d+)', text)
    if sqrt_frac_match:
        num, den = int(sqrt_frac_match.group(1)), int(sqrt_frac_match.group(2))
        if den != 0: 
            return math.sqrt(num) / den

    # Try plain fraction a/b
    fraction_match = re.search(r'(-?\d+)\s*/\s*(\d+)', text)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        if den != 0:
            return num / den

    # Try any decimal/integer
    number_match = re.search(r'-?\d+[\.,]?\d*', text)
    if number_match:
        return float(number_match.group().replace(',', '.'))

    return None


def compare_answers(predicted: float | None, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Compare predicted answer against ground truth (supports multiple values separated by , or ;)."""
    if predicted is None:
        return False

    gt_values = []
    for part in ground_truth.replace(';', ',').split(','):
        part = part.strip()
        if '/' in part:
            try:
                num, den = part.split('/')
                gt_values.append(float(num) / float(den))
            except Exception:
                pass
        else:
            try:
                gt_values.append(float(part))
            except Exception:
                pass

    return any(abs(predicted - gt) <= tolerance for gt in gt_values)


def load_dataset(csv_path: str, limit: int | None = None) -> list[dict]:
    """Load dataset from CSV, optionally limiting the number of rows."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            data.append(row)
    return data


def print_statistics(results: list[dict]):
    """Print summary statistics."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100 if total > 0 else 0
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / total if total > 0 else 0

    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"Всего задач:   {total}")
    print(f"Правильных:    {correct}")
    print(f"Accuracy:      {accuracy:.1f}%")
    print(f"Общее время:   {total_time:.1f} сек")
    print(f"Среднее время: {avg_time:.1f} сек/задача")
    print(f"{'='*60}")


# ============================================================================
# Main evaluation loop
# ============================================================================

def run_evaluation(dataset_path: str, limit: int | None = None):
    """Run two-step LLM evaluation on the dataset."""

    print(f"📂 Загрузка датасета: {dataset_path}")
    dataset = load_dataset(dataset_path, limit)
    print(f"   Задач: {len(dataset)}")

    print(f"\n🤖 Загрузка модели: {MODEL_NAME}")
    load_start = time.time()
    model, tokenizer = load(MODEL_NAME)
    print(f"   Загружена за {time.time() - load_start:.1f} сек")

    results = []
    correct_count = 0

    print(f"\n{'='*60}")
    print("ЗАПУСК ОЦЕНКИ")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["verifiable_answer"]

        print(f"\n[{i+1}/{len(dataset)}] {question[:80]}...")

        task_start = time.time()

        # ── Step 1: solve ────────────────────────────────────────────────────
        solve_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Задача: {question}\n\n{SOLVE_INSTRUCTION}"},
        ]
        prompt_solve = tokenizer.apply_chat_template(
            solve_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        solution = generate(
            model, tokenizer,
            prompt=prompt_solve,
            max_tokens=SOLVE_MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=0.0),
        )
        solution_clean = re.sub(r'<think>.*?</think>', '', solution, flags=re.DOTALL)
        solution_clean = solution_clean.replace('<|im_end|>', '').replace('<|eot_id|>', '').strip()

        print(f"   💭 Решение: {solution_clean[:120]}...")

        # ── Step 2: extract answer ───────────────────────────────────────────
        answer_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Задача: {question}\n\n{SOLVE_INSTRUCTION}"},
            {"role": "assistant", "content": solution_clean},
            {"role": "user", "content": ANSWER_INSTRUCTION},
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
        answer_clean = answer_response.replace('<|im_end|>', '').replace('<|eot_id|>', '').strip()

        task_time = time.time() - task_start

        predicted = extract_number(answer_clean)
        is_correct = compare_answers(predicted, ground_truth)

        if is_correct:
            correct_count += 1
            status_sym = "✅"
        else:
            status_sym = "❌"

        print(f"   Предсказано: {predicted}, Правильно: {ground_truth} {status_sym}")
        print(f"   Ответ модели: {answer_clean[:120]}")
        print(f"   Время: {task_time:.1f} сек")

        results.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "solution": solution_clean,
            "answer_response": answer_clean,
            "predicted": predicted,
            "correct": is_correct,
            "time": task_time,
        })

    # ── Statistics ───────────────────────────────────────────────────────────
    print_statistics(results)

    # ── Save results ─────────────────────────────────────────────────────────
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(dataset_path).stem
    output_file = output_dir / f"baseline_{dataset_name}_{timestamp}.json"

    total = len(dataset)
    accuracy = correct_count / total if total > 0 else 0

    output_data = {
        "timestamp": timestamp,
        "dataset": dataset_path,
        "model": MODEL_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "solve_instruction": SOLVE_INSTRUCTION,
        "answer_instruction": ANSWER_INSTRUCTION,
        "solve_max_tokens": SOLVE_MAX_TOKENS,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "total": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "total_time": sum(r['time'] for r in results),
        "results": results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Результаты сохранены: {output_file}")

    # ── Save solutions CSV (solutions + answer responses) ────────────────────
    solutions_dir = Path("data/baseline_solutions")
    solutions_dir.mkdir(parents=True, exist_ok=True)

    solutions_file = solutions_dir / f"solutions_{dataset_name}_{timestamp}.csv"
    with open(solutions_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "question", "ground_truth",
            "solution", "answer_response", "predicted", "correct", "time",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"📝 Решения сохранены:   {solutions_file}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline LLM evaluation for geometry problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 baseline.py --dataset data/t_geom_lettered.csv --limit 5
  python3 baseline.py --dataset data/t_geom_lettered.csv
        """,
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to the CSV dataset file (must have "question" and "verifiable_answer" columns)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of problems to evaluate (default: all)',
    )

    args = parser.parse_args()
    run_evaluation(args.dataset, args.limit)


if __name__ == "__main__":
    main()
