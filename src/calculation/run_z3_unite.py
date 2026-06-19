#!/usr/bin/env python3
"""
Z3 Solver Runner with LLM Fallback

This script runs all generated Z3 solver files and evaluates their results against
the expected answers from the dataset. When a solver returns 'error' or 'no_solution',
the problem is sent to an LLM (same as baseline.py) for a second attempt.

Usage:
    python3 src/calculation/run_z3_unite.py --dataset data/t_geom_lettered_validated.csv --limit 51

The script will:
1. Find all solver files for the specified dataset
2. Run each solver and capture its output
3. If solver returns error/no_solution → run LLM fallback
4. Compare results with expected answers
5. Generate statistics showing correct answers split by source (Z3 vs LLM)
6. Save results to CSV
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# LLM / model settings (mirrored from baseline.py)
# ============================================================================
MODEL_NAME = "TheCluster/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-MLX-mxfp8"

SOLVE_MAX_TOKENS = 8192
ANSWER_MAX_TOKENS = 64

SYSTEM_PROMPT = """Ты — решатель геометрических задач.
Твоя задача — решать геометрические задачи и давать точный числовой ответ."""

SOLVE_INSTRUCTION = "Реши следующую геометрическую задачу. Покажи полное решение с пояснениями."

ANSWER_INSTRUCTION = (
    "Теперь дай финальный числовой ответ строго в формате: \"Ответ: число\". "
    "Ответ должен быть целым числом или десятичной дробью. "
    "Никаких пояснений, только строка с ответом."
)

# Statuses that trigger LLM fallback
FALLBACK_STATUSES = {"error", "no_solution", "parse_error", "not_implemented"}


# ============================================================================
# Dataset helpers (from run_z3.py)
# ============================================================================

def extract_dataset_name(dataset_path: str) -> str:
    """Extract dataset name from path without extension."""
    return Path(dataset_path).stem


def get_preprocessed_path(dataset_path: str) -> str:
    """Transform dataset path to preprocessed conditions file path."""
    path = Path(dataset_path)
    directory = path.parent if path.parent.name else Path("data")
    filename = path.stem
    extension = path.suffix
    preprocessed_filename = f"all_conditions_{filename}{extension}"
    return str(directory / preprocessed_filename)


def read_dataset(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Read dataset CSV and return list of rows."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def find_solver_files(dataset_name: str, limit: Optional[int] = None) -> List[str]:
    """Find all solver files for the given dataset, sorted by index."""
    solvers_dir = Path("src/calculation/processing/solvers")
    if not solvers_dir.exists():
        raise FileNotFoundError(f"Solvers directory not found: {solvers_dir}")

    pattern = f"solve_all_conditions_{dataset_name}_*.py"
    solver_files = list(solvers_dir.glob(pattern))

    def get_index(filepath):
        return int(filepath.stem.split("_")[-1])

    solver_files.sort(key=get_index)

    if limit is not None:
        solver_files = solver_files[:limit]

    return [str(f) for f in solver_files]


# ============================================================================
# Z3 solver runner (from run_z3.py)
# ============================================================================

def run_solver(solver_path: str, timeout: int = 30) -> Tuple[str, str, int, float]:
    """Run a single solver file and capture its output."""
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python3", solver_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        execution_time = time.time() - start_time
        return result.stdout, result.stderr, result.returncode, execution_time
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return "", f"TIMEOUT after {timeout}s", -1, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return "", f"EXCEPTION: {str(e)}", -1, execution_time


def parse_solver_output(stdout: str, stderr: str, return_code: int) -> Tuple[Optional[float], str]:
    """Parse solver output to extract answer and status."""
    if return_code != 0:
        if "TIMEOUT" in stderr:
            return None, "timeout"
        return None, "error"

    stdout = stdout.strip()

    if not stdout:
        return None, "no_output"

    if stdout.startswith("ANSWER:"):
        try:
            answer_str = stdout.split("ANSWER:")[1].strip()
            answer = float(answer_str)
            return answer, "success"
        except (ValueError, IndexError):
            return None, "parse_error"

    elif stdout.startswith("ERROR:"):
        error_msg = stdout.split("ERROR:")[1].strip().lower()
        if "no solution" in error_msg:
            return None, "no_solution"
        elif "not implemented" in error_msg:
            return None, "not_implemented"
        else:
            return None, "error"

    # Backward compatibility: plain number
    try:
        answer = float(stdout)
        return answer, "success"
    except ValueError:
        return None, "parse_error"


# ============================================================================
# LLM fallback (from baseline.py)
# ============================================================================

def extract_number(text: str) -> Optional[float]:
    """Extract a numeric answer from model output."""
    text = text.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()

    if "</think>" in text:
        text = text.split("</think>")[0]

    answer_match = re.search(r"[Оо]твет[:\s]+(-?\d+[\.,]?\d*)", text)
    if answer_match:
        return float(answer_match.group(1).replace(",", "."))

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        content = boxed_match.group(1)
        frac_match = re.search(r"(-?)\\d?frac\{(-?\d+)\}\{(\d+)\}", content)
        if frac_match:
            sign = -1 if frac_match.group(1) == "-" else 1
            num = int(frac_match.group(2))
            den = int(frac_match.group(3))
            if den != 0:
                return sign * num / den
        num_match = re.search(r"-?\d+[\.,]?\d*", content)
        if num_match:
            return float(num_match.group().replace(",", "."))

    sqrt_frac_match = re.search(r"sqrt\((\d+)\)\s*/\s*(\d+)", text)
    if sqrt_frac_match:
        num, den = int(sqrt_frac_match.group(1)), int(sqrt_frac_match.group(2))
        if den != 0:
            return math.sqrt(num) / den

    fraction_match = re.search(r"(-?\d+)\s*/\s*(\d+)", text)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        if den != 0:
            return num / den

    number_match = re.search(r"-?\d+[\.,]?\d*", text)
    if number_match:
        return float(number_match.group().replace(",", "."))

    return None


def run_llm_fallback(
    question: str,
    model,
    tokenizer,
) -> Tuple[Optional[float], float]:
    """
    Run the two-step LLM pipeline (solve → extract answer) for a single question.

    Returns:
        (predicted_answer, elapsed_seconds)
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    start = time.time()

    # Step 1: solve
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
        model,
        tokenizer,
        prompt=prompt_solve,
        max_tokens=SOLVE_MAX_TOKENS,
        verbose=False,
        sampler=make_sampler(temp=0.0),
    )
    solution_clean = re.sub(r"<think>.*?</think>", "", solution, flags=re.DOTALL)
    solution_clean = (
        solution_clean.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
    )

    # Step 2: extract answer
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
        model,
        tokenizer,
        prompt=prompt_answer,
        max_tokens=ANSWER_MAX_TOKENS,
        verbose=False,
        sampler=make_sampler(temp=0.0),
    )
    answer_clean = (
        answer_response.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
    )

    elapsed = time.time() - start
    predicted = extract_number(answer_clean)
    return predicted, elapsed


# ============================================================================
# Answer comparison
# ============================================================================

def compare_answers(
    solver_answer: Optional[float],
    expected_answer: str,
    tolerance: float = 0.01,
) -> bool:
    """Compare solver answer with expected answer (supports multiple values separated by , or ;)."""
    if solver_answer is None:
        return False

    gt_values = []
    for part in expected_answer.replace(";", ",").split(","):
        part = part.strip()
        if "/" in part:
            try:
                num, den = part.split("/")
                gt_values.append(float(num) / float(den))
            except Exception:
                pass
        else:
            try:
                gt_values.append(float(part))
            except Exception:
                pass

    if not gt_values:
        return False

    return any(abs(solver_answer - gt) <= tolerance for gt in gt_values)


# ============================================================================
# Statistics
# ============================================================================

def generate_statistics(results: List[Dict]) -> Dict:
    """Generate statistics from results."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    correct_z3 = sum(1 for r in results if r["is_correct"] and r["source"] == "z3")
    correct_llm = sum(1 for r in results if r["is_correct"] and r["source"] == "llm")

    # Z3 status breakdown
    z3_status_counts: Dict[str, int] = {}
    for r in results:
        s = r["z3_status"]
        z3_status_counts[s] = z3_status_counts.get(s, 0) + 1

    # LLM fallback stats
    llm_total = sum(1 for r in results if r["source"] == "llm")
    llm_correct = correct_llm

    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = sum(r["execution_time"] for r in results) / total if total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "correct_z3": correct_z3,
        "correct_llm": correct_llm,
        "accuracy": accuracy,
        "z3_status_counts": z3_status_counts,
        "llm_total": llm_total,
        "llm_correct": llm_correct,
        "avg_execution_time": avg_time,
    }


def print_statistics(stats: Dict):
    """Print statistics in a formatted way."""
    total = stats["total"]
    correct = stats["correct"]
    correct_z3 = stats["correct_z3"]
    correct_llm = stats["correct_llm"]

    pct = lambda n: (n / total * 100) if total > 0 else 0
    pct_of_correct = lambda n: (n / correct * 100) if correct > 0 else 0

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total problems:        {total}")
    print(f"Correct answers:       {correct}  ({pct(correct):.2f}%)")
    print(f"  - from Z3 solver:    {correct_z3}  "
          f"({pct(correct_z3):.2f}% of total, "
          f"{pct_of_correct(correct_z3):.1f}% of correct)")
    print(f"  - from LLM fallback: {correct_llm}  "
          f"({pct(correct_llm):.2f}% of total, "
          f"{pct_of_correct(correct_llm):.1f}% of correct)")
    print(f"Avg execution time:    {stats['avg_execution_time']:.2f}s")

    print("\nZ3 solver status breakdown:")
    fallback_count = 0
    for status, count in sorted(stats["z3_status_counts"].items()):
        percentage = pct(count)
        note = "  → sent to LLM" if status in FALLBACK_STATUSES else ""
        print(f"  {status:20s}: {count:3d} ({percentage:5.1f}%){note}")
        if status in FALLBACK_STATUSES:
            fallback_count += count

    llm_total = stats["llm_total"]
    llm_correct = stats["llm_correct"]
    llm_incorrect = llm_total - llm_correct
    llm_acc = (llm_correct / llm_total * 100) if llm_total > 0 else 0

    print(f"\nLLM fallback ({llm_total} problems):")
    if llm_total > 0:
        print(f"  correct              : {llm_correct:3d} ({llm_correct / llm_total * 100:5.1f}%)")
        print(f"  incorrect            : {llm_incorrect:3d} ({llm_incorrect / llm_total * 100:5.1f}%)")
        print(f"  LLM accuracy:          {llm_acc:.2f}%")
    else:
        print("  (no problems sent to LLM)")

    print("=" * 60)


# ============================================================================
# Save results
# ============================================================================

def save_results(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    fieldnames = [
        "row_index",
        "question",
        "expected_answer",
        "solver_answer",
        "is_correct",
        "source",
        "z3_status",
        "execution_time",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Z3 solvers with LLM fallback and evaluate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/calculation/run_z3_unite.py --dataset data/t_geom_lettered_validated.csv --limit 51
  python3 src/calculation/run_z3_unite.py --dataset data/t_geom_lettered_validated.csv
        """,
    )
    parser.add_argument("--dataset", required=True, help="Dataset CSV path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for each Z3 solver in seconds (default: 30)",
    )
    args = parser.parse_args()

    try:
        dataset_name = extract_dataset_name(args.dataset)
        preprocessed_path = get_preprocessed_path(args.dataset)

        print(f"📂 Loading dataset: {preprocessed_path}")
        dataset_rows = read_dataset(preprocessed_path, args.limit)
        print(f"   Loaded {len(dataset_rows)} problems")

        print("\n🔍 Finding solver files...")
        solver_files = find_solver_files(dataset_name, args.limit)
        print(f"   Found {len(solver_files)} solver files")

        if len(solver_files) != len(dataset_rows):
            print(
                f"\n⚠️  Warning: Number of solver files ({len(solver_files)}) "
                f"doesn't match dataset rows ({len(dataset_rows)})"
            )

        # Lazy-load LLM model only when needed
        llm_model = None
        llm_tokenizer = None

        def ensure_model_loaded():
            nonlocal llm_model, llm_tokenizer
            if llm_model is None:
                print(f"\n🤖 Loading LLM model: {MODEL_NAME}")
                load_start = time.time()
                from mlx_lm import load
                llm_model, llm_tokenizer = load(MODEL_NAME)
                print(f"   Loaded in {time.time() - load_start:.1f}s")
            return llm_model, llm_tokenizer

        print(f"\n{'='*60}")
        print("RUNNING SOLVERS")
        print(f"{'='*60}")

        results = []

        for i, (solver_path, row) in enumerate(zip(solver_files, dataset_rows)):
            print(f"\n[{i+1}/{len(solver_files)}] Running {Path(solver_path).name}...")

            # ── Z3 solver ────────────────────────────────────────────────────
            stdout, stderr, return_code, exec_time = run_solver(solver_path, args.timeout)
            solver_answer, z3_status = parse_solver_output(stdout, stderr, return_code)

            source = "z3"
            final_answer = solver_answer
            total_time = exec_time

            # ── LLM fallback ─────────────────────────────────────────────────
            if z3_status in FALLBACK_STATUSES:
                print(f"   ⚠️  Z3 status: {z3_status} → running LLM fallback...")
                model, tokenizer = ensure_model_loaded()
                llm_answer, llm_time = run_llm_fallback(row["question"], model, tokenizer)
                final_answer = llm_answer
                source = "llm"
                total_time += llm_time

            # ── Compare ───────────────────────────────────────────────────────
            expected_answer = row["verifiable_answer"]
            is_correct = compare_answers(final_answer, expected_answer)

            result = {
                "row_index": i,
                "question": row["question"],
                "expected_answer": expected_answer,
                "solver_answer": final_answer if final_answer is not None else "N/A",
                "is_correct": is_correct,
                "source": source,
                "z3_status": z3_status,
                "execution_time": total_time,
            }
            results.append(result)

            status_symbol = "✓" if is_correct else "✗"
            print(
                f"   {status_symbol} Expected: {expected_answer}, "
                f"Got: {final_answer}, "
                f"Z3: {z3_status}, Source: {source}, "
                f"Time: {total_time:.2f}s"
            )

        # ── Statistics ────────────────────────────────────────────────────────
        stats = generate_statistics(results)
        print_statistics(stats)

        # ── Save ──────────────────────────────────────────────────────────────
        output_path = f"./data/answers_{dataset_name}.csv"
        save_results(results, output_path)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
