#!/usr/bin/env python3
"""
Z3 Solver Runner and Evaluator

This script runs all generated Z3 solver files and evaluates their results against
the expected answers from the dataset.

Usage:
    python3 src/calculation/run_z3.py --dataset t_easy_cases.csv --limit 5

The script will:
1. Find all solver files for the specified dataset
2. Run each solver and capture its output
3. Compare results with expected answers
4. Generate statistics and save results to CSV
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
    """
    Read dataset CSV and return list of rows.
    
    Args:
        csv_path: Path to the CSV file
        limit: Maximum number of rows to read
    
    Returns:
        List of dictionaries containing all columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    
    return rows


def find_solver_files(dataset_name: str, limit: Optional[int] = None) -> List[str]:
    """
    Find all solver files for the given dataset.
    
    Args:
        dataset_name: Name of the dataset (without extension)
        limit: Maximum number of solver files to find
    
    Returns:
        List of paths to solver files, sorted by index
    """
    solvers_dir = Path("src/calculation/processing/solvers")
    
    if not solvers_dir.exists():
        raise FileNotFoundError(f"Solvers directory not found: {solvers_dir}")
    
    # Find all solver files matching the pattern
    pattern = f"solve_all_conditions_{dataset_name}_*.py"
    solver_files = list(solvers_dir.glob(pattern))
    
    # Sort by index number
    def get_index(filepath):
        stem = filepath.stem  # e.g., "solve_all_conditions_t_easy_cases_0"
        return int(stem.split('_')[-1])
    
    solver_files.sort(key=get_index)
    
    if limit is not None:
        solver_files = solver_files[:limit]
    
    return [str(f) for f in solver_files]


def run_solver(solver_path: str, timeout: int = 30) -> Tuple[str, str, int, float]:
    """
    Run a single solver file and capture its output.
    
    Args:
        solver_path: Path to the solver Python file
        timeout: Maximum execution time in seconds
    
    Returns:
        Tuple of (stdout, stderr, return_code, execution_time)
    """
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python3', solver_path],
            capture_output=True,
            text=True,
            timeout=timeout
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
    """
    Parse solver output to extract answer and status.
    
    Args:
        stdout: Standard output from solver
        stderr: Standard error from solver
        return_code: Process return code
    
    Returns:
        Tuple of (answer, status) where:
        - answer: float value if found, None otherwise
        - status: one of "success", "no_solution", "timeout", "error", "parse_error"
    """
    # Check for errors first
    if return_code != 0:
        if "TIMEOUT" in stderr:
            return None, "timeout"
        return None, "error"
    
    # Parse stdout
    stdout = stdout.strip()
    
    if not stdout:
        return None, "no_output"
    
    # Look for structured output
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
    
    # Try to parse as plain number (for backward compatibility)
    try:
        answer = float(stdout)
        return answer, "success"
    except ValueError:
        return None, "parse_error"


def compare_answers(solver_answer: Optional[float], expected_answer: str, tolerance: float = 0.01) -> bool:
    """
    Compare solver answer with expected answer.
    
    Args:
        solver_answer: Answer from solver (None if no answer)
        expected_answer: Expected answer from dataset
        tolerance: Tolerance for floating point comparison
    
    Returns:
        True if answers match within tolerance, False otherwise
    """
    if solver_answer is None:
        return False
    
    try:
        expected = float(expected_answer)
        return abs(solver_answer - expected) <= tolerance
    except ValueError:
        return False


def generate_statistics(results: List[Dict]) -> Dict[str, any]:
    """
    Generate statistics from results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with statistics
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    
    # Count by status
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate percentages
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Average execution time
    avg_time = sum(r['execution_time'] for r in results) / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'status_counts': status_counts,
        'avg_execution_time': avg_time
    }


def save_results(results: List[Dict], output_path: str):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return
    
    fieldnames = [
        'row_index',
        'question',
        'expected_answer',
        'solver_answer',
        'is_correct',
        'status',
        'execution_time',
        # 'angles_condition',
        # 'cols_condition',
        # 'lines_condition',
        # 'perp_condition',
        # 'points_condition',
        # 'segm_condition',
        # 'statement_condition'
    ]
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n💾 Results saved to: {output_path}")


def print_statistics(stats: Dict):
    """Print statistics in a formatted way."""
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total problems:        {stats['total']}")
    print(f"Correct answers:       {stats['correct']}")
    print(f"Accuracy:              {stats['accuracy']:.2f}%")
    print(f"Avg execution time:    {stats['avg_execution_time']:.2f}s")
    print("\nStatus breakdown:")
    for status, count in sorted(stats['status_counts'].items()):
        percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {status:20s}: {count:3d} ({percentage:5.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Run Z3 solvers and evaluate results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run first 5 solvers
  python3 src/calculation/run_z3.py --dataset t_easy_cases.csv --limit 5
  
  # Run all solvers
  python3 src/calculation/run_z3.py --dataset t_easy_cases.csv
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., t_easy_cases.csv)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of problems to run (default: all)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout for each solver in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract dataset name
        dataset_name = extract_dataset_name(args.dataset)
        
        # Get preprocessed dataset path
        preprocessed_path = get_preprocessed_path(args.dataset)
        
        print(f"📂 Loading dataset: {preprocessed_path}")
        dataset_rows = read_dataset(preprocessed_path, args.limit)
        print(f"   Loaded {len(dataset_rows)} problems")
        
        # Find solver files
        print(f"\n🔍 Finding solver files...")
        solver_files = find_solver_files(dataset_name, args.limit)
        print(f"   Found {len(solver_files)} solver files")
        
        if len(solver_files) != len(dataset_rows):
            print(f"\n⚠️  Warning: Number of solver files ({len(solver_files)}) "
                  f"doesn't match dataset rows ({len(dataset_rows)})")
        
        # Run solvers
        print(f"\n{'='*60}")
        print("RUNNING SOLVERS")
        print(f"{'='*60}")
        
        results = []
        
        for i, (solver_path, row) in enumerate(zip(solver_files, dataset_rows)):
            print(f"\n[{i+1}/{len(solver_files)}] Running {Path(solver_path).name}...")
            
            # Run solver
            stdout, stderr, return_code, exec_time = run_solver(solver_path, args.timeout)
            
            # Parse output
            solver_answer, status = parse_solver_output(stdout, stderr, return_code)
            
            # Compare with expected
            expected_answer = row['verifiable_answer']
            is_correct = compare_answers(solver_answer, expected_answer)
            
            # Store result
            result = {
                'row_index': i,
                'question': row['question'],
                'expected_answer': expected_answer,
                'solver_answer': solver_answer if solver_answer is not None else 'N/A',
                'is_correct': is_correct,
                'status': status,
                'execution_time': exec_time,
                # 'angles_condition': row.get('angles_condition', ''),
                # 'cols_condition': row.get('cols_condition', ''),
                # 'lines_condition': row.get('lines_condition', ''),
                # 'perp_condition': row.get('perp_condition', ''),
                # 'points_condition': row.get('points_condition', ''),
                # 'segm_condition': row.get('segm_condition', ''),
                # 'statement_condition': row.get('statement_condition', '')
            }
            results.append(result)
            
            # Print result
            status_symbol = "✓" if is_correct else "✗"
            print(f"   {status_symbol} Expected: {expected_answer}, Got: {solver_answer}, "
                  f"Status: {status}, Time: {exec_time:.2f}s")
        
        # Generate and print statistics
        stats = generate_statistics(results)
        print_statistics(stats)
        
        # Save results
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


if __name__ == '__main__':
    sys.exit(main())