#!/usr/bin/env python3
"""
Z3 Solver Generator Script

This script generates individual Z3 solver Python files for each row in a preprocessed
geometry problem dataset. It reads condition data from preprocessed CSV files and creates
solver scripts in the solvers/ directory.

Usage:
    python3 src/calculation/processing/z3_program.py --dataset data/all_conditions_t_easy_cases.csv --limit 4

The script will:
1. Read the condition fields from the specified CSV file
2. Generate a solver file for each row using the generate_solver_code() function
3. Save files as solve_{dataset_name}_{row_index}.py in the solvers/ directory
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict


def extract_dataset_name(dataset_path: str) -> str:
    """
    Extract dataset name from path without extension.
    
    Args:
        dataset_path: Path to dataset file
    
    Returns:
        Dataset name without extension
    
    Example:
        >>> extract_dataset_name('data/all_conditions_t_easy_cases.csv')
        'all_conditions_t_easy_cases'
    """
    return Path(dataset_path).stem


def read_conditions_csv(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Read preprocessed CSV and return list of condition dictionaries.
    
    Args:
        csv_path: Path to the preprocessed CSV file
        limit: Maximum number of rows to read (None for all rows)
    
    Returns:
        List of dictionaries containing condition fields
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Preprocessed CSV file not found: {csv_path}\n"
            f"Please run the preprocessing pipeline first."
        )
    
    required_columns = [
        'angles_condition',
        'cols_condition',
        'lines_condition',
        'perp_condition',
        'points_condition',
        'segm_condition',
        'statement_condition'
    ]
    
    conditions_list = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if all required columns are present
        if not all(col in reader.fieldnames for col in required_columns):
            missing = [col for col in required_columns if col not in reader.fieldnames]
            raise ValueError(
                f"Missing required columns in CSV: {missing}\n"
                f"Available columns: {reader.fieldnames}"
            )
        
        # Read rows up to limit
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            
            conditions = {col: row[col] for col in required_columns}
            conditions_list.append(conditions)
    
    return conditions_list


def generate_solver_code(
    angles_condition: str,
    cols_condition: str,
    lines_condition: str,
    perp_condition: str,
    points_condition: str,
    segm_condition: str,
    statement_condition: str
) -> list:
    """
    Generate Z3 solver code based on geometry conditions.
    
    STUB IMPLEMENTATION: Currently returns placeholder code.
    TODO: Implement actual Z3 constraint generation logic.
    
    This function will be replaced with actual implementation that:
    1. Parses each condition string
    2. Creates Z3 variables for points, angles, and segments
    3. Generates Z3 constraints based on the conditions
    4. Adds solver logic to find solutions
    5. Returns list of code lines
    
    Args:
        angles_condition: Angle constraints (e.g., "CPB=22,QCB=90")
        cols_condition: Collinearity conditions (e.g., "- ABCP")
        lines_condition: Line segment conditions (e.g., "APC,BPQ")
        perp_condition: Perpendicularity conditions (e.g., "QC perp BC")
        points_condition: Point definitions (e.g., "A,C,B,P,Q")
        segm_condition: Segment length conditions (e.g., "AC=2PC")
        statement_condition: What to find in the problem (e.g., "2 ACB" for angle, "1 BC" for length)
    
    Returns:
        List of strings, each representing a line of Python code
    """
    res = [
        "from z3 import *",
        "import math",
        "",
        "s = Solver()",
        's.set("timeout", 10000)',
        "",
        "def z3_to_float(val):",
        "    if val is None:",
        "        return None",
        "    if isinstance(val, IntNumRef):",
        "        return float(val.as_long())",
        "    if isinstance(val, RatNumRef):",
        "        return float(val.as_fraction())",
        "    if isinstance(val, AlgebraicNumRef):",
        "        return z3_to_float(val.approx(15))",
        "    if isinstance(val, ArithRef):",
        "        simplified = simplify(val)",
        "        if not isinstance(simplified, ArithRef):",
        "            return z3_to_float(simplified)",
        "        return None",
        "    try:",
        "        return float(str(val).rstrip('?'))",
        "    except:",
        "        return None",
    ]

    #Создаем переменные для точек
    points_condition = points_condition.split(',')
    for point in points_condition:
        res.append(f"{point}x, {point}y = Reals('{point}x {point}y')")
    res.append("")
    
    #Задаем несовпадение точек
    for point in points_condition:
        for point2 in points_condition:
            if point != point2:
                res.append(f"s.add(Or({point}x != {point2}x, {point}y != {point2}y))")
            else:
                break
    
    #Задаем упрощение
    res.append(f"s.add({points_condition[0]}x == 0, {points_condition[0]}y == 0)")
    if len(points_condition) > 1:
        res.append(f"s.add({points_condition[1]}x > 0, {points_condition[1]}y == 0)")
    if len(points_condition) > 2:
        res.append(f"s.add({points_condition[2]}y > 0)")
    
    #Задаем условия для отрезков
    if segm_condition[0] != "-":
        segm_condition = segm_condition.split(',')
        for segm_cond in segm_condition:
            segm_cond = segm_cond.strip()
            if len(segm_cond.split("=")) != 2:
                print("Неправильное условие на равенство отрезков")
                continue
            segm_cond_left, segm_cond_right = segm_cond.split("=")
            segm_cond_left, segm_cond_right = segm_cond_left.strip(), segm_cond_right.strip()
            if len(segm_cond_left) < 2:
                print("Неправильное условие на отрезок")
                continue
            try:
                segm_cond_right = float(segm_cond_right)
                res.append(f"s.add(({segm_cond_left[0]}x - {segm_cond_left[1]}x)**2 + ({segm_cond_left[0]}y - {segm_cond_left[1]}y)**2 == {segm_cond_right * segm_cond_right})")
            except ValueError:
                if segm_cond_right[0].isdigit():
                    lett_start = 0
                    for i in range(len(segm_cond_right[0])):
                        if segm_cond_right[i] != "." and not segm_cond_right[i].isdigit():
                            break
                        else:
                            lett_start += 1
                    factor = float(segm_cond_right[:lett_start])
                    letter = segm_cond_right[lett_start:].strip()
                    if len(letter) > 1:
                        res.append(f"s.add(({segm_cond_left[0]}x - {segm_cond_left[1]}x)**2 + ({segm_cond_left[0]}y - {segm_cond_left[1]}y)**2 == {factor * factor} * (({letter[0]}x - {letter[1]}x)**2 + ({letter[0]}y - {letter[1]}y)**2))")
                else:
                    res.append(f"s.add(({segm_cond_left[0]}x - {segm_cond_left[1]}x)**2 + ({segm_cond_left[0]}y - {segm_cond_left[1]}y)**2 == ({segm_cond_right[0]}x - {segm_cond_right[1]}x)**2 + ({segm_cond_right[0]}y - {segm_cond_right[1]}y)**2)")
    
    #Задаем условия перпендикулярности
    if perp_condition[0] != "-":
        perp_condition = perp_condition.split(',')
        for perp_cond in perp_condition:
            perp_cond = perp_cond.strip()
            if len(perp_cond.split()) != 3:
                print("Неправильное условие на перпендикулярность")
                continue
            perp_cond_left, perp_cond_type, perp_cond_right = perp_cond.split()
            perp_cond_left, perp_cond_type, perp_cond_right = perp_cond_left.strip(), perp_cond_type.strip(), perp_cond_right.strip()
            if len(perp_cond_left) < 2 or len(perp_cond_right) < 2:
                print("Неправильное условие на перпендикулярность")
            elif perp_cond_type[:2] == "pe":
                res.append(f"s.add(({perp_cond_left[0]}x - {perp_cond_left[1]}x)*({perp_cond_right[0]}x - {perp_cond_right[1]}x) + ({perp_cond_left[0]}y - {perp_cond_left[1]}y)*({perp_cond_right[0]}y - {perp_cond_right[1]}y) == 0)")
            elif perp_cond_type[:2] == "pa":
                res.append(f"s.add(({perp_cond_left[0]}x - {perp_cond_left[1]}x)*({perp_cond_right[0]}y - {perp_cond_right[1]}y) - ({perp_cond_left[0]}y - {perp_cond_left[1]}y)*({perp_cond_right[0]}x - {perp_cond_right[1]}x) == 0)")
            else:
                print("Неправильное условие на перпендикулярность")
    
    #задаем условия коллинеарности(прямые)
    if lines_condition[0] != "-":
        lines_condition = lines_condition.split(',')
        for line_cond in lines_condition:
            line_cond = line_cond.strip()
            if len(line_cond) < 3:
                print("Неправильное условие на прямые")
                continue
            first, second, third = line_cond[0], line_cond[1], line_cond[2]
            res.append(f"s.add(({first}x - {second}x) * ({third}y - {second}y) == ({first}y - {second}y) * ({third}x - {second}x))")
    
    #задаем условия на углы
    if angles_condition[0] != "-":
        angles_condition = angles_condition.split(',')
        for angle_cond in angles_condition:
            angle_cond = angle_cond.strip()
            if len(angle_cond.split("=")) < 2:
                print("Неправильное условие на углы")
                continue
            angle_cond_left, angle_cond_right = angle_cond.split("=")[0], angle_cond.split("=")[1]
            angle_cond_left, angle_cond_right = angle_cond_left.strip(), angle_cond_right.strip()
            try:
                angle_cond_right = float(angle_cond_right)
                if len(angle_cond_left) < 3:
                    print("Неправильное условие на углы")
                    continue
                first, second, third = angle_cond_left[0], angle_cond_left[1], angle_cond_left[2]
                if angle_cond_right == 90:
                    res.append(f"s.add(({first}x - {second}x)*({second}x - {third}x) + ({first}y - {second}y)*({second}y - {third}y) == 0)")
                    continue
                tg_angle = math.tan(math.radians(angle_cond_right))
                res.append(f"s.add(Or(({first}x - {second}x) * ({third}y - {second}y) - ({first}y - {second}y) * ({third}x - {second}x) == {tg_angle} * (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)), ({first}x - {second}x) * ({third}y - {second}y) - ({first}y - {second}y) * ({third}x - {second}x) == {-1 * tg_angle} * (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y))))")
                cos_angle = math.cos(math.radians(angle_cond_right))
                if cos_angle >= 0:
                    res.append(f"s.add(({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y) >= 0)")
                else:
                    res.append(f"s.add(({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y) < 0)")
            except ValueError:
                if len(angle_cond_right) < 3:
                    print("Неправильное условие на углы")
                elif angle_cond_right[0].isdigit():
                    pass # доделать
                else:
                    first, second, third = angle_cond_left[0], angle_cond_left[1], angle_cond_left[2]
                    first_2, second_2, third_2 = angle_cond_right[0], angle_cond_right[1], angle_cond_right[2]
                    res.append(f"s.add(Or((({first}x - {second}x) * ({third}y - {second}y) - ({first}y - {second}y) * ({third}x - {second}x)) * (({first_2}x - {second_2}x) * ({third_2}x - {second_2}x) + ({first_2}y - {second_2}y) * ({third_2}y - {second_2}y)) == (({first_2}x - {second_2}x) * ({third_2}y - {second_2}y) - ({first_2}y - {second_2}y) * ({third_2}x - {second_2}x)) * (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)), (({first}x - {second}x) * ({third}y - {second}y) - ({first}y - {second}y) * ({third}x - {second}x)) * (({first_2}x - {second_2}x) * ({third_2}x - {second_2}x) + ({first_2}y - {second_2}y) * ({third_2}y - {second_2}y)) == -1 * (({first_2}x - {second_2}x) * ({third_2}y - {second_2}y) - ({first_2}y - {second_2}y) * ({third_2}x - {second_2}x)) * (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y))))")
                    res.append(f"s.add(Or(And((({first_2}x - {second_2}x) * ({third_2}x - {second_2}x) + ({first_2}y - {second_2}y) * ({third_2}y - {second_2}y)) >= 0, (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)) >= 0), And((({first_2}x - {second_2}x) * ({third_2}x - {second_2}x) + ({first_2}y - {second_2}y) * ({third_2}y - {second_2}y)) < 0, (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)) < 0)))")


    #Задаем вывод результата
    if statement_condition[0] == "1":
        statement = statement_condition.split()[1]
        if len(statement) < 2:
            res.append("print('ERROR: Wrong statement')")
            return res
        res.append("answer_sq = Real('answer_sq')")
        res.append(f"s.add(({statement[0]}x - {statement[1]}x)**2 + ({statement[0]}y - {statement[1]}y)**2 == answer_sq)")
        res.append("")

        res.append("if s.check() == sat:")
        res.append("    model = s.model()")
        res.append("    answer = model[answer_sq]")
        # res.append("    if isinstance(answer, RatNumRef):")
        # res.append("        answer = float(answer.as_fraction()) ** 0.5")
        # res.append("    elif isinstance(answer, AlgebraicNumRef):")
        # res.append("        answer = answer.approx(20) ** 0.5")
        # res.append("    else:")
        # res.append("        answer = answer")
        res.append("    print(f'ANSWER: {z3_to_float(answer) ** 0.5}')")
        res.append("else:")
        res.append("    print('ERROR: No solution found')")
    elif statement_condition[0] == "2":
        statement = statement_condition.split()[1]
        if len(statement) < 3:
            res.append("print('ERROR: Wrong statement')")
            return res
        first, second, third = statement[0], statement[1], statement[2]
        res.append("answer_tg = Real('answer_tg')")
        res.append("answer_sc_pro = Real('answer_sc_pro')")
        res.append(f"s.add(({first}x - {second}x) * ({third}y - {second}y) - ({first}y - {second}y) * ({third}x - {second}x) == answer_tg * (({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)))")
        res.append(f"s.add((({first}x - {second}x) * ({third}x - {second}x) + ({first}y - {second}y) * ({third}y - {second}y)) == answer_sc_pro)")
        res.append("if s.check() == sat:")
        res.append("    model = s.model()")
        res.append("    answer_tg = z3_to_float(model[answer_tg])")
        res.append("    answer_sc_pro = z3_to_float(model[answer_sc_pro])")
        res.append("    res = abs(math.degrees(math.atan(answer_tg)))")
        res.append("    if (res >= 90 and answer_sc_pro >= 0) or (res < 90 and answer_sc_pro < 0):")
        res.append("        res = 180 - res")
        res.append("    print(f'ANSWER: {res}')")
        res.append("else:")
        res.append("    print('ERROR: No solution found')")
    else:
        line_1 = statement_condition.split()[1]
        line_2 = statement_condition.split()[2]
        if len(line_1) != 2  or len(line_2) != 2:
            res.append("print('ERROR: Wrong statement')")
            return res
        first, second, third, fourth = line_1[0], line_1[1], line_2[0], line_2[1]
        res.append("answer_tg = Real('answer_tg')")
        res.append("answer_sc_pro = Real('answer_sc_pro')")
        res.append(f"s.add(({first}x - {second}x) * ({third}y - {fourth}y) - ({first}y - {second}y) * ({third}x - {fourth}x) == answer_tg * (({first}x - {second}x) * ({third}x - {fourth}x) + ({first}y - {second}y) * ({third}y - {fourth}y)))")
        res.append(f"s.add((({first}x - {second}x) * ({third}x - {fourth}x) + ({first}y - {second}y) * ({third}y - {fourth}y)) == answer_sc_pro)")
        res.append("if s.check() == sat:")
        res.append("    model = s.model()")
        res.append("    answer_tg = z3_to_float(model[answer_tg])")
        res.append("    answer_sc_pro = z3_to_float(model[answer_sc_pro])")
        res.append("    res = abs(math.degrees(math.atan(answer_tg)))")
        res.append("    if (res >= 90 and answer_sc_pro >= 0) or (res < 90 and answer_sc_pro < 0):")
        res.append("        res = 180 - res")
        res.append("    print(f'ANSWER: {res}')")
        res.append("else:")
        res.append("    print('ERROR: No solution found')")
    return res


def write_solver_file(
    output_dir: str,
    dataset_name: str,
    row_index: int,
    code_lines: list
) -> str:
    """
    Write generated solver code to file.
    
    Args:
        output_dir: Directory where solver files will be saved
        dataset_name: Name of the dataset (without extension)
        row_index: Zero-based index of the row
        code_lines: List of strings, each representing a line of Python code
    
    Returns:
        Path to the created file
    
    Raises:
        OSError: If file cannot be written
    """
    filename = f"solve_{dataset_name}_{row_index}.py"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(code_lines))
        f.write('\n')  # Add final newline
    
    return filepath


def main():
    """
    Main execution flow:
    1. Parse command-line arguments
    2. Create output directory if needed
    3. Read CSV with conditions
    4. For each row (up to limit):
        a. Extract condition fields
        b. Generate solver code (stub)
        c. Write to file
    5. Report success/failure
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate Z3 solver files from preprocessed geometry problem dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate solvers for first 4 problems
  python3 src/calculation/processing/z3_program.py --dataset data/all_conditions_t_easy_cases.csv --limit 4
  
  # Generate solvers for all problems
  python3 src/calculation/processing/z3_program.py --dataset data/all_conditions_t_easy_cases.csv
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to the preprocessed dataset CSV file with conditions (e.g., data/all_conditions_t_easy_cases.csv)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of rows to process (default: all rows)'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract dataset name and read CSV
        dataset_name = extract_dataset_name(args.dataset)
        
        print(f"Reading data from: {args.dataset}")
        
        # Read conditions from CSV
        conditions_list = read_conditions_csv(args.dataset, args.limit)
        
        if not conditions_list:
            print("No data to process.")
            return 0
        
        print(f"Processing {len(conditions_list)} rows...")
        
        # Create output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'solvers')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate solver files
        created_files = []
        for row_index, conditions in enumerate(conditions_list):
            try:
                # Generate solver code (returns list of lines)
                code_lines = generate_solver_code(
                    angles_condition=conditions['angles_condition'],
                    cols_condition=conditions['cols_condition'],
                    lines_condition=conditions['lines_condition'],
                    perp_condition=conditions['perp_condition'],
                    points_condition=conditions['points_condition'],
                    segm_condition=conditions['segm_condition'],
                    statement_condition=conditions['statement_condition']
                )
                
                # Write to file
                filepath = write_solver_file(output_dir, dataset_name, row_index, code_lines)
                created_files.append(filepath)
                print(f"Created: {filepath}")
            except Exception as e:
                import traceback
                tb = traceback.extract_tb(sys.exc_info()[2])
                # Find the frame in z3_program.py (this file)
                error_line = None
                for frame in tb:
                    if 'z3_program.py' in frame.filename:
                        error_line = frame.lineno
                        error_func = frame.name
                        error_code = frame.line
                
                print(f"\n{'='*70}", file=sys.stderr)
                print(f"ERROR processing row {row_index} (CSV line {row_index + 2})", file=sys.stderr)
                if error_line:
                    print(f"Failed at z3_program.py:{error_line} in function '{error_func}'", file=sys.stderr)
                    print(f"Code: {error_code}", file=sys.stderr)
                print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)
                print(f"\nConditions from CSV:", file=sys.stderr)
                print(f"  Points: {conditions.get('points_condition', 'N/A')}", file=sys.stderr)
                print(f"  Segments: {conditions.get('segm_condition', 'N/A')}", file=sys.stderr)
                print(f"  Angles: {conditions.get('angles_condition', 'N/A')}", file=sys.stderr)
                print(f"  Lines: {conditions.get('lines_condition', 'N/A')}", file=sys.stderr)
                print(f"  Perp: {conditions.get('perp_condition', 'N/A')}", file=sys.stderr)
                print(f"  Cols: {conditions.get('cols_condition', 'N/A')}", file=sys.stderr)
                print(f"  Statement: {conditions.get('statement_condition', 'N/A')}", file=sys.stderr)
                print(f"\nFull traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print(f"{'='*70}\n", file=sys.stderr)
                # Continue processing other rows
                continue
        
        print(f"\nSuccessfully generated {len(created_files)} solver files.")
        if len(created_files) < len(conditions_list):
            print(f"Warning: {len(conditions_list) - len(created_files)} files failed to generate.", file=sys.stderr)
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())