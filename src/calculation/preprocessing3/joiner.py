#!/usr/bin/env python3
"""
Скрипт для объединения всех preprocessed датасетов в один.
Также объединяет файлы с рассуждениями в data/thinkings/.
"""

import argparse
import pandas as pd
from pathlib import Path


def get_dataset_name(dataset_path: str) -> str:
    """Извлекает имя датасета из пути к файлу."""
    return Path(dataset_path).stem


def main():
    parser = argparse.ArgumentParser(
        description="Join all preprocessed condition datasets into one"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the original dataset (used to determine dataset name)"
    )
    args = parser.parse_args()

    dataset_name = get_dataset_name(args.dataset)
    data_dir = Path("./data")
    thinkings_dir = data_dir / "thinkings"

    # Маппинг: префикс файла -> имя колонки в результате
    condition_files = {
        "angles": "angles_condition",
        "cols": "cols_condition",
        "lines": "lines_condition",
        "perp_conds": "perp_condition",
        "points": "points_condition",
        "segm_conds": "segm_condition",
        "statements": "statement_condition",
    }

    # ----------------------------------------------------------------
    # Объединяем файлы условий
    # ----------------------------------------------------------------
    first_prefix = list(condition_files.keys())[0]
    first_file = data_dir / f"{first_prefix}_{dataset_name}.csv"

    if not first_file.exists():
        raise FileNotFoundError(f"File not found: {first_file}")

    result_df = pd.read_csv(first_file)
    result_df = result_df.rename(columns={"condition": condition_files[first_prefix]})

    for prefix, col_name in list(condition_files.items())[1:]:
        file_path = data_dir / f"{prefix}_{dataset_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        result_df[col_name] = df["condition"]

    condition_columns = list(condition_files.values())
    final_columns = ["question"] + condition_columns + ["verifiable_answer"]
    result_df = result_df[final_columns]

    output_path = data_dir / f"all_conditions_{dataset_name}.csv"
    result_df.to_csv(output_path, index=False)

    print(f"Successfully joined {len(condition_files)} condition files")
    print(f"Total rows: {len(result_df)}")
    print(f"Columns: {', '.join(result_df.columns)}")
    print(f"Output saved to: {output_path}")

    # ----------------------------------------------------------------
    # Объединяем файлы рассуждений
    # ----------------------------------------------------------------
    thinkings_dir.mkdir(parents=True, exist_ok=True)

    thinking_first_file = thinkings_dir / f"{first_prefix}_{dataset_name}.csv"
    if not thinking_first_file.exists():
        print(f"\n⚠️  Файл рассуждений не найден: {thinking_first_file}, пропускаем объединение рассуждений")
        return

    thinking_df = pd.read_csv(thinking_first_file)
    thinking_df = thinking_df.rename(columns={"thinking": f"{first_prefix}_thinking"})

    for prefix in list(condition_files.keys())[1:]:
        thinking_path = thinkings_dir / f"{prefix}_{dataset_name}.csv"

        if not thinking_path.exists():
            print(f"⚠️  Файл рассуждений не найден: {thinking_path}, пропускаем")
            continue

        df = pd.read_csv(thinking_path)
        thinking_df[f"{prefix}_thinking"] = df["thinking"]

    thinking_columns = [f"{p}_thinking" for p in condition_files.keys()
                        if f"{p}_thinking" in thinking_df.columns]
    thinking_final_columns = ["question"] + thinking_columns + ["verifiable_answer"]
    thinking_final_columns = [c for c in thinking_final_columns if c in thinking_df.columns]
    thinking_df = thinking_df[thinking_final_columns]

    thinking_output_path = thinkings_dir / f"all_thinkings_{dataset_name}.csv"
    thinking_df.to_csv(thinking_output_path, index=False)

    print(f"\nSuccessfully joined {len(thinking_columns)} thinking files")
    print(f"Thinking columns: {', '.join(thinking_columns)}")
    print(f"Thinkings saved to: {thinking_output_path}")


if __name__ == "__main__":
    main()
