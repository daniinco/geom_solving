#!/usr/bin/env python3
"""
Скрипт для объединения всех валидированных датасетов в один.
Читает исходный all_conditions_*.csv и заменяет условия на валидированные версии.
Выходной файл: {название входного}_validated.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def get_dataset_name(dataset_path: str) -> str:
    """Извлекает имя датасета из пути к файлу."""
    return Path(dataset_path).stem


def main():
    parser = argparse.ArgumentParser(
        description="Join all validated condition datasets into one"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the all_conditions CSV file (used as base and to determine dataset name)"
    )
    args = parser.parse_args()

    dataset_name = get_dataset_name(args.dataset)
    validation_dir = Path("./data/validation")

    # Маппинг: префикс файла валидации -> имя колонки в результате
    # Порядок и имена колонок совпадают с joiner.py из preprocessing3
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
    # Загружаем исходный all_conditions файл как базу
    # ----------------------------------------------------------------
    base_file = Path(args.dataset)
    if not base_file.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_file}")

    result_df = pd.read_csv(base_file)
    print(f"Загружен базовый датасет: {base_file}")
    print(f"Строк: {len(result_df)}, Колонки: {', '.join(result_df.columns)}")

    # ----------------------------------------------------------------
    # Заменяем каждую колонку условий на валидированную версию
    # ----------------------------------------------------------------
    replaced_count = 0
    for prefix, col_name in condition_files.items():
        val_file = validation_dir / f"{prefix}_{dataset_name}_validated.csv"

        if not val_file.exists():
            print(f"⚠️  Файл валидации не найден: {val_file}, колонка '{col_name}' остаётся без изменений")
            continue

        val_df = pd.read_csv(val_file)

        if col_name not in val_df.columns:
            print(f"⚠️  Колонка '{col_name}' не найдена в {val_file}, пропускаем")
            continue

        if len(val_df) != len(result_df):
            print(f"⚠️  Количество строк в {val_file} ({len(val_df)}) не совпадает с базовым ({len(result_df)}), пропускаем")
            continue

        result_df[col_name] = val_df[col_name].values
        replaced_count += 1
        print(f"✓ Заменена колонка '{col_name}' из {val_file.name}")

    # ----------------------------------------------------------------
    # Сохраняем результат — все поля называются так же, как во входном
    # ----------------------------------------------------------------
    output_path = Path(args.dataset).parent / f"{dataset_name}_validated.csv"
    result_df.to_csv(output_path, index=False)

    print(f"\nЗаменено колонок: {replaced_count} из {len(condition_files)}")
    print(f"Всего строк: {len(result_df)}")
    print(f"Колонки: {', '.join(result_df.columns)}")
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()
