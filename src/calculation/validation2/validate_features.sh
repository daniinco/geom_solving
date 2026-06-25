#!/bin/bash

# Скрипт для запуска всех validation скриптов и объединения результатов
# Принимает на вход all_conditions_*.csv (выход preproc_features.sh)
# Для каждого типа условий:
#   1) Рассуждение о правильности (350-500 токенов)
#   2) Вердикт: Правильно / Не правильно
#   3) Если неправильно — исправление условия
# Выходной файл: {название входного}_validated.csv

# Парсинг аргументов
DATASET=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Проверка обязательных аргументов
if [ -z "$DATASET" ]; then
    echo "Error: --dataset argument is required"
    echo "Usage: $0 --dataset <path_to_all_conditions_csv> [--limit <number>]"
    exit 1
fi

# Формируем аргументы для скриптов
ARGS="--dataset $DATASET"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

# Директория со скриптами
SCRIPT_DIR="src/calculation/validation"

# Список скриптов для запуска
SCRIPTS=(
    "val_angles.py"
    # "val_col.py"
    "val_lines.py"
    "val_perp.py"
    "val_points.py"
    "val_segm_conds.py"
    "val_statement.py"
)

echo "Starting validation pipeline..."
echo "Dataset: $DATASET"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT"
fi
echo "Validated results will be saved to: ./data/validation/"
echo "================================"

# Создаём папку для результатов валидации заранее
mkdir -p ./data/validation

# Запускаем все скрипты валидации
for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "▶ Running $script..."
    echo "--------------------------------"
    python3 "$SCRIPT_DIR/$script" $ARGS

    if [ $? -ne 0 ]; then
        echo "Error: $script failed"
        exit 1
    fi
    echo "✓ $script completed successfully"
    echo "--------------------------------"
done

echo ""
echo "All validation scripts completed"
echo "================================"

# Запускаем joiner
echo "▶ Running val_joiner.py..."
python3 "$SCRIPT_DIR/val_joiner.py" --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo "Error: val_joiner.py failed"
    exit 1
fi

DATASET_NAME=$(basename "$DATASET" .csv)
DATASET_DIR=$(dirname "$DATASET")

echo "================================"
echo "Validation pipeline completed successfully!"
echo "Result saved to: ${DATASET_DIR}/${DATASET_NAME}_validated.csv"
echo "Per-type validated files in: ./data/validation/"
