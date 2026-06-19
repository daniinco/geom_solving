#!/bin/bash

# Скрипт для запуска всех preprocessing скриптов с рассуждениями и объединения результатов
# Отличие от preprocessing2: каждый скрипт делает 2 вызова модели:
#   1) enable_thinking=True, max_tokens=350 — получаем рассуждение
#   2) enable_thinking=False — получаем финальный ответ
# Рассуждения выводятся в консоль и сохраняются в data/thinkings/

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
    echo "Usage: $0 --dataset <path_to_dataset> [--limit <number>]"
    exit 1
fi

# Формируем аргументы для скриптов
ARGS="--dataset $DATASET"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

# Директория со скриптами
SCRIPT_DIR="src/calculation/preprocessing3"

# Список скриптов для запуска
SCRIPTS=(
    "get_angles.py"
    "get_col.py"
    "get_lines.py"
    "get_perp.py"
    "get_points.py"
    "get_segm_conds.py"
    "get_statement.py"
)

echo "Starting preprocessing3 pipeline (with thinking)..."
echo "Dataset: $DATASET"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT"
fi
echo "Thinkings will be saved to: ./data/thinkings/"
echo "================================"

# Создаём папку для рассуждений заранее
mkdir -p ./data/thinkings

# Запускаем все скрипты
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
echo "All preprocessing scripts completed"
echo "================================"

# Запускаем joiner
echo "▶ Running joiner.py..."
python3 "$SCRIPT_DIR/joiner.py" --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo "Error: joiner.py failed"
    exit 1
fi

echo "================================"
echo "Pipeline completed successfully!"
echo "Result saved to:     ./data/all_conditions_<dataset_name>.csv"
echo "Thinkings saved to:  ./data/thinkings/"
