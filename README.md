Основной датасет - t_geom.csv

Остальные загруженные датасеты - его промежуточные преобразования

Как прогнать:

Запускаем замены букв:

./src/calculation/prepreprocessing/preproc_letters.py --dataset data/t_geom.csv --limit 51

Запускаем извлечения признаков(работает долго):

./src/calculation/preprocessing3/preproc_features.sh --dataset data/t_geom_lettered.csv --limit 51

Запускаем создание программ для z3:

python3 src/calculation/processing/z3_program.py --dataset data/all_conditions_t_geom_lettered.csv --limit 51

Запускаем 

python3 src/calculation/run_z3.py --dataset data/t_geom_lettered.csv --limit 51

Получаем результаты чистого z3 без валидации


Дальше надо почистить папку src/calculation/processing/solvers

Запускаем валидацию:

./src/calculation/validation/validate_features.sh --dataset data/all_conditions_t_geom_lettered.csv --limit 51

Запускаем создание программ для z3:

python3 src/calculation/processing/z3_program.py --dataset data/all_conditions_t_geom_lettered_validated.csv --limit 51

Запускаем z3:

python3 src/calculation/run_z3.py --dataset data/t_geom_lettered_validated.csv --limit 51

Получаем результаты чистого z3 с валидацией


Запускаем z3 с моделью

python3 src/calculation/run_z3_unite.py --dataset data/t_geom_lettered_validated.csv --limit 51

Получаем результаты чистого z3 с валидацией и дополняющей моделью - основной пайплайн