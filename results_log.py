"""
Журнал экспериментов.

Идея (твоя): имя эксперимента — это ИНДЕКС строки. Запись по индексу означает,
что повторный запуск того же эксперимента ПЕРЕЗАПИШЕТ свою строку, а не добавит
дубликат. Так журнал всегда отражает последний результат каждого эксперимента,
и его можно спокойно перезапускать.

Хранится в results.csv. Открываешь хоть в Excel, хоть глазами.
"""

import os
import pandas as pd
from datetime import datetime

RESULTS_FILE = "results.csv"

# какие метрики храним как колонки
COLUMNS = ["timestamp", "recall@1", "recall@3", "recall@5", "recall@10", "mrr"]


def log_experiment(experiment_name, metrics):
    """
    experiment_name : str   — имя эксперимента, оно же индекс строки
    metrics         : dict  — например {"recall@1": 0.53, "recall@5": 0.60, "mrr": 0.57}
                              можно передавать не все ключи — пропущенные останутся пустыми

    Повторный вызов с тем же experiment_name перезапишет строку (а не добавит новую).
    """
    # 1. загрузить существующий журнал или создать пустой с нужными колонками
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE, index_col=0)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        df.index.name = "experiment"

    # 2. собрать строку: время + переданные метрики
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    row.update(metrics)

    # 3. записать ПО ИНДЕКСУ — это и есть "перезапись по имени эксперимента".
    #    df.loc[имя] = ...  создаёт строку, если имени ещё нет в индексе,
    #    и ПЕРЕЗАПИСЫВАЕТ её целиком, если имя уже есть. Ровно твоя идея.
    df.loc[experiment_name] = [row.get(col, None) for col in COLUMNS]

    # 4. сохранить
    df.to_csv(RESULTS_FILE)
    return df


def show():
    """Показать весь журнал."""
    if not os.path.exists(RESULTS_FILE):
        print("Журнал пуст — ещё не было экспериментов.")
        return None
    df = pd.read_csv(RESULTS_FILE, index_col=0)
    print(df.to_string())
    return df


if __name__ == "__main__":
    show()