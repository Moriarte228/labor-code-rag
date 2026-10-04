from load_document import (retrieve_article_numbers, bm25_article_numbers,
                           hybrid_article_numbers, structure_search,
                           structure, article_titles)   # ← добавь импорт навигации и структуры
from results_log import log_experiment
from eval_set import eval_set
import numpy as np
import time


def evaluate(k, search_fn):           # ← search_fn: функция стратегии (query, k) → список статей
    hits = 0
    reciprocal_ranks = []

    for item in eval_set:
        question = item["question"]
        correct = item["relevant"]

        found = search_fn(question, k)    # ← зовём переданную стратегию, не хардкод
        correct_found = set(correct) & set(found)

        hits += len(correct_found) / len(correct)

        if len(correct_found) == 0:
            reciprocal_ranks.append(0)
        else:
            position = np.argmax(np.isin(np.array(found), correct))
            reciprocal_ranks.append(1 / (position + 1))

        time.sleep(0.5)               # ← пауза: навигация делает 2 LLM-вызова, бережём рейт-лимит

    recall = hits / len(eval_set)
    mrr = sum(reciprocal_ranks) / len(eval_set)
    return recall, mrr

def structure_strategy(query, k):
    """Обёртка навигации под интерфейс evaluate: (query, k) → список статей."""
    found = structure_search(query, structure, article_titles)
    return found[:k]      # навигация возвращает по релевантности, режем до k

if __name__ == "__main__":
    metrics = {}
    last_mrr = 0
    for k in [1, 3, 5, 10]:
        recall, mrr = evaluate(k, structure_strategy)    # ← передаём стратегию навигации
        metrics[f"recall@{k}"] = round(recall, 3)
        last_mrr = mrr
        print(f"k={k:>2}: Recall@{k}={recall:.2f} MRR={mrr:.2f}")
    metrics["mrr"] = round(last_mrr, 3)

    log_experiment("Навигация по структуре (LLM выбирает главы→статьи по заголовкам), 50 вопросов, чанки по абзацам", metrics)