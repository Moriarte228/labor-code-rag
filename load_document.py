import fitz #PyMuPDF
import re
import os
import requests
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi

def load_document(filename):
    """Извлекает текст блоками. Надстрочные цифры в номерах статей
    (29¹ → 29-1) обрабатываются через флаг superscript, иначе они
    схлопываются в 291 и ломают и нумерацию, и сопоставление с датасетом."""
    doc = fitz.open(filename)
    text = ""
    for page in doc:
        d = page.get_text("dict")
        blocks_out = []
        for block in d["blocks"]:
            if "lines" not in block:
                continue
            btext = ""
            for line in block["lines"]:
                for s in line["spans"]:
                    t = s["text"]
                    # флаг 1 = superscript; надстрочную цифру отделяем дефисом
                    if (s["flags"] & 1) and t.strip().isdigit():
                        btext += "-" + t
                    else:
                        btext += t
            btext = " ".join(btext.split())
            if btext:
                # (y, x, текст) для сортировки сверху-вниз, слева-направо
                blocks_out.append((block["bbox"][1], block["bbox"][0], btext))
        blocks_out.sort(key=lambda x: (x[0], x[1]))
        for _, _, bt in blocks_out:
            text += bt + "\n"
    return text

def prepare_parts(document):
    # \d+(?:-\d+)? — ловит и "Статья 28.", и "Статья 29-1."
    return re.split(r"(Статья \d+(?:-\d+)?\.)", document)


document = load_document("tk_rb_cleaned.pdf")
parts = prepare_parts(document)
for part in parts[50:70]:
    print(part)

def split_into_paragraphs(text, max_words=350):
    """Режет текст статьи на чанки по абзацам.
    Длинный абзац (> max_words) дополнительно режется по словам."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    # split("\n") — режем по переносам; if p.strip() — выкидываем пустые строки

    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            # абзац нормального размера → кладём целиком
            chunks.append(para)
        else:
            # редкий гигантский абзац → режем по словам кусками max_words
            for start in range(0, len(words), max_words):
                piece = " ".join(words[start: start + max_words])
                chunks.append(piece)
                pass
    return chunks

def cut_to_the_chunks(parts):
    metadata_list = []
    chunk_list = []
    for i in range(1, len(parts), 2):
        this_metadata = parts[i]
        this_chunk = parts[i + 1]
        paragraphs = split_into_paragraphs(this_chunk)
        this_match = re.search(r"\d+(?:-\d+)?", this_metadata)
        this_number_of_the_article = this_match.group() if this_match else None
        for paragraph in paragraphs:
            chunk_list.append(paragraph)
            metadata_list.append(this_number_of_the_article)  # метка к КАЖДОМУ куску
    return np.array(metadata_list), np.array(chunk_list)

metadata, chunks = cut_to_the_chunks(parts)
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def build_structure(filename):
    doc = fitz.open(filename)
    text = ""
    for page in doc:
        d = page.get_text("dict")
        bo = []
        for block in d["blocks"]:
            if "lines" not in block: continue
            bt = ""
            for line in block["lines"]:
                for s in line["spans"]:
                    t = s["text"]
                    if (s["flags"] & 1) and t.strip().isdigit(): bt += "-" + t
                    else: bt += t
            bt = " ".join(bt.split())
            if bt: bo.append((block["bbox"][1], block["bbox"][0], bt))
        bo.sort(key=lambda x: (x[0], x[1]))
        for _, _, b in bo:
            text += b + "\n"

    # ЧИСТКА: zero-width space \u200b → обычный пробел
    text = text.replace("\u200b", " ")

    # маппинг слипшихся номеров глав (надстрочные, где флаг не сработал)
    chapter_fix = {"181": "18-1", "251": "25-1", "261": "26-1"}

    structure = {}
    article_titles = {}
    current_chapter = None

    for line in text.split("\n"):
        line = line.strip()

        # глава: "Глава 4 Прекращение..." (регистр любой, \u200b уже убран)
        ch = re.match(r"Глава\s+(\d+)\.?\s*(.*)", line, re.IGNORECASE)
        if ch:
            num = ch.group(1)
            num = chapter_fix.get(num, num)   # починить слипшиеся
            title = ch.group(2).strip()
            if "исключена" in title.lower():
                current_chapter = None         # пропускаем исключённые главы
                continue
            current_chapter = f"Глава {num}"
            structure[current_chapter] = {"title": title, "articles": []}
            continue

        # статья: "Статья 42. Расторжение..."
        art = re.match(r"Статья\s+(\d+(?:-\d+)?)\.\s*(.*)", line)
        if art and current_chapter:
            n = art.group(1)
            structure[current_chapter]["articles"].append(n)
            article_titles[n] = art.group(2).strip()

    return structure, article_titles

structure, article_titles = build_structure("tk_rb_cleaned.pdf")

# 1. дефисные статьи на месте?
all_in_structure = [a for ch in structure.values() for a in ch["articles"]]
print("дефисные в структуре:", sorted([a for a in all_in_structure if "-" in a])[:10])

# 2. совпадают ли номера структуры с metadata?
import numpy as np
meta_set = set(metadata.tolist())
struct_set = set(all_in_structure)
print("в структуре, но НЕ в metadata:", struct_set - meta_set)   # должно быть пусто/почти
print("глав найдено:", len(structure))

for n in ["42", "34", "233", "155", "29-1"]:
    print(n, "—", article_titles.get(n, "ПУСТО"))

def bm25_article_numbers(query, k=10):

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)  # скор для каждого чанка
    order = np.argsort(scores)[::-1]  # индексы от лучшего к худшему
    seen = []
    for idx in order:
        article = metadata[idx]
        if article not in seen:
            seen.append(str(article))
        if len(seen) == k:
            break
    return seen


model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
document_embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True   # видно прогресс по 468 кускам
)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def get_article_text(number):
    mask = np.isin(metadata, [str(number)])   # True там, где метка совпадает
    return " ".join(chunks[mask])              # склеить все чанки этой статьи

print(get_article_text(28)[:200])   # должен быть текст про предварительное испытание
print(get_article_text(155)[:200])  # про отпуск

def rerank_article_number(query, candidates):
    pairs = [[query, get_article_text(candidate)] for candidate in candidates]
    reranker_scores = reranker.predict(pairs)
    result = sorted(zip(candidates, reranker_scores), key=lambda x: x[1], reverse=True)
    return [art for art, score in result]

def closest_match_list(query):
    query_embedding = model.encode(query, normalize_embeddings=True)
    closest_match = np.argsort(document_embeddings @ query_embedding)
    return closest_match



def retrieve_relevant(query):
    query_embedding = model.encode(query, normalize_embeddings=True)
    closest_match = np.argsort(document_embeddings @ query_embedding)[-10:]
    return np.unique(metadata[closest_match])


def send_the_message(message):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    message_body = {
        'model': 'openai/gpt-oss-120b:free',
        "messages": [{"role": "user", "content": f"{message}"}]
    }
    print("ключ есть:", bool(api_key), "| длина:", len(api_key) if api_key else 0)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=message_body
    )
    data = response.json()
    if "choices" not in data:
        return f"ERROR {data['error']['code']}: {data['error']['message']}"
    return data["choices"][0]["message"]["content"]




def retrieve_article_numbers(query, k=5, rerank=False):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = document_embeddings @ query_embedding
    order = np.argsort(scores)[::-1]   # индексы от лучшего к худшему
    seen = []
    for idx in order:
        article = metadata[idx]
        if article not in seen:
            seen.append(str(article))
        if len(seen) == k:
            break
    if rerank:
        return rerank_article_number(query, seen[:k] ) # реранкер из 30 → топ k
    else:
        return seen


def generate_hypothetical_answer(query):
    """HyDE: просим LLM написать фейковую 'статью кодекса', отвечающую на вопрос.
    Эмбеддить будем ЕЁ, а не сам вопрос — она ближе по жанру к реальным статьям."""
    hyde_prompt = (
        "Ты пишешь текст в стиле статьи Трудового кодекса Республики Беларусь. "
        "Напиши короткий фрагмент закона (2-4 предложения), который мог бы быть "
        "ответом на следующий вопрос. Пиши сухим юридическим языком, как в кодексе, "
        "без вводных слов, без 'вот ответ', сразу текст нормы. "
        "Не важно, точны ли факты — важен юридический стиль и термины.\n\n"
        f"Вопрос: {query}"
    )
    return send_the_message(hyde_prompt)

def hyde_article_numbers(query, k=10, rerank=False, candidates_n=30):
    """Поиск через HyDE: ищем по сгенерированному псевдо-ответу,
    но реранжируем по ИСХОДНОМУ вопросу."""
    # 1. генерируем фейковую статью и ищем ПО НЕЙ

    hypothetical = generate_hypothetical_answer(query)
    print (hypothetical)
    query_embedding = model.encode(hypothetical, normalize_embeddings=True)
    scores = document_embeddings @ query_embedding
    order = np.argsort(scores)[::-1]

    seen = []
    for idx in order:
        article = metadata[idx]
        if article not in seen:
            seen.append(str(article))
        if len(seen) == candidates_n:   # набираем больше кандидатов для реранкера
            break

    # 2. реранжируем по НАСТОЯЩЕМУ вопросу (не по фейковой статье!)
    if rerank:
        return rerank_article_number(query, seen)[:k]
    else:
        return seen[:k]



def hybrid_article_numbers(query, k=10, rrf_k=30, rerank=False):
    candidates_n = 40
    cosine_ranked = retrieve_article_numbers(query, k=40)
    bm25_ranked = bm25_article_numbers(query, k=10)

    scores = {}
    for i, article in enumerate(cosine_ranked, start=1):
        scores[article] = scores.get(article, 0) + 1 / (rrf_k + i)
    for i, article in enumerate(bm25_ranked):               # твой рабочий вариант со start=0
        scores[article] = scores.get(article, 0) + 1 / (rrf_k + i)

    ranked = sorted(scores, key=scores.get, reverse=True)

    if rerank:
        return rerank_article_number(query, ranked[:candidates_n])[:k]   # реранкер из 30 → топ k
    else:
        return ranked[:k]


def _normalize_spaces(text):
    """Убирает хитрые невидимые пробелы, которые ломают сравнение:
    \u200b (zero-width), \u202f (narrow no-break), \xa0 (no-break)."""
    text = text.replace("\u200b", " ").replace("\u202f", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def structure_search(query, structure, article_titles):
    """Навигация по оглавлению: LLM выбирает главы по названиям,
    потом статьи по заголовкам. Ещё одна стратегия наравне с косинусом."""

    # --- ШАГ 1: выбрать главы ---
    toc = "\n".join(f"{ch}: {data['title']}" for ch, data in structure.items())
    chapter_prompt = (
        "Вопрос пользователя: " + query + "\n\n"
        "Главы Трудового кодекса:\n" + toc + "\n\n"
        "Какие главы могут содержать ответ? Назови номера глав через запятую "
        "(обычно 1-2, можно больше если нужно). Формат: 'Глава 4, Глава 18'."
    )
    chosen = send_the_message(chapter_prompt)
    if not chosen or chosen.startswith("ERROR"):
        return []   # LLM не ответила — пусто, откат на другую стратегию

    # достаём НОМЕРА глав напрямую (устойчиво к любым пробелам и регистру)
    chosen_clean = _normalize_spaces(chosen)
    chapter_nums = re.findall(r"Глава\s*(\d+(?:-\d+)?)", chosen_clean, re.IGNORECASE)
    chapter_keys = [f"Глава {n}" for n in chapter_nums if f"Глава {n}" in structure]
    if not chapter_keys:
        return []

    # --- собрать заголовки статей выбранных глав ---
    candidates = []
    for ch in chapter_keys:
        for num in structure[ch]["articles"]:
            if num in article_titles:
                candidates.append((num, article_titles[num]))
    if not candidates:
        return []

    # --- ШАГ 2: выбрать статьи по заголовкам ---
    list_text = "\n".join(f"Статья {n}: {t}" for n, t in candidates)
    article_prompt = (
        "Вопрос пользователя: " + query + "\n\n"
        "Статьи выбранных глав:\n" + list_text + "\n\n"
        "Какие статьи отвечают на вопрос? Номера через запятую, "
        "по убыванию релевантности. Только номера, например: '42, 43'."
    )
    chosen_arts = send_the_message(article_prompt)
    if not chosen_arts or chosen_arts.startswith("ERROR"):
        return []

    found = re.findall(r"\d+(?:-\d+)?", _normalize_spaces(chosen_arts))
    valid = {n for n, _ in candidates}
    # сохраняем порядок LLM (по релевантности), убираем дубли и невалидные
    seen = []
    for a in found:
        if a in valid and a not in seen:
            seen.append(a)
    return seen


def get_a_prompt(query):
    relevant_articles = retrieve_relevant(query)  # список номеров статей
    mask = np.isin(metadata, relevant_articles)  # True там, где метка в списке
    list_of_relevant_chunks = chunks[mask]  # все куски этих статей
    prompt = ("Вот выдержки из Трудового кодекса: " + "\n\n".join(list_of_relevant_chunks) +
              f". Ответь на их основе на вопрос: {query}. Если ответа в выдержках нет — так и скажи")
    return prompt


from eval_set import eval_set

def structure_search_debug(query, structure, article_titles):
    toc = "\n".join(f"{ch}: {data['title']}" for ch, data in structure.items())
    chapter_prompt = (
        "Вопрос пользователя: " + query + "\n\n"
        "Главы Трудового кодекса:\n" + toc + "\n\n"
        "Какие главы могут содержать ответ? Назови номера через запятую. "
        "Только в формате: 'Глава 4, Глава 18'."
    )
    chosen = send_the_message(chapter_prompt)
    print(">>> LLM выбрала главы (сырой ответ):", repr(chosen))   # ← смотрим
    # НОРМАЛИЗАЦИЯ: все хитрые пробелы (\u202f, \u200b, \xa0 и пр.) → обычный пробел
    chosen_clean = re.sub(r"\s+", " ", chosen.replace("\u200b", " ").replace("\u202f", " ").replace("\xa0", " "))

    chapter_keys = [ch for ch in structure if re.search(rf"{re.escape(ch)}\b", chosen_clean)]
    print(">>> распознанные главы:", chapter_keys)

    candidates = []
    for ch in chapter_keys:
        for num in structure[ch]["articles"]:
            if num in article_titles:
                candidates.append((num, article_titles[num]))
    print(">>> кандидатов-статей:", len(candidates))             # ← смотрим

    if not candidates:
        return []

    list_text = "\n".join(f"Статья {n}: {t}" for n, t in candidates)
    article_prompt = (
        "Вопрос пользователя: " + query + "\n\n"
        "Статьи:\n" + list_text + "\n\n"
        "Какие статьи отвечают? Номера через запятую, например: '42, 43'."
    )
    chosen_arts = send_the_message(article_prompt)
    print(">>> LLM выбрала статьи (сырой ответ):", repr(chosen_arts))  # ← смотрим

    found = re.findall(r"\d+(?:-\d+)?", chosen_arts)
    valid = {n for n, _ in candidates}
    print("found:", found[:5], type(found[0]))
    print("valid примеры:", list(valid)[:5])
    print("'42' in valid?:", '42' in valid)
    print("пересечение:", set(found) & valid)
    return [a for a in found if a in valid]

if __name__ == "__main__":
    structure_search_debug("За что начальник может уволить по своей инициативе?", structure, article_titles)

