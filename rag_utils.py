from functools import lru_cache
from io import BytesIO

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_embedding_model():
    # Load once and reuse it during this Python process.
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )


def build_index(pdf_bytes):
    model = load_embedding_model()
    tokenizer = model.tokenizer

    reader = PdfReader(BytesIO(pdf_bytes))
    
    
    if reader.is_encrypted:
        raise ValueError(
            "This PDF is encrypted. Please upload an unencrypted copy "
            "that you have permission to use."
        )

    if len(reader.pages) > 100:
        raise ValueError(
            "This PDF has more than 100 pages. "
            "Please split it into smaller PDFs."
        )
        
        
    chunks = []
    empty_pages = []

    # Stay below the model's input limit.
    chunk_size = min(200, model.max_seq_length - 2)
    overlap = 40
    step = chunk_size - overlap

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            empty_pages.append(page_number)
            continue

        # Token offsets let us keep the original text in each excerpt.
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True
        )

        offsets = encoded["offset_mapping"]

        if not offsets:
            empty_pages.append(page_number)
            continue

        for start in range(0, len(offsets), step):
            end = min(start + chunk_size, len(offsets))

            first_character = offsets[start][0]
            last_character = offsets[end - 1][1]

            chunk_text = text[first_character:last_character]

            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

            if end == len(offsets):
                break

    if not chunks:
        raise ValueError(
            "No readable text found. Use a PDF with selectable text."
        )

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=16,
        show_progress_bar=False
    )

    # Keep the same four-part structure expected by app.py.
    return chunks, model, embeddings, empty_pages


def retrieve(question, index, top_k=4, min_score=0.25):
    chunks, model, embeddings, _ = index

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    scores = (question_embedding @ embeddings.T)[0]
    best_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index_number in best_indices:
        score = float(scores[index_number])

        if score >= min_score:
            results.append({
                "page": chunks[index_number]["page"],
                "text": chunks[index_number]["text"],
                "score": score
            })

    return results
