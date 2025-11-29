import pdfplumber
import faiss
import requests
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/Sample.pdf"
TOP_K = 10      

LOCAL_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def load_pdf_pages(path):
    """
    Read PDF pages and return a list of dictionaries:
    [{ "page": page_number, "text": page_text }, ...]
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def split_into_sentences(text):
    """
    Simple sentence splitter using punctuation.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_pages(pages, sentences_per_chunk=5):
    """
    Convert pages into smaller chunks containing ~N sentences each.
    Returns a list: [{ "page": page_number, "text": chunk_text }, ...]
    """
    chunks = []
    for p in pages:
        sents = split_into_sentences(p["text"])
        cur = []
        for s in sents:
            cur.append(s)
            if len(cur) >= sentences_per_chunk:
                chunk_text = " ".join(cur)
                chunks.append({
                    "page": p["page"],
                    "text": chunk_text
                })
                cur = []
        if cur:
            chunk_text = " ".join(cur)
            chunks.append({
                "page": p["page"],
                "text": chunk_text
            })
    return chunks

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    Convert a list of strings into embeddings (vectors).
    Returns a numpy array of shape (n_texts, embedding_dim).
    """
    return embed_model.encode(texts, convert_to_numpy = True, normalize_embeddings = True)

def build_index(chunks):
    """
    Build a FAISS index from the chunk texts.
    Returns:
      - index: the FAISS index object
      - embeddings: the numpy array of all chunk embeddings
    """
    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    return index, embeddings

def retrieve(query, index, chunks, top_k=TOP_K):
    """
    Given a user query, find the top_k most similar chunks
    from the document using the FAISS index.
    Returns a list of chunk dicts with an added 'score' field.
    """

    q_emb = embed_texts([query])[0].reshape(1, -1)

    D, I = index.search(q_emb, top_k)

    results= []
    for rank, idx in enumerate(I[0]):
        c = chunks[idx]
        c_copy = dict(c)
        c_copy["score"] = float(D[0][rank])
        results.append(c_copy)

    return results


def call_local_llm(question, retrieved_chunks, model=LOCAL_MODEL):
    """
    Call a local LLM running via Ollama using the retrieved context.
    """
    # Build context string from chunks (same as Groq)
    context_parts = []
    for c in retrieved_chunks:
        context_parts.append(f"[Page {c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant answering questions based strictly on the "
        "provided document excerpts. If the answer is not clearly present in the "
        "context, say you don't know. Always mention page numbers in brackets "
        "when you use information, for example [Page 3].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely, citing pages like [Page 3]."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Ollama returns the generated text in the 'response' field
    return data.get("response", "").strip()


def main():
    print(f"Reading PDF: {PDF_PATH}")
    pages = load_pdf_pages(PDF_PATH)
    print(f"Loaded {len(pages)} pages with text.")

    print("\nSample preview:\n", pages[0]["text"][:500], "\n---\n")

    print("Chunking pages...")
    chunks = chunk_pages(pages, sentences_per_chunk = 5)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS index...")
    index, _ = build_index(chunks)
    print("Index ready.\n")

    print("Local RAG is running with Ollama. Ask questions (type 'exit' to quit):\n")

    while True:
        q = input("Q: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        retrieved = retrieve(q, index, chunks, top_k=TOP_K)

        answer = call_local_llm(q, retrieved)

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    main()
    