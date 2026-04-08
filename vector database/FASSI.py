import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# -------------------------------
# Task 1: Basic Vector Search
# -------------------------------
def task1_basic_vector_search():
    print("=== Task 1: Basic Vector Search ===")

    d = 64   # dimension
    nb = 100 # database size
    nq = 5   # number of queries

    np.random.seed(1234)

    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    index = faiss.IndexFlatL2(d)
    index.add(xb)

    k = 4
    D, I = index.search(xq, k)

    print("Top neighbors indices:\n", I)
    print("Distances:\n", D)
    print()


# -------------------------------
# Task 2: Semantic Search
# -------------------------------
def task2_semantic_search():
    print("=== Task 2: Semantic Search ===")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast fox leaps over a sleepy dog.",
        "AI is transforming the world.",
        "Machine learning is improving daily.",
        "It is raining outside."
    ]

    embeddings = model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = "A fox jumps over a resting dog."
    query_emb = model.encode([query])

    D, I = index.search(query_emb, 2)

    print("Query:", query)
    for i, idx in enumerate(I[0]):
        print(f"Rank {i+1}: {sentences[idx]} (Dist: {D[0][i]:.4f})")
    print()


# -------------------------------
# Task 3: Context Retrieval
# -------------------------------
def task3_context_retrieval():
    print("=== Task 3: Context-Based Retrieval ===")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    docs = [
        {"context": "Science", "text": "Photosynthesis helps plants make food."},
        {"context": "Science", "text": "Gravity pulls objects toward Earth."},
        {"context": "History", "text": "Industrial Revolution changed manufacturing."},
        {"context": "Tech", "text": "Cloud computing provides online resources."}
    ]

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def search(query, context=None):
        q_emb = model.encode([query])
        D, I = index.search(q_emb, len(docs))

        results = []
        for idx in I[0]:
            if context and docs[idx]["context"] != context:
                continue
            results.append(docs[idx]["text"])
        return results[:2]

    print(search("How do plants get food?"))
    print(search("Manufacturing changes?", context="History"))
    print()


# -------------------------------
# Task 4: Duplicate Detection
# -------------------------------
def task4_duplicate_detection():
    print("=== Task 4: Duplicate Detection ===")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    data = [
        "Company revenue increased in Q3.",
        "Company profits grew in third quarter.",
        "Opening new office in London.",
        "New London branch will open soon.",
        "Buy groceries today.",
        "Sky is clear tonight."
    ]

    embeddings = model.encode(data)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    D, I = index.search(embeddings, 2)

    threshold = 1.0
    print("Duplicate pairs:")

    for i in range(len(data)):
        j = I[i][1]
        if D[i][1] < threshold:
            print(f"\nPair:")
            print(data[i])
            print(data[j])


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    task1_basic_vector_search()
    task2_semantic_search()
    task3_context_retrieval()
    task4_duplicate_detection()