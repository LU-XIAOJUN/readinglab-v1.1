
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pretrained embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Test the model with one English question
question = "What does the passage mainly talk about?"

embedding = model.encode(question)

print("Model loaded successfully!")
print("Embedding shape:", embedding.shape)
print("Embedding dimension:", len(embedding))


import json
from pathlib import Path

# Load the same 200 article IDs as TF-IDF
with open("data/selected_ids.json", "r", encoding="utf-8") as file:
    article_ids = json.load(file)

passages = []

for article_id in article_ids:
    file_name = article_id.removeprefix("high")
    file_path = Path("RACE/train/high") / file_name

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["id"] == article_id
    passages.append(data["article"])

# Check token lengths
max_tokens = model.max_seq_length

token_counts = [
    len(model.tokenizer.encode(text, truncation=False))
    for text in passages
]

long_articles = sum(count > max_tokens for count in token_counts)

print("\nArticle length analysis:")
print("Total articles:", len(passages))
print("Model token limit:", max_tokens)
print("Articles exceeding limit:", long_articles)
print("First article token count:", token_counts[0])



# Split the first article into smaller chunks

first_passage = passages[0]

token_ids = model.tokenizer.encode(
    first_passage,
    add_special_tokens=False
)

chunk_size = 240

chunks = [
    token_ids[i:i + chunk_size]
    for i in range(0, len(token_ids), chunk_size)
]

chunk_texts = [
    model.tokenizer.decode(chunk, skip_special_tokens=True)
    for chunk in chunks
]

chunk_lengths = [
    len(model.tokenizer.encode(text))
    for text in chunk_texts
]

print("\nChunking test:")
print("Number of chunks:", len(chunks))
print("Chunk token lengths:", chunk_lengths)
print("All chunks within model limit:",
      all(length <= max_tokens for length in chunk_lengths))


# Generate embeddings for the first article's chunks

chunk_embeddings = model.encode(chunk_texts)

print("\nChunk embeddings:")
print("Chunk embeddings shape:", chunk_embeddings.shape)

# Average the chunk embeddings
article_embedding = chunk_embeddings.mean(axis=0)

print("Article embedding shape:", article_embedding.shape)


# Generate embeddings for all 200 articles

article_embeddings = []
total_chunks = 0

for index, passage in enumerate(passages):

    token_ids = model.tokenizer.encode(
        passage,
        add_special_tokens=False
    )

    chunks = [
        token_ids[i:i + 240]
        for i in range(0, len(token_ids), 240)
    ]

    chunk_texts = [
        model.tokenizer.decode(chunk, skip_special_tokens=True)
        for chunk in chunks
    ]

    chunk_lengths = [
        len(model.tokenizer.encode(text))
        for text in chunk_texts
    ]

    assert all(length <= max_tokens for length in chunk_lengths), (
        f"Chunk too long: {article_ids[index]}"
    )

    embeddings = model.encode(chunk_texts)

    article_vector = embeddings.mean(axis=0)

    article_embeddings.append(article_vector)

    total_chunks += len(chunks)

    if (index + 1) % 50 == 0:
        print("Processed articles:", index + 1)

# Combine all article vectors into one matrix
article_matrix = np.vstack(article_embeddings)

print("\nAll article embeddings:")
print("Total articles:", len(article_embeddings))
print("Total chunks:", total_chunks)
print("Article embedding matrix shape:", article_matrix.shape)


# Prepare all 673 reading comprehension questions

queries = []

for article_id in article_ids:
    file_name = article_id.removeprefix("high")
    file_path = Path("RACE/train/high") / file_name

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["id"] == article_id

    for number, question in enumerate(data["questions"], start=1):
        queries.append({
            "question_id": f"{article_id}_q{number}",
            "text": question,
            "article_id": article_id
        })

query_texts = [q["text"] for q in queries]

# Check question lengths before encoding
query_lengths = [
    len(model.tokenizer.encode(text, truncation=False))
    for text in query_texts
]

assert all(length <= max_tokens for length in query_lengths)

# Generate question embeddings
query_embeddings = model.encode(
    query_texts,
    batch_size=32,
    show_progress_bar=True
)

print("\nAll question embeddings:")
print("Total questions:", len(queries))
print("Unique question IDs:", len({q["question_id"] for q in queries}))
print("Query embedding matrix shape:", query_embeddings.shape)


# Calculate semantic cosine similarities

semantic_scores = cosine_similarity(
    query_embeddings,
    article_matrix
)

print("\nSemantic Retrieval:")
print("Similarity matrix shape:", semantic_scores.shape)

# Retrieve the top-ranked article for the first question
first_query_scores = semantic_scores[0]

best_index = first_query_scores.argmax()

print("First query:", queries[0]["text"])
print("Correct article:", queries[0]["article_id"])
print("Top-1 retrieved article:", article_ids[best_index])
print("Top-1 similarity:", round(float(first_query_scores[best_index]), 4))


# Rank articles for the first query

ranked_indices = first_query_scores.argsort()[::-1]

print("\nSemantic Top-5 retrieved articles:")

for rank, index in enumerate(ranked_indices[:5], start=1):
    article_id = article_ids[index]
    score = first_query_scores[index]

    print(f"{rank}. {article_id} | Score: {score:.4f}")

# Find the correct article's rank

ranked_ids = [article_ids[i] for i in ranked_indices]

correct_id = queries[0]["article_id"]

correct_rank = ranked_ids.index(correct_id) + 1

print("\nCorrect article:", correct_id)
print("Correct article rank:", correct_rank)
print("Correct article in Top-5:", correct_rank <= 5)


# Evaluate semantic retrieval for all 673 questions

ranks = []

for q, scores in zip(queries, semantic_scores):


    ranked_indices = sorted(
        range(len(article_ids)),
        key=lambda i: (-float(scores[i]), article_ids[i])
    )

    ranked_ids = [article_ids[i] for i in ranked_indices]

    correct_id = q["article_id"]

    correct_rank = ranked_ids.index(correct_id) + 1

    ranks.append(correct_rank)

# Calculate evaluation metrics

total_queries = len(ranks)

hit_at_1 = sum(rank == 1 for rank in ranks) / total_queries

hit_at_5 = sum(rank <= 5 for rank in ranks) / total_queries

mrr = sum(1 / rank for rank in ranks) / total_queries

print("\nSemantic Retrieval Evaluation:")
print("Total ranks:", len(ranks))
print("First five ranks:", ranks[:5])
print("Hit@1:", round(hit_at_1, 4))
print("Hit@5:", round(hit_at_5, 4))
print("MRR:", round(mrr, 4))


# Save semantic retrieval evaluation results

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

metrics = {
    "method": "Sentence Transformers",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dataset": "RACE-H",
    "num_documents": len(article_ids),
    "num_queries": total_queries,
    "total_chunks": total_chunks,
    "chunk_size": 240,
    "pooling": "mean",
    "hit_at_1": hit_at_1,
    "hit_at_5": hit_at_5,
    "mrr": mrr
}

with open(results_folder / "semantic_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nSemantic evaluation results saved!")
print("File: results/semantic_metrics.json")


# Save per-question MiniLM retrieval ranks

rank_records = []

for q, rank in zip(queries, ranks):
    rank_records.append({
        "question_id": q["question_id"],
        "question": q["text"],
        "correct_article": q["article_id"],
        "semantic_rank": int(rank)
    })

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

with open(results_folder / "semantic_ranks.json", "w", encoding="utf-8") as file:
    json.dump(rank_records, file, indent=2, ensure_ascii=False)

print("\nMiniLM question ranks saved!")
print("Total records:", len(rank_records))
print("File: results/semantic_ranks.json")
