
from sklearn.feature_extraction.text import TfidfVectorizer

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the selected article IDs
with open("data/selected_ids.json", "r", encoding="utf-8") as file:
    article_ids = json.load(file)

# Load the article texts
passages = []

for article_id in article_ids:
    file_name = article_id.removeprefix("high")
    file_path = Path("RACE/train/high") / file_name

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["id"] == article_id
    passages.append(data["article"])

# Convert passages into TF-IDF vectors
vectorizer = TfidfVectorizer()
passage_vectors = vectorizer.fit_transform(passages)

print("Total articles:", len(passages))
print("TF-IDF matrix shape:", passage_vectors.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


query = "What does the passage mainly talk about?"

query_vector = vectorizer.transform([query])

print("Query vector shape:", query_vector.shape)
print("Non-zero TF-IDF values:", query_vector.nnz)


# Calculate cosine similarities
scores = cosine_similarity(query_vector, passage_vectors)[0]

# Find the article with the highest score
best_index = scores.argmax()

print("Number of similarity scores:", len(scores))
print("Top-1 article ID:", article_ids[best_index])
print("Top-1 similarity score:", round(float(scores[best_index]), 4))


# Rank articles by similarity score
top_indices = scores.argsort()[::-1][:5]

print("\nTop-5 retrieved articles:")

for rank, index in enumerate(top_indices, start=1):
    article_id = article_ids[index]
    score = scores[index]

    print(f"{rank}. {article_id} | Score: {score:.4f}")

# Check whether the correct article appears in Top-5
correct_id = "high1417.txt"

top5_ids = [article_ids[i] for i in top_indices]

print("Correct article:", correct_id)
print("Correct article in Top-5:", correct_id in top5_ids)


# Prepare all questions for batch retrieval
queries = []

for article_id in article_ids:
    file_name = article_id.removeprefix("high")
    file_path = Path("RACE/train/high") / file_name

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for number, question in enumerate(data["questions"], start=1):
        queries.append({
            "question_id": f"{article_id}_q{number}",
            "text": question,
            "article_id": article_id
        })

print("\nTotal queries:", len(queries))
print("Unique query IDs:", len({q["question_id"] for q in queries}))
print("All ground-truth articles exist:",
      all(q["article_id"] in article_ids for q in queries))


# Convert all questions into TF-IDF vectors
query_texts = [q["text"] for q in queries]

query_vectors = vectorizer.transform(query_texts)

# Calculate similarities for all questions
similarity_matrix = cosine_similarity(
    query_vectors,
    passage_vectors
)

print("\nBatch retrieval:")
print("Query matrix shape:", query_vectors.shape)
print("Similarity matrix shape:", similarity_matrix.shape)


# Find the correct article rank for each question
ranks = []

for q, scores in zip(queries, similarity_matrix):

    ranked_indices = sorted(
    range(len(article_ids)),
    key=lambda i: (-float(scores[i]), article_ids[i])

)

    ranked_ids = [article_ids[i] for i in ranked_indices]

    correct_id = q["article_id"]

    correct_rank = ranked_ids.index(correct_id) + 1

    ranks.append(correct_rank)

print("\nRanking results:")
print("Total ranks:", len(ranks))
print("First five ranks:", ranks[:5])
print("First query correct rank:", ranks[0])


# Evaluate TF-IDF retrieval performance

total_queries = len(ranks)

hit1_count = sum(rank == 1 for rank in ranks)

hit5_count = sum(rank <= 5 for rank in ranks)

hit_at_1 = hit1_count / total_queries

hit_at_5 = hit5_count / total_queries

mrr = sum(1 / rank for rank in ranks) / total_queries

print("\nTF-IDF Evaluation Results:")
print("Total queries:", total_queries)
print("Hit@1:", round(hit_at_1, 4))
print("Hit@5:", round(hit_at_5, 4))
print("MRR:", round(mrr, 4))


# Save TF-IDF evaluation results

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

metrics = {
    "method": "TF-IDF",
    "dataset": "RACE-H",
    "num_documents": len(article_ids),
    "num_queries": total_queries,
    "hit_at_1": hit_at_1,
    "hit_at_5": hit_at_5,
    "mrr": mrr
}

with open(results_folder / "tfidf_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nEvaluation results saved successfully!")
print("File: results/tfidf_metrics.json")

# Check tied Top-1 similarity scores

top1_tie_count = 0

for row in similarity_matrix:
    highest_score = row.max()
    number_at_top = (row == highest_score).sum()

    if number_at_top > 1:
        top1_tie_count += 1

print("\nTF-IDF Tie Analysis:")
print("Queries with tied Top-1 scores:", top1_tie_count)
print("Total queries:", similarity_matrix.shape[0])



# Check ties across retrieval rankings

any_ties = 0
top5_boundary_ties = 0

for row in similarity_matrix:
    sorted_scores = sorted(row, reverse=True)

    if len(set(row)) < len(row):
        any_ties += 1

    if sorted_scores[4] == sorted_scores[5]:
        top5_boundary_ties += 1

print("\nDetailed TF-IDF Tie Analysis:")
print("Queries with any tied scores:", any_ties)
print("Queries tied at rank 5/6:", top5_boundary_ties)


# Check ties involving the correct article

correct_tie_count = 0
correct_zero_tie_count = 0

for q, row in zip(queries, similarity_matrix):

    correct_index = article_ids.index(q["article_id"])

    correct_score = row[correct_index]

    tied_count = (row == correct_score).sum()

    if tied_count > 1:
        correct_tie_count += 1

        if correct_score == 0:
            correct_zero_tie_count += 1

print("\nGround Truth Tie Analysis:")
print("Queries with tied correct scores:", correct_tie_count)
print("Correct articles tied at zero:", correct_zero_tie_count)



# Save per-question TF-IDF retrieval ranks

rank_records = []

for q, rank in zip(queries, ranks):
    rank_records.append({
        "question_id": q["question_id"],
        "question": q["text"],
        "correct_article": q["article_id"],
        "tfidf_rank": int(rank)
    })

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

with open(results_folder / "tfidf_ranks.json", "w", encoding="utf-8") as file:
    json.dump(rank_records, file, indent=2, ensure_ascii=False)

print("\nTF-IDF question ranks saved!")
print("Total records:", len(rank_records))
print("File: results/tfidf_ranks.json")
