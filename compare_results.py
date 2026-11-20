
import json

# Load TF-IDF results
with open("results/tfidf_metrics.json", "r") as file:
    tfidf = json.load(file)

# Load semantic retrieval results
with open("results/semantic_metrics.json", "r") as file:
    semantic = json.load(file)

# Check dataset sizes
assert tfidf["dataset"] == semantic["dataset"]
assert tfidf["num_documents"] == semantic["num_documents"]
assert tfidf["num_queries"] == semantic["num_queries"]

# Compare evaluation metrics
metrics = ["hit_at_1", "hit_at_5", "mrr"]

print("ReadingLab Retrieval Comparison")

for metric in metrics:
    tfidf_score = tfidf[metric]
    semantic_score = semantic[metric]

    difference = semantic_score - tfidf_score

    print(f"\nMetric: {metric}")
    print(f"TF-IDF: {tfidf_score:.4f}")
    print(f"MiniLM: {semantic_score:.4f}")
    print(f"Difference: {difference:+.4f}")
