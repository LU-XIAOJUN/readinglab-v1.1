
import json

# Load question-level retrieval results
with open("results/tfidf_ranks.json", "r", encoding="utf-8") as file:
    tfidf_records = json.load(file)

with open("results/semantic_ranks.json", "r", encoding="utf-8") as file:
    semantic_records = json.load(file)

# Match records using question IDs
tfidf_by_id = {
    record["question_id"]: record
    for record in tfidf_records
}

semantic_by_id = {
    record["question_id"]: record
    for record in semantic_records
}

# Validate the records
assert len(tfidf_by_id) == len(tfidf_records) == 673
assert len(semantic_by_id) == len(semantic_records) == 673
assert tfidf_by_id.keys() == semantic_by_id.keys()

# Count four retrieval outcomes
both_correct = 0
tfidf_only = 0
semantic_only = 0
neither_correct = 0

for question_id, tfidf in tfidf_by_id.items():

    semantic = semantic_by_id[question_id]

    assert tfidf["correct_article"] == semantic["correct_article"]
    assert tfidf["question"] == semantic["question"]

    tfidf_hit = tfidf["tfidf_rank"] == 1
    semantic_hit = semantic["semantic_rank"] == 1

    if tfidf_hit and semantic_hit:
        both_correct += 1

    elif tfidf_hit and not semantic_hit:
        tfidf_only += 1

    elif not tfidf_hit and semantic_hit:
        semantic_only += 1

    else:
        neither_correct += 1

print("\nError Analysis — Hit@1")
print("Both correct:", both_correct)
print("TF-IDF only:", tfidf_only)
print("MiniLM only:", semantic_only)
print("Neither correct:", neither_correct)

total = (
    both_correct + tfidf_only
    + semantic_only + neither_correct
)

print("Total questions classified:", total)
assert total == 673




# Select examples for qualitative error analysis

examples = {
    "TF-IDF only": [],
    "MiniLM only": [],
    "Neither correct": []
}

for question_id, tfidf in tfidf_by_id.items():

    semantic = semantic_by_id[question_id]

    t_rank = tfidf["tfidf_rank"]
    s_rank = semantic["semantic_rank"]

    if t_rank == 1 and s_rank > 1:
        category = "TF-IDF only"

    elif t_rank > 1 and s_rank == 1:
        category = "MiniLM only"

    elif t_rank > 1 and s_rank > 1:
        category = "Neither correct"

    else:
        continue

    if len(examples[category]) < 2:
        examples[category].append({
            "question_id": question_id,
            "question": tfidf["question"],
            "tfidf_rank": t_rank,
            "semantic_rank": s_rank
        })

# Display selected examples

for category, records in examples.items():

    print("\nCategory:", category)

    for record in records:
        print("Question ID:", record["question_id"])
        print("Question:", record["question"])
        print("TF-IDF rank:", record["tfidf_rank"])
        print("MiniLM rank:", record["semantic_rank"])
        print()
