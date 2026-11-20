
import json
import random
from pathlib import Path

folder = Path("RACE/train/high")

files = sorted(folder.glob("*.txt"))

selected_files = random.Random(42).sample(files, 200)

documents = []

for file_path in selected_files:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents.append(data)

print("Total articles:", len(documents))
print("Unique IDs:", len({doc["id"] for doc in documents}))
print("Unique passages:", len({doc["article"] for doc in documents}))
print("Total questions:", sum(len(doc["questions"]) for doc in documents))


queries = []

for doc in documents:
    for number, question in enumerate(doc["questions"], start=1):
        queries.append({
            "question_id": f"{doc['id']}_q{number}",
            "text": question,
            "article_id": doc["id"]
        })

print("Total query records:", len(queries))
print("Unique question IDs:", len({q["question_id"] for q in queries}))
print("First query record:", queries[0])


output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

selected_ids = [doc["id"] for doc in documents]

with open(output_folder / "selected_ids.json", "w", encoding="utf-8") as file:
    json.dump(selected_ids, file, indent=2)

print("Saved article IDs:", len(selected_ids))
print("Output:", output_folder / "selected_ids.json")
