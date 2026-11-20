# ReadingLab V1.1

**Reading Passage Retrieval: Comparing Lexical and Semantic Retrieval for English Reading Comprehension**

## 1. Project Overview

ReadingLab is an independent Python project developed alongside the Information Retrieval and Text Mining course in the M.Sc. Computational Linguistics program at the University of Stuttgart.

The project connects information retrieval techniques with my previous experience in high school English reading instruction.

Its goal is to investigate how effectively an English reading comprehension question can retrieve its original reading passage from a collection of candidate articles.

Two retrieval approaches are compared:

- **Lexical Retrieval:** TF-IDF with Cosine Similarity
- **Semantic Retrieval:** Sentence Transformers with Cosine Similarity

## 2. Dataset

The project uses the high-school subset of the RACE reading comprehension dataset (RACE-H).

The current experiment includes:

- 200 distinct reading passages
- 673 reading comprehension questions
- One ground-truth passage for each question

The passages are selected from `RACE/train/high` using a fixed random seed of 42.

The dataset is not included in this repository. Users must obtain it separately from the original dataset distribution.

## 3. Retrieval Methods

### TF-IDF Baseline

Reading passages are converted into TF-IDF vectors using scikit-learn.

Each question is represented in the same vector space. Cosine similarity is then used to rank candidate passages.

### Semantic Retrieval

The semantic method uses the pretrained Sentence Transformer model `sentence-transformers/all-MiniLM-L6-v2`.

Because many reading passages exceed the model's input length, articles are divided into chunks containing up to 240 content tokens.

Each chunk is encoded into a 384-dimensional embedding. Chunk embeddings are averaged to obtain one vector per passage.

Question embeddings are compared with passage embeddings using cosine similarity.

Both approaches rank the same 200 candidate passages. Tied scores are resolved using alphabetical article-ID order.

## 4. Evaluation

The retrieval methods are evaluated using:

- **Hit@1:** Proportion of questions whose correct passage ranks first.
- **Hit@5:** Proportion of questions whose correct passage appears in the top five.
- **MRR:** Mean Reciprocal Rank of the correct passages.

### Experimental Results

| Metric | TF-IDF | MiniLM |
|---|---:|---:|
| Hit@1 | 0.5230 | 0.4889 |
| Hit@5 | 0.6018 | 0.5810 |
| MRR | 0.5653 | 0.5337 |

Under the current experimental setup, TF-IDF outperforms MiniLM on all three metrics.

These results are specific to the selected RACE-H sample, retrieval configuration, and pretrained semantic model. They do not establish that lexical retrieval is generally superior to semantic retrieval.

## 5. Preliminary Error Analysis

Question-level Hit@1 results show:

| Outcome | Questions |
|---|---:|
| Both methods correct | 291 |
| TF-IDF only correct | 61 |
| MiniLM only correct | 38 |
| Neither method correct | 283 |

The results suggest that lexical and semantic retrieval provide complementary successes on some questions.

Initial inspection also suggests that questions with little topic-specific information may be challenging for both approaches. This explanation requires further investigation using the corresponding reading passages.

## 6. Installation

The project was developed using Python 3.11.

Install the required Python packages:

`python -m pip install -r requirements.txt`

Download the RACE dataset separately and extract it into the project directory so that the path `RACE/train/high` exists.

On first execution, the semantic retrieval program may download the pretrained MiniLM model.

## 7. Running the Experiments

Run the following commands from the project root, in order:

1. `python sample_200.py` — Select 200 passages and save their IDs.
2. `python tfidf_baseline.py` — Run TF-IDF retrieval and evaluation.
3. `python semantic_baseline.py` — Run MiniLM retrieval and evaluation.
4. `python compare_results.py` — Compare evaluation metrics.
5. `python error_analysis.py` — Analyze question-level retrieval outcomes.

Summary metrics are saved in the `results/` directory.

## 8. Limitations and Future Work

The current version is a small-scale exploratory retrieval experiment.

Its limitations include:

- Evaluation on a fixed subset of 200 passages.
- No independent held-out evaluation across multiple samples.
- A simple mean-pooling method for long passage embeddings.
- Limited qualitative analysis of question types and retrieval errors.
- No statistical significance testing.

Future work may investigate larger passage collections, alternative passage representations, and differences between reading comprehension question types.

## 9. Data and Reproducibility

The repository contains independently developed project code and summary evaluation results.

RACE source passages, original questions, and question-level records containing dataset text are excluded from the public repository.

The sample selection uses a fixed seed of 42. Full reproducibility across fresh environments has not yet been independently verified.

## 10. Project Context

ReadingLab was developed as a learning-oriented computational linguistics project connecting information retrieval coursework with practical questions from English reading comprehension instruction.

The focus is on implementing, evaluating, and understanding retrieval methods rather than building a production retrieval service.
