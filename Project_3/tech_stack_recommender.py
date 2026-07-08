import csv
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "raw_skills.csv")
TOP_N = 3


def load_job_roles(path):
    roles = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = [s.strip().lower() for s in row["skills"].split(",") if s.strip()]
            roles.append({"role": row["job_role"], "skills": skills})
    return roles


def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    return sorted(vocab)


def compute_idf(documents, vocab):
    total_docs = len(documents)
    idf = {}
    for term in vocab:
        containing = sum(1 for doc in documents if term in doc)
        containing = max(containing, 1)
        idf[term] = math.log(total_docs / containing)
    return idf


def compute_tf(document):
    counts = Counter(document)
    total = len(document) if document else 1
    return {term: count / total for term, count in counts.items()}


def vectorize(document, vocab, idf):
    tf = compute_tf(document)
    return [tf.get(term, 0.0) * idf.get(term, 0.0) for term in vocab]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def recommend(user_skills, roles, vocab, idf, top_n=TOP_N):
    user_doc = [s.strip().lower() for s in user_skills if s.strip()]
    user_vector = vectorize(user_doc, vocab, idf)

    scored = []
    for role in roles:
        role_vector = vectorize(role["skills"], vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scored.append((role["role"], score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def get_user_skills():
    print("Enter at least 3 skills or interests (comma-separated):")
    raw = input("> ").strip()
    skills = [s.strip() for s in raw.split(",") if s.strip()]
    while len(skills) < 3:
        print("Please enter at least 3 skills.")
        raw = input("> ").strip()
        skills = [s.strip() for s in raw.split(",") if s.strip()]
    return skills


def main():
    roles = load_job_roles(DATASET_PATH)
    vocab = build_vocabulary([role["skills"] for role in roles])
    idf = compute_idf([role["skills"] for role in roles], vocab)

    user_skills = get_user_skills()
    results = recommend(user_skills, roles, vocab, idf)

    if all(score == 0 for _, score in results):
        print("\nNo strong matches found in the current dataset. Showing trending roles instead:")
        results = [(role["role"], 0.0) for role in roles[:TOP_N]]

    print(f"\nTop {TOP_N} recommended career paths for {', '.join(user_skills)}:\n")
    for rank, (role, score) in enumerate(results, start=1):
        print(f"{rank}. {role} — match score: {score:.2f}")


if __name__ == "__main__":
    sys.exit(main())
