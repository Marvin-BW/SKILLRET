"""
Step 2: Distribution analysis + stable group extraction

1) Extract action×object combos covering 80% of skills
2) Qwen embedding → kmeans consensus clustering (k=5,7,10,15,20)
3) Output strict/moderate/loose stable groups

Input:  outputs/skill_tags_clean.jsonl
Output: outputs/stable_groups.json
"""

import json, os
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
EMBEDDING_URL = "http://10.7.80.140:8000/v1"


def load_tags():
    tags = []
    with open(os.path.join(OUT_DIR, "skill_tags_clean.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("primary_object") == "document":
                d["primary_object"] = "documentation"
            tags.append(d)
    return tags


def embed_texts(texts):
    from openai import OpenAI
    client = OpenAI(base_url=EMBEDDING_URL, api_key="dummy")
    resp = client.embeddings.create(input=texts, model="Qwen/Qwen3-Embedding-8B")
    return np.array([d.embedding for d in resp.data])


def main():
    print("=" * 60)
    print("Step 2: Distribution analysis + stable group extraction")
    print("=" * 60)

    tags = load_tags()
    total = len(tags)
    print(f"Total skills: {total:,}")

    # ── 80% coverage combo extraction ──
    combo_counts = Counter((t["primary_action"], t["primary_object"]) for t in tags)
    cum = 0
    combos = []
    for (a, o), cnt in combo_counts.most_common():
        cum += cnt
        combos.append({"action": a, "object": o, "count": cnt})
        if cum / total >= 0.80:
            break

    print(f"80% coverage: {len(combos)} combos ({cum:,} skills)")

    # ── Embedding ──
    texts = [f"{c['action']} {c['object']}" for c in combos]
    embeddings = embed_texts(texts)
    print(f"Embeddings: {embeddings.shape}")

    # ── Consensus clustering ──
    n = len(combos)
    k_values = [5, 7, 10, 15, 20]
    co_matrix = np.zeros((n, n))

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    co_matrix[i][j] += 1
                    co_matrix[j][i] += 1

    runs = len(k_values)

    def find_groups(threshold):
        visited = set()
        groups = []
        for i in range(n):
            if i in visited:
                continue
            group = {i}
            queue = [i]
            while queue:
                node = queue.pop(0)
                for j in range(n):
                    if j not in group and co_matrix[node][j] >= threshold:
                        group.add(j)
                        queue.append(j)
            if len(group) >= 2:
                visited.update(group)
                groups.append(sorted(group))
        singletons = [i for i in range(n) if i not in visited]
        return groups, singletons

    results = {}
    for name, thresh in [("strict", runs), ("moderate", runs - 1), ("loose", runs - 2)]:
        groups, singletons = find_groups(thresh)
        group_data = [
            [(combos[i]["action"], combos[i]["object"], combos[i]["count"]) for i in g]
            for g in groups
        ]
        singleton_data = [(combos[i]["action"], combos[i]["object"], combos[i]["count"]) for i in singletons]
        results[name] = {"groups": group_data, "singletons": singleton_data}

        total_in = sum(c for g in group_data for _, _, c in g)
        print(f"\n[{name}] threshold={thresh}/{runs}")
        print(f"  Groups: {len(group_data)} ({total_in:,}), Singletons: {len(singleton_data)}")

        for i, g in enumerate(group_data):
            members = ", ".join(f"{a}×{o}({c:,})" for a, o, c in g)
            print(f"  G{i+1} [{sum(c for _,_,c in g):,}]: {members}")

    with open(os.path.join(OUT_DIR, "stable_groups.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: outputs/stable_groups.json")


if __name__ == "__main__":
    main()
