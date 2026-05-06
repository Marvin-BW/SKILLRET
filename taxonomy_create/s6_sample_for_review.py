"""
Step 6: Sample 200 skills for human review (stratified by Major x Sub)
======================================================================

Output: data/tagging_review_200_v2.csv

Columns match the previous review CSV so reviewers can work with the same
template:
  No, skill_id, name, description, LLM_action, LLM_object, LLM_domain,
  assigned_major, assigned_sub, llm_major, llm_sub,
  major_correct(Y/N), sub_correct(Y/N), notes

Stratified counts per Major (total 200):
  Software Engineering     120
  AI Agents                 20
  Business & Planning       20
  Data & ML                 15
  Content Creation          15
  Information Retrieval     10
Within each Major, Subs get proportional allocation with a minimum of 2.
"""

import json, os, csv, random
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

SEED = 42
MAJOR_QUOTA = {
    "Software Engineering":  120,
    "AI Agents":              20,
    "Business & Planning":    20,
    "Data & ML":              15,
    "Content Creation":       15,
    "Information Retrieval":  10,
}
MIN_PER_SUB = 2


def load():
    tags = {}
    tag_path = os.path.join(OUT_DIR, "skill_tags_clean.jsonl")
    if os.path.exists(tag_path):
        with open(tag_path) as f:
            for line in f:
                d = json.loads(line)
                tags[d["id"]] = d

    llm_path = os.path.join(OUT_DIR, "skill_assignments_llm.jsonl")
    rule_path = os.path.join(OUT_DIR, "skill_assignments.jsonl")
    assign_path = llm_path if os.path.exists(llm_path) else rule_path
    print(f"  Assignments source: {os.path.basename(assign_path)}")

    assigns = {}
    with open(assign_path) as f:
        for line in f:
            d = json.loads(line)
            assigns[d["id"]] = d

    meta = {}
    with open(os.path.join(DATA_DIR, "filtered_skills.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            meta[d["id"]] = d
    return tags, assigns, meta


def allocate(pool, quota):
    """Allocate `quota` samples across sub-pools proportionally,
    giving each non-empty sub at least MIN_PER_SUB when possible."""
    total = sum(len(v) for v in pool.values())
    if total == 0:
        return {}
    raw = {sub: len(v) / total * quota for sub, v in pool.items()}
    alloc = {sub: max(MIN_PER_SUB, int(round(raw[sub]))) for sub in pool}
    # adjust to exact quota
    while sum(alloc.values()) > quota:
        biggest = max(alloc, key=alloc.get)
        if alloc[biggest] <= MIN_PER_SUB:
            break
        alloc[biggest] -= 1
    while sum(alloc.values()) < quota:
        smallest_gap = min(pool, key=lambda s: alloc[s] - raw[s])
        alloc[smallest_gap] += 1
    # cap to available pool size
    for sub in alloc:
        alloc[sub] = min(alloc[sub], len(pool[sub]))
    # top up if any sub was capped
    total_alloc = sum(alloc.values())
    while total_alloc < quota:
        expandable = [s for s in alloc if alloc[s] < len(pool[s])]
        if not expandable:
            break
        for sub in sorted(expandable, key=lambda s: len(pool[s]) - alloc[s], reverse=True):
            alloc[sub] += 1
            total_alloc += 1
            if total_alloc >= quota:
                break
    return alloc


def main():
    random.seed(SEED)
    print("=" * 60)
    print("Step 6: Stratified 200-sample for taxonomy review")
    print("=" * 60)

    tags, assigns, meta = load()

    by_major = defaultdict(lambda: defaultdict(list))
    for sid, a in assigns.items():
        by_major[a["major"]][a["sub"]].append(sid)

    selected = []
    print(f"\n{'Major':<25} {'Sub':<28} {'alloc':>6} {'pool':>6}")
    print("-" * 68)
    for major, quota in MAJOR_QUOTA.items():
        pool = by_major.get(major, {})
        alloc = allocate(pool, quota)
        for sub in sorted(alloc, key=lambda s: -alloc[s]):
            n = alloc[sub]
            sids = random.sample(pool[sub], n)
            for sid in sids:
                selected.append({"id": sid, "major": major, "sub": sub})
            print(f"{major:<25} {sub:<28} {n:>6} {len(pool[sub]):>6}")

    print("-" * 68)
    print(f"{'TOTAL':<25} {'':<28} {len(selected):>6}")

    rows = []
    for i, s in enumerate(selected, 1):
        t = tags.get(s["id"], {})
        m = meta.get(s["id"], {})
        rows.append({
            "No": i,
            "skill_id": s["id"],
            "name": m.get("name", ""),
            "description": m.get("description", ""),
            "LLM_action": t.get("primary_action", ""),
            "LLM_object": t.get("primary_object", ""),
            "LLM_domain":  t.get("domain", ""),
            "assigned_major": s["major"],
            "assigned_sub":   s["sub"],
            "llm_major": "",
            "llm_sub":   "",
            "major_correct(Y/N)": "",
            "sub_correct(Y/N)":   "",
            "notes": "",
        })

    out_csv = os.path.join(DATA_DIR, "tagging_review_200_v2.csv")
    cols = ["No","skill_id","name","description",
            "LLM_action","LLM_object","LLM_domain",
            "assigned_major","assigned_sub","llm_major","llm_sub",
            "major_correct(Y/N)","sub_correct(Y/N)","notes"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {out_csv}")
    print(f"Rows : {len(rows)}  (seed={SEED})")

    m_cnt = Counter(r["assigned_major"] for r in rows)
    s_cnt = Counter((r["assigned_major"], r["assigned_sub"]) for r in rows)
    print("\nSanity check — Major distribution:")
    for m, c in m_cnt.most_common():
        print(f"  {m:<25} {c:>3}")


if __name__ == "__main__":
    main()
