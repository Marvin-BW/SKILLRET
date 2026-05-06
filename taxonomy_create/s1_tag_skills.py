"""
Step 1: 2-pass LLM skill tagging

Pass 1: Send all skill names (name + desc 100 chars) to LLM
        to discover natural primary_action / primary_object / domain categories
Pass 2: Inject discovered categories into prompt, batch-classify each skill

Usage:
  python s1_tag_skills.py pass1          # Pass 1: Discover categories
  python s1_tag_skills.py pass2          # Pass 2: Tag all skills
  python s1_tag_skills.py pass2 --resume # Pass 2: Resume
  python s1_tag_skills.py clean          # Deduplicate

Input:  data/filtered_skills.jsonl (s0_filter.py output, 17,810 skills)
Output:
  outputs/discovered_categories.json  (Pass 1)
  outputs/skill_tags.jsonl            (Pass 2 raw)
  outputs/skill_tags_clean.jsonl      (Pass 2 deduplicated, 17,810 skills)

NOTE: Skip if outputs/skill_tags_clean.jsonl already exists (~$15 API cost)
"""

import json, os, sys, argparse
import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "filtered_skills.jsonl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "outputs", "discovered_categories.json")
TAGS_PATH = os.path.join(BASE_DIR, "outputs", "skill_tags.jsonl")
CLEAN_PATH = os.path.join(BASE_DIR, "outputs", "skill_tags_clean.jsonl")

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 100


# ─── Pass 1: Discover categories ─────────────────────────────

PASS1_SYSTEM = """You are a skill taxonomy analyst. You will receive a list of ~17,000 AI coding skill names with short descriptions.

Your task: analyze ALL skills and discover the natural categories that exist across three dimensions.

For each dimension, identify **distinct, non-overlapping categories** at an appropriate granularity level (roughly 8-15 categories per dimension). Each category should have a short lowercase label (1-2 words, snake_case) and a brief description.

Dimensions:
1. **primary_action**: What the skill DOES (the core verb/activity)
2. **primary_object**: What the skill acts ON (the target/subject)
3. **domain**: What technical field the skill belongs to

Output strict JSON with this structure:
{
  "primary_action": [{"label": "...", "description": "..."}],
  "primary_object": [{"label": "...", "description": "..."}],
  "domain": [{"label": "...", "description": "..."}]
}

No markdown fences, no explanations outside the JSON."""


def run_pass1():
    print("=" * 60)
    print("Pass 1: Discover natural categories from all skills")
    print("=" * 60)

    skills = []
    with open(DATA_PATH) as f:
        for line in f:
            skills.append(json.loads(line))
    print(f"Loaded {len(skills)} skills")

    skill_lines = [f"{s['name']}: {s['description'][:100]}" for s in skills]
    user_text = "Here are all the skills:\n\n" + "\n".join(skill_lines)
    print(f"User prompt: {len(user_text):,} chars (~{len(user_text)//4:,} tokens)")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=MODEL, max_tokens=8192,
        system=PASS1_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]

    categories = json.loads(text)

    for dim in ["primary_action", "primary_object", "domain"]:
        cats = categories[dim]
        print(f"\n  {dim} ({len(cats)} categories):")
        for c in cats:
            print(f"    - {c['label']}: {c['description']}")

    with open(CATEGORIES_PATH, "w") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {CATEGORIES_PATH}")


# ─── Pass 2: Tag each skill ──────────────────────────────────

def build_pass2_system():
    with open(CATEGORIES_PATH) as f:
        categories = json.load(f)

    lines = ["You are a skill taxonomy classifier. For each AI coding skill, assign exactly 3 labels.\n"]
    for dim in ["primary_action", "primary_object", "domain"]:
        cats = categories[dim]
        lines.append(f"**{dim}** — choose ONE from:")
        for c in cats:
            lines.append(f"  - {c['label']}: {c['description']}")
        lines.append("")
    lines.append('Respond ONLY with a JSON array. Each element: {"id": "...", "primary_action": "...", "primary_object": "...", "domain": "..."}.')
    lines.append("No explanations, no markdown fences.")
    return "\n".join(lines)


def run_pass2(resume=False):
    print("=" * 60)
    print(f"Pass 2: Tag each skill (batch={BATCH_SIZE})")
    print("=" * 60)

    system_prompt = build_pass2_system()
    skills = []
    with open(DATA_PATH) as f:
        for line in f:
            skills.append(json.loads(line))

    done_ids = set()
    if resume and os.path.exists(TAGS_PATH):
        with open(TAGS_PATH) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        skills = [s for s in skills if s["id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(skills)} remaining")

    if not skills:
        print("Nothing to do!")
        return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    mode = "a" if resume else "w"

    with open(TAGS_PATH, mode) as fout:
        for i in range(0, len(skills), BATCH_SIZE):
            batch = skills[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(skills) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}...", end="", flush=True)

            try:
                user_prompt = "Tag these skills:\n" + "\n".join(
                    f"{s['id']}|{s['name']}: {s['description'][:200]}" for s in batch
                )
                resp = client.messages.create(
                    model=MODEL, max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                results = json.loads(text)
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                print(f" OK ({len(results)})")
            except Exception as e:
                print(f" ERROR: {e}")

    print(f"Output: {TAGS_PATH}")


# ─── Clean: Deduplicate ──────────────────────────────────────

def clean():
    seen = {}
    with open(TAGS_PATH) as f:
        for line in f:
            d = json.loads(line)
            sid = d["id"].split("|")[0].strip()
            d["id"] = sid
            seen[sid] = d

    with open(CLEAN_PATH, "w") as f:
        for d in seen.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Clean: {len(seen)} unique skills → {CLEAN_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pass1", "pass2", "clean"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "pass1":
        run_pass1()
    elif args.mode == "pass2":
        run_pass2(resume=args.resume)
    else:
        clean()


if __name__ == "__main__":
    main()
