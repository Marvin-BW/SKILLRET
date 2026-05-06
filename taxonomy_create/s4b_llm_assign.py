"""
Step 4b: LLM-based skill assignment
====================================

Classifies all 17,810 skills into the finalized 6 Major / 18 Sub taxonomy
using Claude Sonnet 4.6 via the Anthropic API.

Unlike the tag-based rule assignment (s4_taxonomy.py), this script feeds
each skill's **name + description** directly to the LLM, enabling
context-aware classification that captures nuances missed by 3-axis tags.

Input:
    data/filtered_skills.jsonl          (id, name, description, ...)
    outputs/taxonomy.json               (6 Major / 18 Sub definitions)

Output:
    outputs/skill_assignments_llm.jsonl (id, major, sub)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python s4b_llm_assign.py [--batch-size 50] [--resume]
"""

import json, os, sys, time, argparse
from collections import Counter

import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_FILE = os.path.join(OUT_DIR, "skill_assignments_llm.jsonl")
MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5


def load_taxonomy():
    with open(os.path.join(OUT_DIR, "taxonomy.json")) as f:
        data = json.load(f)
    return data["taxonomy"]


def build_system_prompt(taxonomy):
    lines = [
        "You are a taxonomy classifier for AI agent skills. "
        "Each skill is a reusable instruction file that extends an LLM's capabilities. "
        "Given a skill's name and description, assign it to exactly one "
        "(Major, Sub-category) pair from the taxonomy below.",
        "",
        "TAXONOMY:",
    ]
    valid_pairs = []
    for cat in taxonomy:
        major = cat["major"]
        lines.append(f"\n## {major}")
        lines.append(f"   {cat['major_description']}")
        for sub_info in cat["subs"]:
            sub = sub_info["sub"]
            lines.append(f"   - {sub}: {sub_info['sub_description']}")
            valid_pairs.append((major, sub))

    lines.extend([
        "",
        "CLASSIFICATION PRINCIPLE:",
        "- Classify by the DOMAIN in which the skill's capability is used.",
        "- Every skill extends an agent's capabilities, but classify by WHAT "
        "the extended capability is about, not the fact that an agent uses it.",
        "- Technical documentation (README, API docs) → SE / Documentation.",
        "- Product planning (PRD, roadmaps, sprints, Jira) → Business & Planning / Project Management.",
        "- Pure business analysis (market research, competitive analysis) → Business & Planning / Business Analysis.",
        "- Text/media as the final product (blogs, novels, graphics) → Content Creation.",
        "- AI Agents is ONLY for the agent system itself "
        "(prompt design, multi-agent routing, MCP server building, agent evaluation).",
        "- Information Retrieval is ONLY when the PRIMARY output is found/retrieved content.",
        "",
        "OUTPUT: a JSON array, one object per skill.",
        '  {"id": "...", "major": "...", "sub": "..."}',
        "No markdown fences. No explanations.",
    ])
    return "\n".join(lines), valid_pairs


def load_skills():
    skills = []
    with open(os.path.join(DATA_DIR, "filtered_skills.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            skills.append({
                "id": d["id"],
                "name": d.get("name", ""),
                "description": d.get("description", "") or "",
            })
    return skills


def load_existing():
    done = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                d = json.loads(line)
                done[d["id"]] = d
    return done


def build_user_message(batch):
    lines = ["Classify these skills:\n"]
    for s in batch:
        desc = s["description"][:300].replace("\n", " ").strip()
        lines.append(f'{s["id"]}|{s["name"]}: {desc}')
    return "\n".join(lines)


def parse_response(text, batch_ids, valid_pairs):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]

    results = json.loads(text)
    valid_set = set(valid_pairs)
    parsed = {}
    for r in results:
        sid = r["id"]
        major = r["major"]
        sub = r["sub"]
        if (major, sub) not in valid_set:
            print(f"  WARNING: invalid pair ({major}, {sub}) for {sid}, skipping")
            continue
        parsed[sid] = {"id": sid, "major": major, "sub": sub}

    missing = set(batch_ids) - set(parsed.keys())
    if missing:
        print(f"  WARNING: {len(missing)} skills missing from response")
    return parsed


def classify_batch(client, system_prompt, valid_pairs, batch):
    user_msg = build_user_message(batch)
    batch_ids = [s["id"] for s in batch]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
            return parse_response(text, batch_ids, valid_pairs)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    print(f"  FAILED batch after {MAX_RETRIES} attempts, skipping")
    return {}


def main():
    parser = argparse.ArgumentParser(description="LLM-based skill classification")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without making API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("Step 4b: LLM-based Skill Assignment")
    print("=" * 60)

    taxonomy = load_taxonomy()
    system_prompt, valid_pairs = build_system_prompt(taxonomy)
    skills = load_skills()
    print(f"Total skills: {len(skills):,}")
    print(f"Taxonomy: {len(taxonomy)} Majors, {sum(len(c['subs']) for c in taxonomy)} Subs")
    print(f"Model: {MODEL}")
    print(f"Batch size: {args.batch_size}")

    done = load_existing() if args.resume else {}
    if done:
        print(f"Resuming: {len(done):,} already classified")

    remaining = [s for s in skills if s["id"] not in done]
    n_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    print(f"Remaining: {len(remaining):,} skills in {n_batches} batches")

    if args.dry_run:
        print("\n[DRY RUN] Would classify these skills. Exiting.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    if not args.resume and os.path.exists(OUTPUT_FILE):
        os.rename(OUTPUT_FILE, OUTPUT_FILE + ".bak")

    outf = open(OUTPUT_FILE, "a")
    t0 = time.time()

    try:
        for i in range(0, len(remaining), args.batch_size):
            batch = remaining[i : i + args.batch_size]
            batch_num = i // args.batch_size + 1
            elapsed = time.time() - t0
            rate = batch_num / elapsed * 3600 if elapsed > 0 else 0

            print(f"\nBatch {batch_num}/{n_batches} "
                  f"({len(done) + i:,}/{len(skills):,} done, "
                  f"{rate:.0f} batches/hr)")

            results = classify_batch(client, system_prompt, valid_pairs, batch)

            for sid, rec in results.items():
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[sid] = rec
            outf.flush()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved. Use --resume to continue.")
    finally:
        outf.close()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Classified: {len(done):,} / {len(skills):,}")
    print(f"Time: {elapsed/60:.1f} min")

    major_counter = Counter(d["major"] for d in done.values())
    print(f"\nMajor distribution:")
    for m, c in major_counter.most_common():
        print(f"  {m:<25} {c:>6} ({c/len(done)*100:.1f}%)")

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
