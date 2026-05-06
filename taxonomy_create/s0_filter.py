"""
Step 0: Quality filtering (22,795 → 17,810)

  0-1: Content dedup (skill_md normalized hash → SHA256)
  0-2: Search target dedup (name+description normalized hash → SHA256)
  0-3: Description parse error recovery/removal (<10 chars)
  0-4: Non-English skill removal (ratio > 3%)
  0-5: License filter (frontmatter-based, keep MIT/Apache only)

Input:  <RAW_DATA_DIR>/skills.jsonl + skills_metadata.jsonl (22,795)
Output: data/filtered_skills.jsonl (17,810)
"""

import json, hashlib, re, os, yaml
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("RAW_SKILL_DATA", os.path.join(BASE_DIR, "data"))  # TODO: set path to raw skills
SKILLS_PATH = os.path.join(DATA_DIR, "skills.jsonl")
META_PATH = os.path.join(DATA_DIR, "skills_metadata.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "filtered_skills.jsonl")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

print("=" * 60)
print("Step 0: Quality Filtering")
print("=" * 60)

# ── Load ──
print("\nLoading...")
skills = []
with open(SKILLS_PATH) as f:
    for line in f:
        skills.append(json.loads(line))

meta_map = {}
with open(META_PATH) as f:
    for line in f:
        m = json.loads(line)
        meta_map[m["id"]] = m

for s in skills:
    m = meta_map.get(s["id"], {})
    for key in ("stars", "installs", "license", "repo", "source_url", "raw_url", "author", "namespace"):
        s[key] = m.get(key, "" if key not in ("stars", "installs") else 0)

total_original = len(skills)
print(f"  loaded: {total_original}")
removal_log = defaultdict(list)

# ── 0-1: Content dedup ──
def normalize_md(text):
    text = re.sub(r'^---\s*\n.*?\n---\s*\n?', '', text, count=1, flags=re.DOTALL)
    text = text.lower()
    return re.sub(r'[^a-z0-9]', '', text)

print("\n[0-1] Content dedup...")
hash_groups = defaultdict(list)
for s in skills:
    h = hashlib.sha256(normalize_md(s["skill_md"]).encode()).hexdigest()
    hash_groups[h].append(s)

keep_ids = set()
for h, group in hash_groups.items():
    best = max(group, key=lambda x: (x.get("stars", 0), x.get("installs", 0)))
    keep_ids.add(best["id"])
    for s in group:
        if s["id"] != best["id"]:
            removal_log["hash_duplicate"].append(s["id"])

skills = [s for s in skills if s["id"] in keep_ids]
print(f"  remaining: {len(skills)} (-{total_original - len(skills)})")

# ── 0-2: Search target dedup ──
print("\n[0-2] Search target dedup (name+desc)...")
nd_groups = defaultdict(list)
for s in skills:
    nd_key = re.sub(r'[^a-z0-9]', '', (s.get("name", "") + "|||" + s.get("description", "")).lower())
    h = hashlib.sha256(nd_key.encode()).hexdigest()
    nd_groups[h].append(s)

prev = len(skills)
keep_ids = set()
for h, group in nd_groups.items():
    best = max(group, key=lambda x: (x.get("stars", 0), x.get("installs", 0)))
    keep_ids.add(best["id"])

skills = [s for s in skills if s["id"] in keep_ids]
print(f"  remaining: {len(skills)} (-{prev - len(skills)})")

# ── 0-3: Description recovery ──
print("\n[0-3] Description recovery...")

def extract_desc_from_yaml(skill_md):
    fm_match = re.match(r"^---\n(.*?)\n---", skill_md, re.DOTALL)
    if not fm_match:
        return None
    try:
        fm = yaml.safe_load(fm_match.group(1))
        if isinstance(fm, dict):
            desc = fm.get("description", "")
            if isinstance(desc, str) and len(desc.strip()) >= 10:
                return desc.strip()
    except yaml.YAMLError:
        pass
    return None

def extract_desc_from_body(skill_md):
    body = re.sub(r"^---.*?---\s*", "", skill_md, count=1, flags=re.DOTALL) if skill_md.startswith("---") else skill_md
    body = body.strip()
    if not body:
        return None
    first_para = re.sub(r"^#+ +", "", body.split("\n\n")[0].strip())
    return first_para if 10 <= len(first_para) <= 500 else None

remove_ids = set()
for s in skills:
    if len(s.get("description", "").strip()) >= 10:
        continue
    yaml_desc = extract_desc_from_yaml(s["skill_md"])
    if yaml_desc:
        s["description"] = yaml_desc
        continue
    body_desc = extract_desc_from_body(s["skill_md"])
    if body_desc:
        s["description"] = body_desc
        continue
    remove_ids.add(s["id"])

prev = len(skills)
skills = [s for s in skills if s["id"] not in remove_ids]
print(f"  remaining: {len(skills)} (-{prev - len(skills)})")

# ── 0-4: Non-English filter ──
print("\n[0-4] Non-English filter (>3%)...")
non_eng = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]")
remove_ids = set()
for s in skills:
    md = s["skill_md"]
    if len(md) > 0 and len(non_eng.findall(md)) / len(md) > 0.03:
        remove_ids.add(s["id"])

prev = len(skills)
skills = [s for s in skills if s["id"] not in remove_ids]
print(f"  remaining: {len(skills)} (-{prev - len(skills)})")

# ── 0-5: License filter ──
print("\n[0-5] License filter (MIT/Apache only)...")
remove_ids = set()
for s in skills:
    fm_match = re.match(r'^---\s*\n(.*?)\n---', s["skill_md"], re.DOTALL)
    if not fm_match:
        continue
    lic_match = re.search(r'license\s*:\s*(.+)', fm_match.group(1), re.IGNORECASE)
    if not lic_match:
        continue
    body_license = lic_match.group(1).strip().strip('"').strip("'").lower()
    if body_license and 'mit' not in body_license and 'apache' not in body_license:
        remove_ids.add(s["id"])

prev = len(skills)
skills = [s for s in skills if s["id"] not in remove_ids]
print(f"  remaining: {len(skills)} (-{prev - len(skills)})")

# ── Save ──
print(f"\n{'=' * 60}")
print(f"  {total_original:,} → {len(skills):,} ({len(skills)/total_original*100:.1f}%)")
with open(OUTPUT_PATH, "w") as f:
    for s in skills:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"  → {OUTPUT_PATH}")
