"""
Step 3: Stable groups → Major/Sub taxonomy mapping

Part A: Stable groups + singletons → Major mapping
        - Groups with shared object → direct Major mapping
        - Groups with shared action → split by object, then map each
        - Singletons → individual mapping by object/action rules
Part B: Action distribution within each Major → Sub grouping

Input:  outputs/stable_groups.json, outputs/skill_tags_clean.jsonl
Output: outputs/group_to_major.json   (group/singleton → Major mapping)
        outputs/sub_grouping.json     (Major → Sub grouping)
"""

import json, os
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")


OBJECT_TO_MAJOR = {
    "code": "Software Engineering",
    "api": "Software Engineering",
    "ui_component": "Software Engineering",
    "dependency": "Software Engineering",
    "infrastructure": "Software Engineering",
    "pipeline": "Software Engineering",
    "security": "Software Engineering",
    "agent_skill": "AI Agents",
    "documentation": "Documentation & Knowledge",
    "content": "Content Creation",
    "project": "Project Management",
    "data": "Data & Analytics",
    "database": "Data & Analytics",
    "test_suite": "Testing & QA",
}

ACTION_OVERRIDE = {
    "search": "Information Retrieval",
    "test": "Testing & QA",
}


def get_major(action, obj):
    if action in ACTION_OVERRIDE:
        return ACTION_OVERRIDE[action]
    if obj in OBJECT_TO_MAJOR:
        return OBJECT_TO_MAJOR[obj]
    if action == "deploy":
        return "Software Engineering"
    return "Unknown"


def load_tags():
    tags = []
    with open(os.path.join(OUT_DIR, "skill_tags_clean.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("primary_object") == "document":
                d["primary_object"] = "documentation"
            tags.append(d)
    return tags


def part_a_group_to_major():
    """Stable groups + singletons → Major mapping"""
    print("=" * 70)
    print("Part A: Stable groups → Major mapping")
    print("=" * 70)

    with open(os.path.join(OUT_DIR, "stable_groups.json")) as f:
        stable = json.load(f)

    strict = stable["strict"]
    groups = strict["groups"]
    singletons = strict["singletons"]

    total_combos = sum(c for g in groups for _, _, c in g) + sum(c for _, _, c in singletons)
    print(f"Strict(5/5): {len(groups)} groups + {len(singletons)} singletons = {total_combos:,} skills")

    result = {"groups": [], "singletons": [], "coverage": {}}
    major_skills = Counter()

    print(f"\n{'─' * 70}")
    print("Group mapping")
    print(f"{'─' * 70}")

    for i, group in enumerate(groups):
        members = [(a, o, c) for a, o, c in group]
        group_skills = sum(c for _, _, c in members)

        objects = set(o for _, o, _ in members)
        actions = set(a for a, _, _ in members)
        majors_in_group = Counter()
        member_mapping = []

        for a, o, c in members:
            m = get_major(a, o)
            majors_in_group[m] += c
            member_mapping.append({"action": a, "object": o, "count": c, "major": m})

        unique_majors = set(majors_in_group.keys())

        if len(objects) == 1:
            binding = f"object={list(objects)[0]}"
        elif len(actions) == 1:
            binding = f"action={list(actions)[0]}"
        else:
            binding = "mixed"

        if len(unique_majors) == 1:
            mapping_type = "direct"
        else:
            mapping_type = "split"

        for m, cnt in majors_in_group.items():
            major_skills[m] += cnt

        group_info = {
            "group_id": f"G{i+1}",
            "members": member_mapping,
            "total_skills": group_skills,
            "objects": sorted(objects),
            "actions": sorted(actions),
            "binding": binding,
            "mapping_type": mapping_type,
            "mapped_majors": dict(majors_in_group),
        }
        result["groups"].append(group_info)

        members_str = ", ".join(f"{a}×{o}({c})" for a, o, c in members)
        print(f"\n  G{i+1} [{group_skills:,}] ({binding}): {members_str}")
        print(f"    → {mapping_type}: {dict(majors_in_group)}")

    print(f"\n{'─' * 70}")
    print("Singleton mapping")
    print(f"{'─' * 70}")

    for a, o, c in singletons:
        m = get_major(a, o)
        major_skills[m] += c
        reason = "action override" if a in ACTION_OVERRIDE else "object rule"
        singleton_info = {
            "action": a, "object": o, "count": c,
            "major": m, "reason": reason,
        }
        result["singletons"].append(singleton_info)
        mark = " ← action override" if a in ACTION_OVERRIDE else ""
        print(f"  {a}×{o}({c}) → {m}{mark}")

    print(f"\n{'─' * 70}")
    print("Major totals (within 80% coverage)")
    print(f"{'─' * 70}")

    for m, cnt in major_skills.most_common():
        pct = cnt / total_combos * 100
        result["coverage"][m] = {"skills": cnt, "pct": round(pct, 1)}
        print(f"  {m:<30} {cnt:>5,} ({pct:.1f}%)")

    with open(os.path.join(OUT_DIR, "group_to_major.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: group_to_major.json")

    return major_skills


def part_b_sub_grouping():
    """Action distribution within Major → Sub grouping"""
    print(f"\n{'=' * 70}")
    print("Part B: Action distribution within Major → Sub grouping")
    print("=" * 70)

    tags = load_tags()
    total = len(tags)

    major_actions = defaultdict(lambda: Counter())
    for t in tags:
        m = get_major(t["primary_action"], t["primary_object"])
        major_actions[m][t["primary_action"]] += 1

    MAJOR_ORDER = [
        "Software Engineering", "AI Agents", "Documentation & Knowledge",
        "Content Creation", "Project Management", "Data & Analytics",
        "Testing & QA", "Information Retrieval",
    ]

    from s4_taxonomy import ACTION_TO_SUB, DOMAIN_TO_SUB_IR

    result = {}

    for major in MAJOR_ORDER:
        action_counts = major_actions[major]
        m_total = sum(action_counts.values())

        print(f"\n{'─' * 70}")
        print(f"■ {major} [{m_total:,}, {m_total/total*100:.1f}%]")
        print(f"{'─' * 70}")

        if major == "Information Retrieval":
            print(f"  action 100% search → domain-based Sub")
            domain_counts = Counter()
            for t in tags:
                if get_major(t["primary_action"], t["primary_object"]) == major:
                    domain_counts[t.get("domain", "")] += 1

            sub_counts = Counter()
            for d, cnt in domain_counts.items():
                sub = DOMAIN_TO_SUB_IR.get(d, "Technical Search")
                sub_counts[sub] += cnt

            major_info = {
                "sub_basis": "domain",
                "reason": "action is 100% search — cannot distinguish by action",
                "action_distribution": dict(action_counts.most_common()),
                "subs": {},
            }
            for sub, cnt in sub_counts.most_common():
                pct = cnt / m_total * 100
                domains_in_sub = [d for d, _ in domain_counts.most_common()
                                  if DOMAIN_TO_SUB_IR.get(d, "Technical Search") == sub]
                major_info["subs"][sub] = {"count": cnt, "pct": round(pct, 1), "domains": domains_in_sub}
                print(f"  ├─ {sub} [{cnt:,}, {pct:.0f}%]: {', '.join(domains_in_sub)}")

            result[major] = major_info
            continue

        print(f"  Action distribution:")
        for action, cnt in action_counts.most_common():
            pct = cnt / m_total * 100
            print(f"    {action:<14} {cnt:>5,} ({pct:>5.1f}%)")

        sub_rules = ACTION_TO_SUB.get(major, {})
        sub_actions = defaultdict(list)
        sub_counts = Counter()

        for action, cnt in action_counts.items():
            sub = sub_rules.get(action, "Other")
            sub_actions[sub].append((action, cnt))
            sub_counts[sub] += cnt

        if major == "Software Engineering":
            sec_count = sum(1 for t in tags
                           if t["primary_object"] == "security"
                           and get_major(t["primary_action"], t["primary_object"]) == major)
            if sec_count:
                sub_counts["Security"] = sec_count
                sub_actions["Security"].append(("*:security", sec_count))

        print(f"\n  Sub grouping (action-based):")
        major_info = {
            "sub_basis": "action",
            "reason": "action distinguishes work nature within the same Major",
            "action_distribution": dict(action_counts.most_common()),
            "subs": {},
        }

        for sub, cnt in sub_counts.most_common():
            pct = cnt / m_total * 100
            actions_in = [(a, c) for a, c in sub_actions[sub]]
            actions_str = ", ".join(f"{a}({c})" for a, c in sorted(actions_in, key=lambda x: -x[1]))
            print(f"  ├─ {sub} [{cnt:,}, {pct:.0f}%]: {actions_str}")
            major_info["subs"][sub] = {
                "count": cnt, "pct": round(pct, 1),
                "actions": {a: c for a, c in actions_in},
            }

        result[major] = major_info

    with open(os.path.join(OUT_DIR, "sub_grouping.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: sub_grouping.json")

    return result


def main():
    print("=" * 70)
    print("Step 3: Stable groups → Major → Sub mapping")
    print("=" * 70)

    part_a_group_to_major()
    part_b_sub_grouping()

    print(f"\n{'━' * 70}")
    print("Done. Outputs: group_to_major.json, sub_grouping.json")


if __name__ == "__main__":
    main()
