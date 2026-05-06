"""
Step 4: Final taxonomy construction + skill assignment
=======================================================

Skill Retrieval Taxonomy — Hybrid (Special-Domain + Activity) Design
--------------------------------------------------------------------
Philosophy:
  - Primary signal: "purpose" inferred from (domain, action, object, keyword)
  - SE Sub-categories use a two-tier approach:
      Tier 1: Tag-based special domains (Security, Version Control) extracted first
      Tier 2: Remaining skills split by activity phase
  - Other Majors follow function/output-form splits
  - Designed iteratively through 3-reviewer human validation

Structure (6 Major / 18 Sub):
  1. Software Engineering
       Development / Analysis & Testing / Infrastructure & DevOps
       Documentation / Version Control / Security
  2. AI Agents
       Agent Development / Agent Orchestration / Agent Evaluation
  3. Data & ML
       Data Engineering / Data Analysis / ML Development
  4. Content Creation
       Writing & Text / Visual & Media
  5. Business & Planning
       Business Analysis / Project Management
  6. Information Retrieval
       General Search / Technical Search

Input:  outputs/skill_tags_clean.jsonl
        data/filtered_skills.jsonl   (for name/description keyword matching)
Output: outputs/taxonomy.json
        outputs/skill_assignments.jsonl
"""

import json, os, re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ══════════════════════════════════════════════════════════════
# Keyword rules
# ══════════════════════════════════════════════════════════════

SECURITY_KW = {
    "security","authentication","authorization","encryption","vulnerability",
    "compliance","threat","firewall","pentest","rbac","oauth","sso","iam",
    "mfa","csrf","xss","sql-injection","secret","certificate","tls","ssl",
    "penetration","hardening","sast","dast","zero-trust","cve","siem",
}

VERSION_CONTROL_RE = re.compile(
    r"\b(git(?!lab)|github|commit|branch(?:ing)?|merge|pull.request|pr.review|rebase|cherry.pick)\b",
    re.IGNORECASE,
)

AGENT_KW_RE = re.compile(
    r"\b(ai.agents?|autonomous.agents?|multi.agents?|agent.orchestrat\w*|agentic)\b",
    re.IGNORECASE,
)

ML_DEV_RE = re.compile(
    r"\b(fine.?tun|train|model|llm|embedding|neural|pytorch|tensorflow|"
    r"transformer|whisper|diffus|rag|retriev|vector|embed|qlora|lora|"
    r"classifi|regressor|prompt.tun|agent.skill)\b",
    re.IGNORECASE,
)

AV_RE = re.compile(
    r"\b(audio|video|ffmpeg|whisperx?|transcrib|transcription|podcast|"
    r"voiceover|subtitle|caption|mp4|mp3|wav|soundtrack|voice.?over|"
    r"film|broadcast|recording|m4b|speech)\b",
    re.IGNORECASE,
)

VISUAL_RE = re.compile(
    r"\b(image|photo|picture|graphic|logo|icon|meme|mockup|wireframe|"
    r"illustration|sketch|visual.design|ui.design|ux.design|infographic|"
    r"typograph|color.theor)\b",
    re.IGNORECASE,
)

SE_DOMAINS = {
    "backend_api","web_frontend","mobile","devops_infra","testing_qa",
    "developer_tools","security","systems",
}

CONTENT_GEN_RE = re.compile(
    r"\b(diagram|chart|mermaid|excalidraw|ascii.art|readme|changelog|"
    r"badge|svg.gen|markdown.gen|flowchart|draw)\b",
    re.IGNORECASE,
)


def _is_version_control(name, desc):
    text = f"{name} {desc[:150]}"
    return bool(VERSION_CONTROL_RE.search(text))


# ══════════════════════════════════════════════════════════════
# Major mapping
# ══════════════════════════════════════════════════════════════

def get_major(action, obj, domain, name="", desc=""):
    """Return Major for a skill.

    Priority order:
      1. Agent keyword override (desc)              -> AI Agents
      2. action=search                              -> Information Retrieval
      3. developer_tools special routing (project-mgmt / document-formatting)
      4. Domain-based routing (primary signal)
      5. business_ops action-based routing
      6. Default                                    -> SE
    """
    text = f"{name} {desc[:200]}"

    # 1. Agent keyword in name/desc overrides domain
    #    but not when the primary action is a SE activity (review/debug/test/implement)
    if AGENT_KW_RE.search(text) and action not in ("review","debug","test","implement","refactor"):
        return "AI Agents"

    # 2. Search action → IR
    if action == "search":
        return "Information Retrieval"

    # 3. developer_tools: split project-management and content-generation out of SE
    if domain == "developer_tools":
        if obj == "project" and action in ("analyze","orchestrate"):
            return "Business & Planning"
        if obj == "documentation" and action == "review":
            return "Business & Planning"
        if action == "generate" and obj in ("documentation","content") and CONTENT_GEN_RE.search(text):
            return "Content Creation"
        return "Software Engineering"

    # 4. Domain-based
    if domain == "ai_agents":
        if action in ("review","debug","test") and obj in ("code","api","test_suite"):
            return "Software Engineering"
        return "AI Agents"

    if domain == "data_ml":
        return "Data & ML"

    # 4b. database: analyze/search → Data & ML, rest → SE
    if domain == "database":
        if action in ("analyze","search"):
            return "Data & ML"
        return "Software Engineering"

    if domain in SE_DOMAINS:
        return "Software Engineering"

    if domain == "product_design":
        return "Content Creation"

    # 5. business_ops routing
    if domain == "business_ops":
        if action in ("implement","configure","debug","test","refactor","deploy"):
            return "Software Engineering"
        if action == "generate" or (action == "review" and obj == "content"):
            return "Content Creation"
        if obj == "content" and action == "implement":
            return "Content Creation"
        return "Business & Planning"

    return "Software Engineering"


# ══════════════════════════════════════════════════════════════
# Sub mapping
# ══════════════════════════════════════════════════════════════

def get_sub(major, action, obj, domain, name="", desc=""):
    text = f"{name} {desc[:200]}"

    if major == "Software Engineering":
        # Tier 1: Tag-based special domains (highest priority)
        if domain == "security" or obj == "security":
            return "Security"
        if _is_version_control(name, desc) and domain != "devops_infra":
            return "Version Control"
        if obj == "documentation" and action in ("document","generate","review","analyze","configure","search"):
            return "Documentation"

        # Tier 2: Activity-based (for remaining skills)
        if domain == "devops_infra":
            return "Infrastructure & DevOps"
        if action in ("configure","deploy","orchestrate"):
            if obj in ("infrastructure", "test_suite"):
                return "Infrastructure & DevOps"
            return "Development"
        if action in ("debug","review","analyze","test") or domain == "testing_qa":
            return "Analysis & Testing"
        return "Development"

    if major == "AI Agents":
        if action in ("implement","generate","design","configure"):
            return "Agent Development"
        if action in ("orchestrate","deploy","document"):
            return "Agent Orchestration"
        return "Agent Evaluation"

    if major == "Data & ML":
        if action == "analyze":
            return "Data Analysis"
        if ML_DEV_RE.search(text):
            return "ML Development"
        return "Data Engineering"

    if major == "Content Creation":
        if AV_RE.search(text):
            return "Visual & Media"
        if VISUAL_RE.search(text) or obj == "ui_component" or domain == "product_design":
            return "Visual & Media"
        return "Writing & Text"

    if major == "Business & Planning":
        if action in ("orchestrate","configure","review","document"):
            return "Project Management"
        return "Business Analysis"

    if major == "Information Retrieval":
        tech_domains = {"developer_tools","backend_api","data_ml","devops_infra",
                        "security","database","systems","testing_qa"}
        if domain in tech_domains:
            return "Technical Search"
        return "General Search"

    return "Other"


# ══════════════════════════════════════════════════════════════
# Metadata
# ══════════════════════════════════════════════════════════════

MAJOR_ORDER = [
    "Software Engineering", "AI Agents", "Data & ML",
    "Content Creation", "Business & Planning", "Information Retrieval",
]

MAJOR_DESC = {
    "Software Engineering":  "Software development, infrastructure, testing, version control, security, and technical documentation.",
    "AI Agents":             "Building, orchestrating, and evaluating the AI agent system itself. Does NOT include domain-specific tools agents use.",
    "Data & ML":             "Data engineering, data analysis, and machine learning model development including training and deployment.",
    "Content Creation":      "Creating content where the text or media IS the final deliverable — creative writing, marketing copy, visual design, audio/video.",
    "Business & Planning":   "Business strategy, market analysis, project management, and business communication. Non-technical business activities.",
    "Information Retrieval": "Skills whose PRIMARY purpose is searching/retrieving existing information. Output is found content, not analysis.",
}

SUB_DESC = {
    ("Software Engineering", "Development"):             "Implementing, generating, designing, and refactoring application code, APIs, UI components, and libraries.",
    ("Software Engineering", "Analysis & Testing"):      "Debugging, reviewing, analyzing, and testing code and systems.",
    ("Software Engineering", "Infrastructure & DevOps"): "Configuring, deploying, and orchestrating infrastructure, CI/CD pipelines, and cloud platforms.",
    ("Software Engineering", "Security"):                "Authentication, authorization, encryption, vulnerability analysis, and security compliance.",
    ("Software Engineering", "Version Control"):         "Git, GitHub, PR reviews, branching, merging, and source control workflows.",
    ("Software Engineering", "Documentation"):           "Technical documentation for software projects — README, API docs, code comments, changelogs, and developer guides.",

    ("AI Agents", "Agent Development"):    "Implementing, designing, and configuring AI agent skills, prompts, and system configurations.",
    ("AI Agents", "Agent Orchestration"):  "Orchestrating multi-agent workflows, routing between agents, and managing agent pipelines.",
    ("AI Agents", "Agent Evaluation"):     "Analyzing, reviewing, testing, and debugging AI agent performance and quality.",

    ("Data & ML", "Data Engineering"): "Building data pipelines, schemas, and data transformation systems.",
    ("Data & ML", "Data Analysis"):    "Exploring, analyzing, and deriving insights from datasets.",
    ("Data & ML", "ML Development"):   "Training, fine-tuning, and deploying machine learning models (LLMs, embeddings, classifiers).",

    ("Content Creation", "Writing & Text"): "Writing, editing, and creating textual content where the text is the product — blogs, novels, marketing copy, scripts.",
    ("Content Creation", "Visual & Media"): "Visual design, graphics, images, audio, and video content creation.",

    ("Business & Planning", "Business Analysis"):    "Business strategy, market research, competitive analysis, methodology frameworks, compliance auditing, and non-product business decision-making.",
    ("Business & Planning", "Project Management"):   "Product development process management — PRD, sprint/kanban/scrum workflows, Jira/task tracking, roadmaps, stakeholder communication, and meeting notes.",

    ("Information Retrieval", "Technical Search"): "Searching code, APIs, data, infrastructure, and technical documentation.",
    ("Information Retrieval", "General Search"):   "Searching business information, AI agents, content, and general knowledge.",
}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 4: Final Taxonomy Construction")
    print("=" * 60)

    # Load tags
    tags = []
    with open(os.path.join(OUT_DIR, "skill_tags_clean.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("primary_object") == "document":
                d["primary_object"] = "documentation"
            tags.append(d)

    # Load skill metadata (name/description) for keyword rules
    meta = {}
    fp = os.path.join(DATA_DIR, "filtered_skills.jsonl")
    if os.path.exists(fp):
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                meta[d["id"]] = d

    total = len(tags)
    print(f"Skills: {total:,}")

    major_counter = Counter()
    sub_counter = Counter()
    assignments = []

    for t in tags:
        a = t["primary_action"]
        o = t.get("primary_object","") or ""
        d = t.get("domain","") or ""
        s = meta.get(t["id"], {})
        nm = s.get("name","") or ""
        ds = s.get("description","") or ""
        major = get_major(a, o, d, nm, ds)
        sub   = get_sub(major, a, o, d, nm, ds)
        major_counter[major] += 1
        sub_counter[(major, sub)] += 1
        assignments.append({"id": t["id"], "major": major, "sub": sub})

    # taxonomy.json
    taxonomy = {"taxonomy": [], "total_skills": total}
    for major in MAJOR_ORDER:
        m_cnt = major_counter[major]
        subs = [
            {"sub": s, "sub_description": SUB_DESC.get((major, s), ""),
             "skill_count": cnt, "pct": round(cnt / total * 100, 2)}
            for (m, s), cnt in sub_counter.most_common() if m == major
        ]
        taxonomy["taxonomy"].append({
            "major": major, "major_description": MAJOR_DESC.get(major, ""),
            "skill_count": m_cnt, "pct": round(m_cnt / total * 100, 2),
            "subs": subs,
        })

    with open(os.path.join(OUT_DIR, "taxonomy.json"), "w") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "skill_assignments.jsonl"), "w") as f:
        for a in assignments:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'━' * 60}")
    assigned = total - major_counter.get("Unknown", 0)
    print(f"Coverage: {assigned:,} / {total:,} ({assigned/total*100:.1f}%)")
    print(f"{'━' * 60}")

    for major in MAJOR_ORDER:
        m_cnt = major_counter[major]
        print(f"\n■ {major} [{m_cnt:,}, {m_cnt/total*100:.1f}%]")
        for (m, s), cnt in sub_counter.most_common():
            if m == major:
                print(f"  ├─ {s} [{cnt:,}, {cnt/m_cnt*100:.0f}%]")

    print(f"\nSaved: taxonomy.json, skill_assignments.jsonl")
    print(f"Majors: {len(MAJOR_ORDER)}, Subs: {sum(len(t['subs']) for t in taxonomy['taxonomy'])}")


if __name__ == "__main__":
    main()
