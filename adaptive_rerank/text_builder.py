"""Build the document text fed to the reranker.

The original ``skillret.eval._rerank_skill_text`` always concatenates
``name | description | skill_md``. For our ablation we need a per-candidate
switch: include the body or only the metadata.
"""

from __future__ import annotations


def build_skill_text(skill: dict, use_body: bool) -> str:
    """Return the doc-text for one candidate skill.

    Args:
        skill: dict with keys ``name``, ``description``, ``skill_md``.
        use_body: if True include the full skill body, else only metadata.

    Returns:
        ``"<name> | <description> | <skill_md>"`` if ``use_body`` else
        ``"<name> | <description>"``.
    """
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    if use_body:
        body = (skill.get("skill_md") or "").strip()
        return f"{name} | {desc} | {body}"
    return f"{name} | {desc}"


def approx_token_count(text: str) -> int:
    """Cheap token-count proxy used for cost accounting (whitespace split).

    For accurate accounting use tiktoken; this is fine for ratio comparisons.
    """
    if not text:
        return 0
    return len(text.split())
