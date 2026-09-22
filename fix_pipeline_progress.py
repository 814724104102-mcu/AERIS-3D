import sys

content = open("core/pipeline.py").read()

content = content.replace(
    "# ── Phase 3: Structure Analysis ───────────────────────────",
    "if progress_callback:\n        progress_callback(\"Structure analysis\", 30)\n\n    # ── Phase 3: Structure Analysis ───────────────────────────"
)
content = content.replace(
    "# ── Phase 5: Candidate Generation ─────────────────────────",
    "if progress_callback:\n        progress_callback(\"Candidate generation\", 45)\n\n    # ── Phase 5: Candidate Generation ─────────────────────────"
)
content = content.replace(
    "# ── Phase 6+7: CHF + ISCL ─────────────────────────────────",
    "if progress_callback:\n        progress_callback(\"Counterfactual testing\", 55)\n\n    # ── Phase 6+7: CHF + ISCL ─────────────────────────────────"
)
content = content.replace(
    "# ── Phase 8: Shadow Test ───────────────────────────────────",
    "if progress_callback:\n        progress_callback(\"Verification\", 65)\n\n    # ── Phase 8: Shadow Test ───────────────────────────────────"
)
content = content.replace(
    "# ── Phase 11: EGSS ─────────────────────────────────────────",
    "if progress_callback:\n        progress_callback(\"Self-disproof\", 75)\n\n    # ── Phase 11: EGSS ─────────────────────────────────────────"
)

with open("core/pipeline.py", "w") as f:
    f.write(content)

print("done")
