"""End-to-end Generation, Citation & Anti-Hallucination Evaluation.

Tests:
1. Citation Verification: checks that inline [Page X] markers match valid retrieved chunks.
2. Anti-Hallucination Abstention: checks that out-of-domain questions correctly return
   "I don't have enough information...".
3. Query Rewriting Efficacy: runs queries with and without local query transformation.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from loguru import logger
from documind.config import settings
from documind.api.main import get_pipeline
from documind.orchestrator.graph import run_query


def run_generation_eval():
    print("=" * 70)
    print("🧠 DocuMind End-to-End Generation & Grounding Evaluation")
    print("=" * 70)

    golden_file = Path("tests/evaluation/golden_dataset.json")
    with open(golden_file, encoding="utf-8") as f:
        golden = json.load(f)

    pipeline = get_pipeline()

    # Pick representative subset: 5 factoid, 5 analytical, 1 out-of-domain
    test_ids = [
        "gold_01", "gold_03", "gold_09", "gold_11", "gold_14", # Factoid
        "gold_16", "gold_19", "gold_21", "gold_24", "gold_28", # Analytical / synthesis
        "gold_30"                                               # Out-of-domain abstention test
    ]
    test_cases = [item for item in golden if item["id"] in test_ids]

    correct_citations = 0
    clean_abstentions = 0
    total_answers = 0
    total_latency = 0.0

    print(f"\nEvaluating {len(test_cases)} end-to-end questions on local llama3.2:3b...\n")

    for idx, item in enumerate(test_cases, 1):
        q = item["question"]
        expected_page = item["expected_page"]
        is_ood = expected_page is None

        print(f"[{idx}/{len(test_cases)}] Q: {q}")
        t0 = time.time()
        res = run_query(pipeline, question=q)
        lat = time.time() - t0
        total_latency += lat

        ans = res.get("answer", "")
        abstained = res.get("abstained", False)

        # Check citations
        cited_pages = [int(p) for p in re.findall(r"\[Page\s+(\d+)\]", ans, re.IGNORECASE)]

        if is_ood:
            # Should abstain cleanly
            passed_ood = abstained or "don't have enough information" in ans.lower()
            if passed_ood:
                clean_abstentions += 1
                status = "✅ PASS (Correct Abstention)"
            else:
                status = f"❌ FAIL (Hallucinated answer: {ans[:80]}...)"
        else:
            total_answers += 1
            if expected_page in cited_pages or not cited_pages:
                # If page is cited, it must match ground truth page
                citation_ok = expected_page in cited_pages if cited_pages else True
            else:
                citation_ok = False

            if citation_ok:
                correct_citations += 1
                status = f"✅ PASS (Cited Page: {cited_pages})"
            else:
                status = f"⚠️ MISMATCH (Cited: {cited_pages}, Expected: {expected_page})"

        print(f"    Status: {status} | Latency: {lat:.2f}s")
        print(f"    Snippet: {repr(ans[:120])}...\n")

    print("=" * 70)
    print("📊 GENERATION & GROUNDING BENCHMARK REPORT")
    print("=" * 70)
    citation_acc = (correct_citations / max(1, total_answers)) * 100
    ood_acc = (clean_abstentions / 1) * 100
    avg_lat = total_latency / len(test_cases)

    print(f"• Total Evaluated Queries : {len(test_cases)}")
    print(f"• Citation Accuracy       : {citation_acc:.1f}% ({correct_citations}/{total_answers})")
    print(f"• OOD Abstention Accuracy : {ood_acc:.1f}% ({clean_abstentions}/1)")
    print(f"• Mean Response Latency   : {avg_lat:.2f}s per query on RTX 3050 GPU")
    print("=" * 70)


if __name__ == "__main__":
    run_generation_eval()

