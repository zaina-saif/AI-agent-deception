"""One-shot live end-to-end test (reads .env; do not commit secrets)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


def main() -> int:
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print("FAIL: OPENAI_API_KEY missing in .env")
        return 1

    from pipeline import DeceptionDetectionPipeline

    print("=== Live E2E: Generator → Detector → Verifier → decision ===\n")
    pipeline = DeceptionDetectionPipeline(os.environ["OPENAI_API_KEY"])

    question = "What is the capital of France?"
    result = pipeline.process_question(question)

    ok = (
        bool(result.generated_answer.strip())
        and 0 <= result.deception_score <= 1
        and result.final_decision in ("truthful", "deceptive", "uncertain")
        and 0 <= result.confidence <= 1
    )

    print("\n--- Result summary ---")
    print(f"Decision: {result.final_decision} (confidence {result.confidence:.2f})")
    print(f"Deception score: {result.deception_score:.2f}")
    print(f"Verification items: {len(result.verification_results)}")

    if ok:
        print("\nOK: end-to-end pipeline completed successfully.")
        return 0
    print("\nFAIL: unexpected result shape or values.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
