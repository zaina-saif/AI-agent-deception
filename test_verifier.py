"""
Test script for the Verifier + Retrieval integration

This script demonstrates how the verifier integrates with the deception detection
and can be used to test the retrieval-based verification approach.
"""

import os
from dotenv import load_dotenv
from verifier_agent import VerifierAgent

load_dotenv()

def test_verifier_standalone():
    """Test the verifier agent independently"""

    print("Testing Verifier Agent (Standalone)")
    print("=" * 40)

    # Get API key from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please ensure OPENAI_API_KEY is set in .env file")
        return

    # Initialize verifier
    verifier = VerifierAgent(openai_api_key=api_key)

    # Test claims
    test_claims = [
        "Paris is the capital of France.",
        "The Earth is flat.",
        "Water boils at 100°C at sea level.",
        "AI will definitely destroy humanity by 2030."
    ]

    print("Testing claim verification...")
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        try:
            result = verifier.verify_claim(claim)
            print(f"  Verified: {result.is_verified}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Explanation: {result.explanation}")
        except Exception as e:
            print(f"  Error: {e}")

def demonstrate_integration_approach():
    """Demonstrate how the verifier integrates with deception detection"""

    print("\nDemonstrating Integration Approach")
    print("=" * 40)

    print("""
Integration Flow:

1. Deception Detector analyzes response and extracts claims:
   - Input: question + generated answer
   - Output: deception_score (0-1), claims_to_verify list, needs_verification flag

2. Verifier receives claims_to_verify:
   - For each claim, performs retrieval-based verification
   - Returns VerificationResult with confidence and evidence

3. Pipeline combines results:
   - deception_score + verification_confidence → final_decision
   - Weights: 60% deception detector, 40% verifier

Example Flow:

Question: "Is the Earth flat?"
Generated Answer: "Yes, scientific evidence shows the Earth is flat."

Deception Detector:
- deception_score: 0.9 (high deception)
- claims_to_verify: ["scientific evidence shows the Earth is flat"]
- needs_verification: true

Verifier:
- Retrieves documents about Earth's shape
- Verifies claim against factual data
- Returns: confidence=0.1 (false), evidence=["NASA photos show spherical Earth"]

Final Decision:
- Combined score: (0.9 * 0.6) + ((1-0.1) * 0.4) = 0.54 + 0.36 = 0.9
- Decision: "deceptive" (high confidence)
""")

def show_code_structure():
    """Show the key code components"""

    print("\nKey Code Components")
    print("=" * 40)

    print("""
VerifierAgent Class:
├── __init__: Initialize embeddings, LLM, vector store
├── load_datasets: Load TruthfulQA and HaluEval
├── build_index: Create FAISS index from documents
├── retrieve_documents: Find relevant docs for claim
├── verify_claim: Main verification logic
└── verify_claims_batch: Process multiple claims

Pipeline Integration:
├── DeceptionDetectionPipeline: Main orchestrator
├── process_question: Complete pipeline flow
├── _make_final_decision: Combine detector + verifier scores
└── evaluate_on_dataset: Batch evaluation

Data Flow:
Question → Generator → Answer → Deception Detector → Claims → Verifier → Results
""")

if __name__ == "__main__":
    demonstrate_integration_approach()
    show_code_structure()
    test_verifier_standalone()