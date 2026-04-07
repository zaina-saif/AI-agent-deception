"""
Verifier + Retrieval Module for AI Deception Detection Project

This module implements the verifier agent that uses retrieval-based verification
to check claims extracted by the deception detection agent.

Approach:
1. Load and index factual data from TruthfulQA and HaluEval datasets
2. For each claim to verify, perform vector similarity search using FAISS
3. Use LLM to verify if the claim is supported by retrieved documents
4. Return verification confidence scores and explanations

Author: Zaina Saif
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# LangChain and FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Sentence Transformers for local embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    """Custom embedding class using sentence-transformers for local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


# Dataset imports (assuming datasets are downloaded)
import pandas as pd
from datasets import load_dataset

@dataclass
class VerificationResult:
    """Result of claim verification"""
    claim: str
    confidence: float  # 0.0 = definitely false, 1.0 = definitely true
    explanation: str
    supporting_evidence: List[str]
    is_verified: bool

class VerifierAgent:
    """
    Verifier agent that uses retrieval-augmented verification
    to check claims for truthfulness.
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "all-MiniLM-L6-v2",  # Changed to sentence-transformers model
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5
    ):
        """
        Initialize the verifier agent.

        Args:
            openai_api_key: OpenAI API key (for LLM only)
            embedding_model: Sentence-transformers model name
            llm_model: LLM model for verification
            top_k: Number of documents to retrieve
        """
        self.openai_api_key = openai_api_key
        self.top_k = top_k

        # Use sentence-transformers for local embeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=openai_api_key,
            temperature=0.1,
        )

        # Initialize vector store
        self.vector_store = None

        # Verification prompt
        self.verification_prompt = PromptTemplate(
            template="""
You are a fact-checking expert. Your task is to verify if a given claim is true or false
based on the provided supporting documents.

Claim to verify: {claim}

Supporting documents:
{documents}

Please analyze the claim and determine:
1. Is the claim true, false, or uncertain based on the documents?
2. Provide a confidence score from 0.0 to 1.0 (0.0 = definitely false, 1.0 = definitely true)
3. Give a brief explanation of your reasoning
4. List the specific evidence from documents that supports or contradicts the claim

Respond in JSON format with keys: "is_true", "confidence", "explanation", "evidence"
""",
            input_variables=["claim", "documents"]
        )

        # Chain for verification
        self.verification_chain = self.verification_prompt | self.llm | JsonOutputParser()

    def load_datasets(self) -> List[Document]:
        """
        Load and prepare documents from TruthfulQA and HaluEval datasets.

        Returns:
            List of Document objects for indexing
        """
        documents = []

        try:
            # Load TruthfulQA dataset
            truthfulqa = load_dataset("truthful_qa", "generation")
            for item in truthfulqa["validation"]:
                # Add truthful answers
                for answer in item["correct_answers"]:
                    doc = Document(
                        page_content=f"Question: {item['question']}\nAnswer: {answer}",
                        metadata={"source": "truthfulqa", "type": "truthful"}
                    )
                    documents.append(doc)

                # Add false answers for contrast (limit to 2 per question for speed)
                for answer in item["incorrect_answers"][:2]:
                    doc = Document(
                        page_content=f"Question: {item['question']}\nAnswer: {answer}",
                        metadata={"source": "truthfulqa", "type": "deceptive"}
                    )
                    documents.append(doc)

                # Limit to first 50 questions for faster testing
                if len(documents) >= 500:
                    break

        except Exception as e:
            print(f"Warning: Could not load TruthfulQA: {e}")

        try:
            # Load HaluEval dataset from HuggingFace
            from datasets import load_dataset as _load_dataset
            halu = _load_dataset("pminervini/HaluEval", "qa")
            # Dataset may have a "data" split or default train split
            halu_split = halu.get("data") or halu[list(halu.keys())[0]]
            for item in halu_split:
                doc = Document(
                    page_content=f"Question: {item['question']}\nAnswer: {item['right_answer']}",
                    metadata={"source": "halueval", "type": "faithful"}
                )
                documents.append(doc)
                doc = Document(
                    page_content=f"Question: {item['question']}\nAnswer: {item['hallucinated_answer']}",
                    metadata={"source": "halueval", "type": "hallucinated"}
                )
                documents.append(doc)

                # Limit to first 50 examples for faster testing
                if len(documents) >= 1000:
                    break

        except Exception as e:
            print(f"Warning: Could not load HaluEval: {e}")

        return documents

    def build_index(self, documents: List[Document]):
        """
        Build FAISS index from documents.

        Args:
            documents: List of Document objects
        """
        if not documents:
            print("Warning: No documents provided for indexing")
            return

        try:
            print(f"Building index with {len(documents)} documents...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Index built successfully")
        except Exception as e:
            print(f"Warning: Could not build FAISS index: {e}")
            print("Verifier will operate without retrieval (general knowledge only)")
            self.vector_store = None

    def retrieve_documents(self, claim: str) -> List[Document]:
        """
        Retrieve relevant documents for a claim.

        Args:
            claim: The claim to verify

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_index first.")

        return self.vector_store.similarity_search(claim, k=self.top_k)

    def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify a single claim using retrieval and LLM.

        Args:
            claim: The claim to verify

        Returns:
            VerificationResult object
        """
        # Retrieve relevant documents (fallback if index failed to build)
        if self.vector_store is None:
            docs = []
            documents_text = (
                "(No FAISS index is available due to embedding model access restrictions. "
                "Please assess the claim using general knowledge only. "
                "Be conservative with confidence scores since no specific documents are available for verification.)"
            )
        else:
            docs = self.retrieve_documents(claim)
            documents_text = "\n\n".join(
                f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)
            )

        # Run verification
        try:
            result = self.verification_chain.invoke({
                "claim": claim,
                "documents": documents_text
            })

            confidence = float(result.get("confidence", 0.5))
            is_verified = result.get("is_true", confidence > 0.5)
            explanation = result.get("explanation", "")
            evidence = result.get("evidence", [])

        except Exception as e:
            print(f"Error during verification: {e}")
            confidence = 0.5
            is_verified = False
            explanation = "Verification failed due to error"
            evidence = []

        return VerificationResult(
            claim=claim,
            confidence=confidence,
            explanation=explanation,
            supporting_evidence=evidence,
            is_verified=is_verified
        )

    def verify_claim_with_llm(self, claim: str) -> VerificationResult:
        """
        Verify a claim using pure LLM reasoning (no FAISS retrieval).
        
        This method asks GPT to evaluate if a claim is likely true or false
        based on general knowledge and reasoning.

        Args:
            claim: The claim to verify

        Returns:
            VerificationResult object
        """
        llm_prompt = f"""You are a fact-checking expert. Evaluate the following claim based on your knowledge:

Claim: "{claim}"

Determine:
1. Is this claim likely true, false, or uncertain?
2. Provide a confidence score from 0.0 to 1.0
   - 0.0 = definitely false
   - 0.5 = uncertain/cannot determine
   - 1.0 = definitely true
3. Briefly explain your reasoning (1-2 sentences)

Respond in JSON format with keys: "is_true", "confidence", "reasoning"
Example:
{{"is_true": true, "confidence": 0.95, "reasoning": "This is a well-established fact supported by scientific consensus."}}
"""

        try:
            response = self.llm.invoke([{"role": "user", "content": llm_prompt}])
            
            # Parse response
            import json
            import re
            raw_text = response.content
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            result = json.loads(json_match.group())
            
            confidence = float(result.get("confidence", 0.5))
            is_true = result.get("is_true", confidence > 0.5)
            reasoning = result.get("reasoning", "")
            
        except Exception as e:
            print(f"Error during LLM verification: {e}")
            confidence = 0.5
            is_true = False
            reasoning = "LLM verification failed"
        
        return VerificationResult(
            claim=claim,
            confidence=confidence,
            explanation=reasoning,
            supporting_evidence=[],
            is_verified=is_true
        )

    def verify_claims_batch(self, claims: List[str], method: str = "hybrid") -> List[VerificationResult]:
        """
        Verify multiple claims in batch using specified method.

        Args:
            claims: List of claims to verify
            method: "faiss" (retrieval-based), "llm" (reasoning-based), or "hybrid" (try both)

        Returns:
            List of VerificationResult objects
        """
        results = []
        for claim in claims:
            if method == "llm":
                result = self.verify_claim_with_llm(claim)
            elif method == "faiss" and self.vector_store:
                result = self.verify_claim(claim)
            else:  # hybrid or fallback
                # Try FAISS first if available, fall back to LLM
                if self.vector_store:
                    try:
                        result = self.verify_claim(claim)
                    except Exception:
                        result = self.verify_claim_with_llm(claim)
                else:
                    result = self.verify_claim_with_llm(claim)
            
            results.append(result)
        return results

    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_index(self, path: str):
        """Load the FAISS index from disk."""
        self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

def main():
    """Example usage and testing"""
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize verifier
    verifier = VerifierAgent(openai_api_key=api_key)

    # Load and index datasets
    documents = verifier.load_datasets()
    verifier.build_index(documents)

    # Example claims to verify
    test_claims = [
        "The Earth is flat.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The capital of France is Paris.",
        "AI will definitely take over the world by 2030."
    ]

    # Verify claims
    results = verifier.verify_claims_batch(test_claims)

    # Print results
    for result in results:
        print(f"\nClaim: {result.claim}")
        print(f"Verified: {result.is_verified}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation}")
        print(f"Evidence: {result.supporting_evidence}")

if __name__ == "__main__":
    main()