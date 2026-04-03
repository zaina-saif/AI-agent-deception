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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

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
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5
    ):
        """
        Initialize the verifier agent.

        Args:
            openai_api_key: OpenAI API key
            embedding_model: Embedding model to use
            llm_model: LLM model for verification
            top_k: Number of documents to retrieve
        """
        self.openai_api_key = openai_api_key
        self.top_k = top_k

        embedding_model = os.getenv("VERIFIER_EMBEDDING_MODEL", embedding_model)
        llm_model = os.getenv("VERIFIER_LLM_MODEL", llm_model)

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
        )

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

                # Add false answers for contrast
                for answer in item["incorrect_answers"]:
                    doc = Document(
                        page_content=f"Question: {item['question']}\nAnswer: {answer}",
                        metadata={"source": "truthfulqa", "type": "deceptive"}
                    )
                    documents.append(doc)

        except Exception as e:
            print(f"Warning: Could not load TruthfulQA: {e}")

        try:
            # Load HaluEval dataset (assuming it's available)
            # Note: HaluEval might need to be downloaded separately
            halu_eval_path = "path/to/halueval/data.json"  # Adjust path as needed
            if os.path.exists(halu_eval_path):
                with open(halu_eval_path, 'r') as f:
                    halu_data = json.load(f)

                for item in halu_data:
                    # Add faithful answers
                    doc = Document(
                        page_content=f"Question: {item['question']}\nAnswer: {item['faithful_answer']}",
                        metadata={"source": "halueval", "type": "faithful"}
                    )
                    documents.append(doc)

                    # Add hallucinated answers
                    doc = Document(
                        page_content=f"Question: {item['question']}\nAnswer: {item['hallucinated_answer']}",
                        metadata={"source": "halueval", "type": "hallucinated"}
                    )
                    documents.append(doc)

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
            raise ValueError("No documents provided for indexing")

        print(f"Building index with {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print("Index built successfully")

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
                "(No FAISS index is available. Assess the claim using general knowledge only; "
                "state lower confidence if unsure.)"
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

    def verify_claims_batch(self, claims: List[str]) -> List[VerificationResult]:
        """
        Verify multiple claims in batch.

        Args:
            claims: List of claims to verify

        Returns:
            List of VerificationResult objects
        """
        results = []
        for claim in claims:
            result = self.verify_claim(claim)
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