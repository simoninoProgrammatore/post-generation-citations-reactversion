"""
Step 1: Generate LLM responses to queries (with optional RAG).

Given a query and optionally a set of passages from the ALCE dataset,
this module produces a raw LLM response that will later be augmented
with citations.
"""

import json
import argparse
from pathlib import Path

from llm_client import call_llm


SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the question clearly and factually. "
    "Do NOT include any citations or references in your response."
)

RAG_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the question ONLY using the information provided in the passages below. "
    "Do NOT use any external knowledge. "
    "If the passages do not contain enough information to answer, say so. "
    "Do NOT include any citations or references in your response."
)


def load_dataset(dataset_path: str) -> list[dict]:
    """Load ALCE dataset from JSON file."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data


def generate_response(query: str, passages: list[dict] = None,
                      model: str = "gemini-2.0-flash", max_tokens: int = 300) -> str:
    """
    Generate a response to a query using the specified LLM.

    Args:
        query:      The input question.
        passages:   Optional list of passages to ground the response.
        model:      Model identifier (e.g. 'gemini-2.0-flash').
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The generated response text (without citations).
    """
    if passages:
        passages_text = "\n\n".join([
            f"[{i+1}] {p.get('title', 'N/A')}:\n{p.get('text', '')}"
            for i, p in enumerate(passages[:10])
        ])
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"Passages:\n{passages_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query}\n\nAnswer:"

    return call_llm(prompt, model=model, max_tokens=max_tokens)


def run(dataset_path: str, output_path: str, model: str = "gemini-2.0-flash"):
    """
    Generate responses for all queries in the dataset.

    Args:
        dataset_path: Path to the ALCE dataset JSON.
        output_path:  Path to save the generated responses.
        model:        Model identifier.
    """
    data = load_dataset(dataset_path)
    results = []

    for example in data:
        query = example["question"]
        passages = example.get("docs", [])
        response = generate_response(query, passages=passages, model=model)
        results.append({
            "question": query,
            "raw_response": response,
            "passages": passages,
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} responses -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM responses")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/generations.json")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    args = parser.parse_args()
    run(args.dataset, args.output, args.model)