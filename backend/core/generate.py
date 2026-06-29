"""
Step 1: Generate LLM responses to queries (with optional RAG).

Given a query and optionally a set of passages from the ALCE+ dataset,
this module produces a raw LLM response that will later be decomposed
into atomic claims and augmented with citations.

The response is deliberately produced WITHOUT citations: citation is a
post-hoc step (P-Cite). The prompt also forbids meta-commentary
("the passage says...") because such phrasing produces degenerate atomic
claims downstream in the decomposition step.

This module is the single source of truth for the generation prompt.
Both the CLI (`run`) and the FastAPI orchestrator
(`pipeline_runners.run_generate`) call `generate_response` / `build_prompt`
so the prompt lives in exactly one place.
"""

import json
import argparse
from pathlib import Path

from core.llm_client import call_llm


# ──────────────────────────────────────────────
# Prompt building (single source of truth)
# ──────────────────────────────────────────────

# Instruction block shared by the RAG (grounded) generation path.
RAG_INSTRUCTIONS = (
    "Answer the question directly and naturally, in plain prose. "
    "Do NOT restate or echo the question; begin straight with the answer. "
    "Start with what the user needs to know, "
    "then add a few additional facts from the passages that are relevant and useful. "
    "Do not write meta-commentary like 'the passage says' or 'according to the text'. "
    "Do not add citation markers like [1] or [2]. "
    "No markdown, no bullet points."
)

# Instruction block for the closed-book (no passages) path.
NO_RAG_INSTRUCTIONS = (
    "Answer the question directly, then add a few useful related facts. "
    "No markdown, bullet points, or citations."
)


def _format_passages(passages: list[dict], max_passages: int = 10) -> str:
    """Render passages as a numbered, titled block for the prompt."""
    return "\n\n".join(
        f"[{i + 1}] {p.get('title', 'N/A')}:\n{p.get('text', '')}"
        for i, p in enumerate(passages[:max_passages])
    )


def build_prompt(query: str, passages: list[dict] | None = None) -> str:
    """
    Build the generation prompt. Single source of truth for both the CLI
    and the FastAPI orchestrator.

    If `passages` are provided, builds the RAG (grounded) prompt; otherwise
    builds the closed-book prompt.
    """
    if passages:
        passages_text = _format_passages(passages)
        return (
            "Read the passages below and answer the question.\n\n"
            f"Passages:\n{passages_text}\n\n"
            f"Question: {query}\n\n"
            f"{RAG_INSTRUCTIONS}\n\n"
            "Answer:"
        )
    return (
        f"Question: {query}\n\n"
        f"{NO_RAG_INSTRUCTIONS}\n\n"
        "Answer:"
    )


def load_dataset(dataset_path: str) -> list[dict]:
    """Load ALCE+ dataset from JSON file."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data


def generate_response(
    query: str,
    passages: list[dict] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
) -> str:
    """
    Generate a response to a query using the specified LLM.

    Args:
        query:      The input question.
        passages:   Optional list of passages to ground the response (RAG).
        model:      Model identifier routed by llm_client.
        max_tokens: Maximum number of tokens in the response. Default 1024
                    so long-form (ELI5) answers are not truncated; lower it
                    for short factoid runs if needed.

    Returns:
        The generated response text (without citations).
    """
    prompt = build_prompt(query, passages=passages)
    return call_llm(prompt, model=model, max_tokens=max_tokens)


def run(dataset_path: str, output_path: str, model: str = "claude-haiku-4-5-20251001"):
    """
    Generate responses for all queries in the dataset.

    Args:
        dataset_path: Path to the ALCE+ dataset JSON.
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
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    args = parser.parse_args()
    run(args.dataset, args.output, args.model)