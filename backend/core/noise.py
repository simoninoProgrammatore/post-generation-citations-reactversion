"""
Noise injection for retrieval simulation.

Adds a random number of unrelated passages (1 to 50% of the original pool)
to simulate realistic retriever noise. Noise passages are drawn from
other questions in the same dataset and marked with `is_noise: True`.

Usage:
    from core.noise import inject_noise

    noisy_docs = inject_noise(
        docs=example["docs"],
        noise_pool=all_other_docs,   # docs from OTHER questions
        seed=42,                      # reproducibility
    )
"""

import random
import math
from copy import deepcopy


def build_noise_pool(dataset: list[dict], exclude_idx: int) -> list[dict]:
    """
    Build a pool of candidate noise passages from all examples
    in the dataset EXCEPT the one at `exclude_idx`.

    Args:
        dataset:     Full list of ALCE examples.
        exclude_idx: Index of the current example (excluded from pool).

    Returns:
        Flat list of passage dicts from all other examples.
    """
    pool = []
    for i, example in enumerate(dataset):
        if i == exclude_idx:
            continue
        for doc in example.get("docs", []):
            pool.append(doc)
    return pool


def inject_noise(
    docs: list[dict],
    noise_pool: list[dict],
    min_ratio: float = 0.0,
    max_ratio: float = 0.5,
    seed: int | None = None,
) -> list[dict]:
    """
    Inject random unrelated passages into a document pool.

    The number of noise passages is random, between:
      - max(1, ceil(len(docs) * min_ratio))
      - max(1, ceil(len(docs) * max_ratio))

    Noise passages are shuffled into the pool so the pipeline
    cannot rely on position. Each noise passage is marked with
    `is_noise: True`; original passages get `is_noise: False`.

    Args:
        docs:       Original passage pool for a question.
        noise_pool: Candidate noise passages (from other questions).
        min_ratio:  Minimum noise ratio (default 0 → at least 1 noise doc).
        max_ratio:  Maximum noise ratio (default 0.5 → up to 50%).
        seed:       Random seed for reproducibility.

    Returns:
        New list of passages (originals + noise), shuffled.
        Each passage has `is_noise` flag added.
    """
    if not docs or not noise_pool:
        return deepcopy(docs)

    rng = random.Random(seed)

    n_original = len(docs)
    n_min = max(1, math.ceil(n_original * min_ratio))
    n_max = max(1, math.ceil(n_original * max_ratio))

    # Ensure we don't try to sample more than available
    n_max = min(n_max, len(noise_pool))
    n_min = min(n_min, n_max)

    n_noise = rng.randint(n_min, n_max)

    # Sample noise passages (without replacement)
    noise_docs = rng.sample(noise_pool, n_noise)

    # Build result: mark originals and noise
    result = []
    for doc in docs:
        d = deepcopy(doc)
        d["is_noise"] = False
        result.append(d)

    for doc in noise_docs:
        d = deepcopy(doc)
        d["is_noise"] = True
        # Remove gold annotations if present (noise is never gold)
        d.pop("is_gold", None)
        d.pop("support_level", None)
        d.pop("evidence_sentence", None)
        d.pop("supports_sub_questions", None)
        d.pop("annotation_note", None)
        d.pop("answers_found", None)
        result.append(d)

    # Shuffle so noise is not always at the end
    rng.shuffle(result)

    return result


def inject_noise_dataset(
    dataset: list[dict],
    min_ratio: float = 0.0,
    max_ratio: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """
    Apply noise injection to every example in a dataset.

    Each example gets a different random number of noise passages,
    drawn from the other examples in the same dataset. Seeds are
    deterministic per-example for reproducibility.

    Args:
        dataset:   List of ALCE examples.
        min_ratio: Minimum noise ratio per example.
        max_ratio: Maximum noise ratio per example.
        seed:      Base seed (each example uses seed + idx).

    Returns:
        New dataset with noise-injected doc pools.
        Each example also gets a `noise_stats` field with metadata.
    """
    result = []

    for idx, example in enumerate(dataset):
        noise_pool = build_noise_pool(dataset, exclude_idx=idx)
        original_docs = example.get("docs", [])

        noisy_docs = inject_noise(
            docs=original_docs,
            noise_pool=noise_pool,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            seed=seed + idx,
        )

        new_example = deepcopy(example)
        new_example["docs"] = noisy_docs
        new_example["noise_stats"] = {
            "original_count": len(original_docs),
            "noise_count": sum(1 for d in noisy_docs if d.get("is_noise")),
            "total_count": len(noisy_docs),
            "noise_ratio": sum(1 for d in noisy_docs if d.get("is_noise")) / len(noisy_docs) if noisy_docs else 0,
        }

        result.append(new_example)

    return result


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Inject noise into ALCE dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON")
    parser.add_argument("--output", type=str, required=True, help="Output dataset JSON")
    parser.add_argument("--min-ratio", type=float, default=0.0)
    parser.add_argument("--max-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    noisy = inject_noise_dataset(data, args.min_ratio, args.max_ratio, args.seed)

    with open(args.output, "w") as f:
        json.dump(noisy, f, indent=2, ensure_ascii=False)

    for ex in noisy:
        stats = ex["noise_stats"]
        print(
            f"  {ex['question'][:50]:50s} | "
            f"{stats['original_count']} orig + {stats['noise_count']} noise = {stats['total_count']} total "
            f"({stats['noise_ratio']:.0%})"
        )
