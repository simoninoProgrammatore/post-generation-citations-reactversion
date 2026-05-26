"""
Step 2: Decompose LLM responses into atomic claims.

Inspired by FActScore (Min et al., 2023), this module breaks a 
generated response into atomic facts — short statements that each 
contain one piece of information.
"""

import json
import argparse
from pathlib import Path

from core.llm_client import call_llm_json


DECOMPOSE_PROMPT = """\
Break the following text into atomic facts: short, self-contained statements \
that each carry exactly one piece of information.

RULES
1. ONE fact = ONE piece of information, answerable by a single closed question \
(yes/no or fill-in-the-blank). Ask yourself: "Can this be verified with one \
question?" If you need two questions, split it.
2. Use a REAL subject, never a pronoun. Never start with "It", "This", "They", \
"He", "She".
3. NEVER write meta-commentary. Do not start a fact with "The text", "The \
passage", "Historians propose that", "It is said that", or similar framing. \
State the fact directly.
   - If the source attributes a claim ("Historians proposed X, Y, Z"), drop \
the attribution shell and state each fact on its own: "X.", "Y.", "Z."
4. Do NOT over-split. Keep a single fact whole when splitting would create \
fragments that lose meaning:
   - Tightly bound coordinations stay together: \
"Gengar hides in shadows to steal the warmth of its victims." is ONE fact.
   - A small closed list stays together when items are equivalent members: \
"Bulbasaur is one of the three starter Pokémon in Pokémon Red and Blue." \
is ONE fact.
5. Keep specific names, dates, and numbers. Use the EXACT words of the source \
where possible.
6. Drop pure connective/filler sentences that carry no standalone fact.

Return ONLY a JSON array of strings, with no preamble or explanation.

GOOD example 1:
Input:
"Gengar is a Ghost and Poison type Pokémon introduced in Generation I. \
It evolves from Haunter when traded, and is known for hiding in shadows \
to steal the warmth of its victims. Gengar can Mega Evolve into Mega Gengar \
using the Gengarite."
Output:
["Gengar is a Ghost type Pokémon.",
 "Gengar is a Poison type Pokémon.",
 "Gengar was introduced in Generation I.",
 "Gengar evolves from Haunter when traded.",
 "Gengar hides in shadows to steal the warmth of its victims.",
 "Gengar can Mega Evolve into Mega Gengar.",
 "Mega Gengar requires the Gengarite to evolve."]

GOOD example 2:
Input:
"Bulbasaur is a Grass and Poison type Pokémon introduced in Generation I \
in 1996. It evolves into Ivysaur starting at level 16, and is one of the \
three starter Pokémon available at the beginning of Pokémon Red and Blue."
Output:
["Bulbasaur is a Grass type Pokémon.",
 "Bulbasaur is a Poison type Pokémon.",
 "Bulbasaur was introduced in Generation I.",
 "Bulbasaur was introduced in 1996.",
 "Bulbasaur evolves into Ivysaur starting at level 16.",
 "Bulbasaur is one of the three starter Pokémon in Pokémon Red and Blue."]

BAD examples (NEVER produce these):
- "It hides in shadows to steal warmth." (pronoun subject)
- "The text says Gengar is a Ghost type." (meta-commentary)
- "Gengar hides in shadows." + "Gengar steals warmth." \
(over-split of a tightly bound action)
- "Gengar is Ghost type." + "Gengar is dark." + "Gengar is scary." \
(hallucinated facts not in the source)
- "Gengar evolves." (too vague)

Text to decompose:
{text}
"""


REFINE_PROMPT = """\
You are given a list of atomic facts. Your task is to verify that each fact \
is truly atomic, and split any that are not.

A fact is atomic if it is answerable by exactly ONE closed question:
- Yes/No question: "Is X true?" → the fact confirms or denies it.
- Fill-in-the-blank question: "What/Who/When/Where is X?" → the fact fills \
exactly one blank.

If a fact requires TWO OR MORE questions to fully verify it, split it into \
separate facts — one per question.

PROCEDURE
For each fact, follow these steps:
1. Ask: what is the minimal closed question this fact answers?
2. If ONE question covers the entire fact → keep it as-is.
3. If TWO OR MORE questions are needed → split into one fact per question.
4. Never introduce information not present in the original fact.
5. DO NOT OVERSPLIT! 
6. Use a REAL subject in every fact. Never start with "It", "This", "They", \
"He", "She".

Return ONLY a JSON array of strings (the refined list), \
with no preamble or explanation.

EXAMPLES

Input facts:
["Gengar was introduced in Generation I in 1996.",
 "Gengar hides in shadows to steal the warmth of its victims.",
 "Marie Curie won the Nobel Prize in Physics in 1903 together with \
Pierre Curie."]

Reasoning (internal, do not output):
- "Gengar was introduced in Generation I in 1996."
  Q1: "In which Generation was Gengar introduced?" → Generation I
  Q2: "In what year was Gengar introduced?" → 1996
  TWO questions → split.
- "Gengar hides in shadows to steal the warmth of its victims."
  Q: "What does Gengar do in shadows?" → steals the warmth of its victims
  ONE question (tightly bound action) → keep.
- "Marie Curie won the Nobel Prize in Physics in 1903 together with \
Pierre Curie."
  Q1: "What prize did Marie Curie win?" → Nobel Prize in Physics
  Q2: "When did she win it?" → 1903
  Q3: "With whom did she win it?" → Pierre Curie
  THREE questions → split into three facts.

Output:
["Gengar was introduced in Generation I.",
 "Gengar was introduced in 1996.",
 "Gengar hides in shadows to steal the warmth of its victims.",
 "Marie Curie won the Nobel Prize in Physics.",
 "Marie Curie won the Nobel Prize in Physics in 1903.",
 "Marie Curie won the Nobel Prize in Physics together with Pierre Curie."]

Input facts:
{claims}
"""


def decompose_with_llm(text: str, model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """
    Decompose text into atomic claims using an LLM.

    Uses a prompt inspired by FActScore to break sentences
    into independent atomic facts.

    Args:
        text:  The response text to decompose.
        model: LLM to use for decomposition.

    Returns:
        A list of atomic claim strings.
    """
    prompt = DECOMPOSE_PROMPT.format(text=text)
    claims = call_llm_json(prompt, model=model, max_tokens=1024)

    if not isinstance(claims, list):
        raise ValueError(f"Expected a JSON list of claims, got: {type(claims)}")

    return [str(c).strip() for c in claims if str(c).strip()]


def refine_with_llm(claims: list[str], model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """
    Refine a list of atomic claims by splitting any that are not truly atomic.

    For each claim, the model identifies the minimal closed question it answers
    (yes/no or fill-in-the-blank). If a claim requires more than one question,
    it is split into separate facts — one per question.

    Args:
        claims: List of candidate atomic claims from the decompose step.
        model:  LLM to use for refinement.

    Returns:
        A refined list of atomic claim strings.
    """
    if not claims:
        return []

    prompt = REFINE_PROMPT.format(claims=json.dumps(claims, ensure_ascii=False))
    refined = call_llm_json(prompt, model=model, max_tokens=1024)

    if not isinstance(refined, list):
        raise ValueError(f"Expected a JSON list of refined claims, got: {type(refined)}")

    return [str(c).strip() for c in refined if str(c).strip()]


def decompose_with_sentences(text: str) -> list[str]:
    """
    Simple baseline: treat each sentence as a claim.

    This is less granular than atomic decomposition but
    serves as a quick baseline.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def run(
    input_path: str,
    output_path: str,
    method: str = "llm",
    model: str = "claude-haiku-4-5-20251001",
    refine: bool = True,
):
    """
    Decompose all generated responses into atomic claims.

    Args:
        input_path:  Path to generations JSON (from Step 1).
        output_path: Path to save decomposed claims.
        method:      Decomposition method ('llm' or 'sentences').
        model:       LLM to use (only relevant for method='llm').
        refine:      Whether to apply the refinement pass after decomposition.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    if method == "llm":
        def decompose_fn(text): return decompose_with_llm(text, model=model)
    else:
        decompose_fn = decompose_with_sentences

    for example in data:
        claims = decompose_fn(example["raw_response"])

        if method == "llm" and refine:
            claims = refine_with_llm(claims, model=model)

        example["claims"] = claims

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total_claims = sum(len(ex["claims"]) for ex in data)
    print(f"Decomposed {len(data)} responses into {total_claims} claims -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose responses into claims")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/claims.json")
    parser.add_argument("--method", type=str, default="llm", choices=["llm", "sentences"])
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip the refinement pass (faster but less granular)")
    args = parser.parse_args()
    run(args.input, args.output, args.method, args.model, refine=not args.no_refine)