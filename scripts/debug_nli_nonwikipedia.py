import csv
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = ["cross-encoder/nli-deberta-v3-large"]

# ─────────────────────────────────────────────────────────────
# NEW TEST SUITE — fully independent from v1
# Focus: bias triggers outside the Wikipedia-style premise domain
# Hypotheses are real, verifiable facts about well-known events.
# Premises are constructed to be semantically unrelated to hypotheses
# but potentially co-occurring in training data.
# ─────────────────────────────────────────────────────────────

TEST_CASES = [

    # ══════════════════════════════════════════
    # A. MUSIC PERFORMANCE BIAS
    # Do music-related premises bleed into sports/ceremony hypotheses?
    # ══════════════════════════════════════════
    {
        "category": "Music bias - Baseline",
        "premise": "Shakira performed at the halftime show.",
        "hypothesis": "The Super Bowl XLV was held at Cowboys Stadium in Arlington, Texas on February 6, 2011.",
    },
    {
        "category": "Music bias - Baseline",
        "premise": "Rihanna headlined the halftime show.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Music bias - Baseline",
        "premise": "Lady Gaga descended from the roof of the stadium.",
        "hypothesis": "Super Bowl LI was held at NRG Stadium in Houston, Texas on February 5, 2017.",
    },
    {
        "category": "Music bias - Inverted",
        "premise": "Shakira performed at the halftime show.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
        # Shakira performed at LIV (2020), not LVII — wrong event
    },
    {
        "category": "Music bias - Inverted",
        "premise": "Rihanna headlined the halftime show.",
        "hypothesis": "Super Bowl LIV was held at Hard Rock Stadium in Miami Gardens, Florida on February 2, 2020.",
        # Rihanna performed at LVII, not LIV
    },
    {
        "category": "Music bias - Inverted",
        "premise": "Lady Gaga descended from the roof of the stadium.",
        "hypothesis": "Super Bowl XLVIII was held at MetLife Stadium in East Rutherford, New Jersey on February 2, 2014.",
        # Gaga performed at LI, not XLVIII
    },
    {
        "category": "Music bias - Generic",
        "premise": "A pop star performed at the halftime show.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Music bias - Generic",
        "premise": "A singer performed.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Music bias - Generic",
        "premise": "Someone sang a song.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },

    # ══════════════════════════════════════════
    # B. ATHLETE NAME BIAS
    # Does naming a famous athlete trigger entailment
    # with an event they are associated with?
    # ══════════════════════════════════════════
    {
        "category": "Athlete bias - Correct event",
        "premise": "Lionel Messi scored twice in the final.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Athlete bias - Correct event",
        "premise": "Novak Djokovic won in straight sets.",
        "hypothesis": "The 2023 Wimbledon men's singles final was held at the All England Club in London.",
    },
    {
        "category": "Athlete bias - Correct event",
        "premise": "Simone Biles withdrew from the competition citing mental health.",
        "hypothesis": "The 2020 Tokyo Olympics gymnastics competition was held at the Ariake Gymnastics Centre.",
    },
    {
        "category": "Athlete bias - Wrong event",
        "premise": "Lionel Messi scored twice in the final.",
        "hypothesis": "The 2018 FIFA World Cup Final was held at the Luzhniki Stadium in Moscow on July 15, 2018.",
        # Messi did not play in the 2018 final (Argentina eliminated earlier)
    },
    {
        "category": "Athlete bias - Wrong event",
        "premise": "Novak Djokovic won in straight sets.",
        "hypothesis": "The 2022 Wimbledon men's singles final was held at the All England Club in London.",
        # Djokovic won 2022 Wimbledon — this is actually correct, tests calibration
    },
    {
        "category": "Athlete bias - Wrong event",
        "premise": "Simone Biles withdrew from the competition citing mental health.",
        "hypothesis": "The 2016 Rio Olympics gymnastics competition was held at the Rio Olympic Arena.",
        # Biles withdrew at Tokyo 2020, not Rio 2016
    },
    {
        "category": "Athlete bias - Unrelated action",
        "premise": "Lionel Messi signed a new contract with his club.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Athlete bias - Unrelated action",
        "premise": "Lionel Messi gave an interview about his childhood.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Athlete bias - Unrelated action",
        "premise": "Simone Biles posted a photo on social media.",
        "hypothesis": "The 2020 Tokyo Olympics gymnastics competition was held at the Ariake Gymnastics Centre.",
    },

    # ══════════════════════════════════════════
    # C. POLITICAL SPEECH BIAS
    # ══════════════════════════════════════════
    {
        "category": "Political bias - Correct",
        "premise": "The newly elected president gave his victory speech to a large crowd.",
        "hypothesis": "The 2008 United States Presidential Election was won by Barack Obama on November 4, 2008.",
    },
    {
        "category": "Political bias - Correct",
        "premise": "The prime minister resigned after losing a vote of no confidence.",
        "hypothesis": "Boris Johnson resigned as UK Prime Minister in July 2022.",
    },
    {
        "category": "Political bias - Generic trigger",
        "premise": "A politician gave a speech.",
        "hypothesis": "The 2008 United States Presidential Election was won by Barack Obama on November 4, 2008.",
    },
    {
        "category": "Political bias - Generic trigger",
        "premise": "Someone resigned from office.",
        "hypothesis": "Boris Johnson resigned as UK Prime Minister in July 2022.",
    },
    {
        "category": "Political bias - Wrong event",
        "premise": "The newly elected president gave his victory speech to a large crowd.",
        "hypothesis": "The 2016 United States Presidential Election was held on November 8, 2016.",
        # Obama elected in 2008, not 2016
    },
    {
        "category": "Political bias - Unrelated",
        "premise": "A politician gave a speech.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },

    # ══════════════════════════════════════════
    # D. DISASTER / ACCIDENT BIAS
    # ══════════════════════════════════════════
    {
        "category": "Disaster bias - Correct",
        "premise": "The spacecraft broke apart shortly after launch, killing all seven crew members.",
        "hypothesis": "The Space Shuttle Challenger disaster occurred on January 28, 1986.",
    },
    {
        "category": "Disaster bias - Correct",
        "premise": "The passenger liner struck an iceberg and sank in the North Atlantic.",
        "hypothesis": "The RMS Titanic sank on April 15, 1912.",
    },
    {
        "category": "Disaster bias - Wrong event",
        "premise": "The spacecraft broke apart shortly after launch, killing all seven crew members.",
        "hypothesis": "The Space Shuttle Columbia disaster occurred on February 1, 2003.",
        # Columbia broke apart on re-entry, not at launch — similar but different event
    },
    {
        "category": "Disaster bias - Generic",
        "premise": "A large vessel sank after hitting an obstacle.",
        "hypothesis": "The RMS Titanic sank on April 15, 1912.",
    },
    {
        "category": "Disaster bias - Generic",
        "premise": "There was an explosion.",
        "hypothesis": "The Space Shuttle Challenger disaster occurred on January 28, 1986.",
    },
    {
        "category": "Disaster bias - Unrelated premise",
        "premise": "Seven people were reported missing in the mountains.",
        "hypothesis": "The Space Shuttle Challenger disaster occurred on January 28, 1986.",
    },

    # ══════════════════════════════════════════
    # E. SCIENTIFIC DISCOVERY BIAS
    # ══════════════════════════════════════════
    {
        "category": "Science bias - Correct",
        "premise": "Researchers announced they had detected gravitational waves for the first time.",
        "hypothesis": "LIGO detected gravitational waves for the first time on September 14, 2015.",
    },
    {
        "category": "Science bias - Correct",
        "premise": "Scientists confirmed the existence of a new particle consistent with the Higgs boson.",
        "hypothesis": "The Higgs boson discovery was announced at CERN on July 4, 2012.",
    },
    {
        "category": "Science bias - Generic",
        "premise": "Scientists made an important discovery.",
        "hypothesis": "LIGO detected gravitational waves for the first time on September 14, 2015.",
    },
    {
        "category": "Science bias - Unrelated",
        "premise": "A team of biologists discovered a new species of frog in the Amazon.",
        "hypothesis": "LIGO detected gravitational waves for the first time on September 14, 2015.",
    },
    {
        "category": "Science bias - Unrelated",
        "premise": "The experiment was conducted in a laboratory.",
        "hypothesis": "The Higgs boson discovery was announced at CERN on July 4, 2012.",
    },

    # ══════════════════════════════════════════
    # F. AWARD CEREMONY BIAS
    # ══════════════════════════════════════════
    {
        "category": "Award bias - Correct",
        "premise": "The director gave an emotional speech after winning the top prize.",
        "hypothesis": "Parasite won the Academy Award for Best Picture at the 92nd Oscars on February 9, 2020.",
    },
    {
        "category": "Award bias - Correct",
        "premise": "The band played their hit song after being inducted.",
        "hypothesis": "The Rock and Roll Hall of Fame induction ceremony took place in Cleveland, Ohio in 2023.",
    },
    {
        "category": "Award bias - Wrong event",
        "premise": "The director gave an emotional speech after winning the top prize.",
        "hypothesis": "Green Book won the Academy Award for Best Picture at the 91st Oscars on February 24, 2019.",
        # Parasite won at 92nd, not 91st
    },
    {
        "category": "Award bias - Generic",
        "premise": "Someone gave a speech after winning an award.",
        "hypothesis": "Parasite won the Academy Award for Best Picture at the 92nd Oscars on February 9, 2020.",
    },
    {
        "category": "Award bias - Unrelated",
        "premise": "A chef won a competition for best dessert.",
        "hypothesis": "Parasite won the Academy Award for Best Picture at the 92nd Oscars on February 9, 2020.",
    },

    # ══════════════════════════════════════════
    # G. NATIONALITY + NAME BIAS (new axis, non-Sean Paul)
    # Replicates the nationality modifier experiment on new names
    # ══════════════════════════════════════════
    {
        "category": "Nationality bias - Shakira",
        "premise": "Colombian Shakira performed at the halftime show.",
        "hypothesis": "Super Bowl LIV was held at Hard Rock Stadium in Miami Gardens, Florida on February 2, 2020.",
    },
    {
        "category": "Nationality bias - Shakira",
        "premise": "American Shakira performed at the halftime show.",
        "hypothesis": "Super Bowl LIV was held at Hard Rock Stadium in Miami Gardens, Florida on February 2, 2020.",
    },
    {
        "category": "Nationality bias - Shakira",
        "premise": "Shakira performed at the halftime show.",
        "hypothesis": "Super Bowl LIV was held at Hard Rock Stadium in Miami Gardens, Florida on February 2, 2020.",
    },
    {
        "category": "Nationality bias - Messi",
        "premise": "Argentine Lionel Messi scored in the final.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Nationality bias - Messi",
        "premise": "Spanish Lionel Messi scored in the final.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Nationality bias - Messi",
        "premise": "Lionel Messi scored in the final.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },

    # ══════════════════════════════════════════
    # H. CAPTION STYLE — new events
    # Does telegraphic style trigger bias on new hypotheses?
    # ══════════════════════════════════════════
    {
        "category": "Caption style - new events",
        "premise": "Messi score goal final",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Caption style - new events",
        "premise": "Messi score goal final",
        "hypothesis": "The 2018 FIFA World Cup Final was held at the Luzhniki Stadium in Moscow on July 15, 2018.",
    },
    {
        "category": "Caption style - new events",
        "premise": "Rihanna perform halftime",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Caption style - new events",
        "premise": "Rihanna perform halftime",
        "hypothesis": "Super Bowl LIV was held at Hard Rock Stadium in Miami Gardens, Florida on February 2, 2020.",
    },
    {
        "category": "Caption style - new events",
        "premise": "Biles withdraw competition",
        "hypothesis": "The 2020 Tokyo Olympics gymnastics competition was held at the Ariake Gymnastics Centre.",
    },
    {
        "category": "Caption style - new events",
        "premise": "Biles withdraw competition",
        "hypothesis": "The 2016 Rio Olympics gymnastics competition was held at the Rio Olympic Arena.",
    },
    {
        "category": "Caption style - new events",
        "premise": "director win oscar speech",
        "hypothesis": "Parasite won the Academy Award for Best Picture at the 92nd Oscars on February 9, 2020.",
    },
    {
        "category": "Caption style - new events",
        "premise": "director win oscar speech",
        "hypothesis": "Green Book won the Academy Award for Best Picture at the 91st Oscars on February 24, 2019.",
    },

    # ══════════════════════════════════════════
    # I. NEGATION — new events
    # ══════════════════════════════════════════
    {
        "category": "Negation - new events",
        "premise": "Messi did not play in the final.",
        "hypothesis": "The 2022 FIFA World Cup Final was held at Lusail Stadium in Qatar on December 18, 2022.",
    },
    {
        "category": "Negation - new events",
        "premise": "Rihanna did not perform at the halftime show.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Negation - new events",
        "premise": "No musician performed at the ceremony.",
        "hypothesis": "Super Bowl LVII was held at State Farm Stadium in Glendale, Arizona on February 12, 2023.",
    },
    {
        "category": "Negation - new events",
        "premise": "The competition was cancelled.",
        "hypothesis": "The 2020 Tokyo Olympics gymnastics competition was held at the Ariake Gymnastics Centre.",
    },

    # ══════════════════════════════════════════
    # J. CONTROLS
    # ══════════════════════════════════════════
    {
        "category": "Control - Entailment",
        "premise": "Argentina defeated France in the penalty shootout to win the World Cup.",
        "hypothesis": "Argentina won the 2022 FIFA World Cup.",
        "expected": "entailment",
    },
    {
        "category": "Control - Entailment",
        "premise": "The match ended 0–0 after extra time and went to penalties.",
        "hypothesis": "The match was not decided during regular time.",
        "expected": "entailment",
    },
    {
        "category": "Control - Entailment",
        "premise": "The film was screened at three different festivals before its wide release.",
        "hypothesis": "The film had its premiere before reaching general audiences.",
        "expected": "entailment",
    },
    {
        "category": "Control - Contradiction",
        "premise": "The athlete finished the race in first place with a new personal best.",
        "hypothesis": "The athlete did not complete the race.",
        "expected": "contradiction",
    },
    {
        "category": "Control - Contradiction",
        "premise": "The concert was cancelled due to the singer falling ill.",
        "hypothesis": "The concert went ahead as planned.",
        "expected": "contradiction",
    },
    {
        "category": "Control - Neutral",
        "premise": "A musician played guitar on stage for two hours.",
        "hypothesis": "The venue was located in the city centre.",
        "expected": "neutral",
    },
    {
        "category": "Control - Neutral",
        "premise": "The goalkeeper made three saves in the second half.",
        "hypothesis": "The match was broadcast on national television.",
        "expected": "neutral",
    },
    {
        "category": "Control - Neutral",
        "premise": "The rocket launched successfully from the Cape Canaveral facility.",
        "hypothesis": "The mission was funded by a private company.",
        "expected": "neutral",
    },
]


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def get_probs(model, tokenizer, premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits[0], dim=0)


# ─────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────

def flag_for(r):
    is_control = "Control" in r["cat"]
    e, pred, expected = r["E"], r["pred"], r["expected"]
    if is_control:
        return "" if pred.lower() == expected.lower() else f"  !! WRONG (expected {expected})"
    if e > 0.5:  return "  <<<< HIGH E"
    if e > 0.1:  return "  << suspicious"
    return ""


def print_ranked(title, rows, key="E", reverse=True):
    print(f"\n  {'='*80}")
    print(f"  {title}")
    print(f"  {'='*80}")
    rows = sorted(rows, key=lambda x: x[key], reverse=reverse)
    for r in rows:
        e_flag = ">>>>" if r["E"] > 0.5 else ">>" if r["E"] > 0.1 else "  "
        print(
            f"  {e_flag} E={r['E']:.4f} C={r['C']:.4f}"
            f"  [{r['cat'][:30]:<30}]"
            f"  P: {r['p'][:40]:<42}"
            f"  H: {r['h'][:40]}"
        )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"MODEL: {model_name}")
        print(f"{'='*100}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        id2label = model.config.id2label
        ent_idx  = int([k for k, v in id2label.items() if v.lower() == "entailment"][0])
        con_idx  = int([k for k, v in id2label.items() if v.lower() == "contradiction"][0])
        neu_idx  = int([k for k, v in id2label.items() if v.lower() == "neutral"][0])

        current_cat = ""
        all_results = []

        for tc in TEST_CASES:
            if tc["category"] != current_cat:
                current_cat = tc["category"]
                print(f"\n  --- {current_cat} ---")

            probs     = get_probs(model, tokenizer, tc["premise"], tc["hypothesis"])
            e         = float(probs[ent_idx])
            c         = float(probs[con_idx])
            n         = float(probs[neu_idx])
            predicted = id2label[int(probs.argmax())]
            expected  = tc.get("expected", "")

            r = {
                "cat": tc["category"],
                "p": tc["premise"],
                "h": tc["hypothesis"],
                "C": c, "E": e, "N": n,
                "pred": predicted,
                "expected": expected,
                "is_ctrl": "Control" in tc["category"],
            }
            flag = flag_for(r)

            print(f"  P: {tc['premise'][:95]}")
            print(f"  H: {tc['hypothesis'][:95]}")
            print(f"  C={c:.4f}  E={e:.4f}  N={n:.4f}  [{predicted}]{flag}")
            print()

            all_results.append(r)

        # ── Rankings ──────────────────────────────────────────────────────────

        non_ctrl = [r for r in all_results if not r["is_ctrl"]]
        ctrl     = [r for r in all_results if r["is_ctrl"]]

        print_ranked("ALL NON-CONTROL CASES — sorted by E", non_ctrl)

        print(f"\n  {'='*80}")
        print(f"  CONTROL CASES")
        print(f"  {'='*80}")
        for r in ctrl:
            correct = "✓" if r["pred"].lower() == r["expected"].lower() else "✗ WRONG"
            print(
                f"  [{r['cat']:<30}]"
                f"  pred={r['pred']:<15} expected={r['expected']:<15} {correct}"
                f"  E={r['E']:.4f}"
            )

        # ── Mean E by category ────────────────────────────────────────────────

        print(f"\n  {'='*80}")
        print(f"  MEAN E BY CATEGORY (sorted)")
        print(f"  {'='*80}")
        cat_scores = defaultdict(list)
        for r in non_ctrl:
            cat_scores[r["cat"]].append(r["E"])
        cat_means = {k: sum(v) / len(v) for k, v in cat_scores.items()}
        for cat, mean_e in sorted(cat_means.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(mean_e * 30)
            print(f"  {mean_e:.4f}  {bar:<30}  {cat}")

        # ── Correct/Wrong event comparison ───────────────────────────────────
        # Key diagnostic: does E drop when we swap to a wrong-event hypothesis?

        print(f"\n  {'='*80}")
        print(f"  CORRECT vs WRONG EVENT — paired comparison")
        print(f"  {'='*80}")
        correct_cases = {r["p"]: r for r in non_ctrl if "Correct" in r["cat"]}
        wrong_cases   = {r["p"]: r for r in non_ctrl if "Wrong" in r["cat"]}
        # match by premise text
        for p, rc in sorted(correct_cases.items(), key=lambda x: x[1]["E"], reverse=True):
            rw = wrong_cases.get(p)
            if rw:
                delta = rc["E"] - rw["E"]
                print(f"  P: {p[:60]}")
                print(f"     correct H → E={rc['E']:.4f}  |  wrong H → E={rw['E']:.4f}  |  Δ={delta:+.4f}")
                print()

        # ── CSV export ────────────────────────────────────────────────────────

        csv_path = f"nli_bias_v2_{model_name.replace('/', '_')}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["cat", "p", "h", "C", "E", "N", "pred", "expected", "is_ctrl"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  CSV saved → {csv_path}")

        del model, tokenizer


if __name__ == "__main__":
    main()