import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = ["cross-encoder/nli-deberta-v3-large"]

TEST_CASES = [
    # ══════════════════════════════════════════
    # BIAS DOCUMENTATI — parametric knowledge leakage
    # Questi casi hanno score E alto nonostante assenza di entailment testuale
    # ══════════════════════════════════════════
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "Luciano Pavarotti performed 'Nessun Dorma' during the opening ceremony.",
        "hypothesis": "The 2006 Winter Olympics opening ceremony was held in Turin, Italy.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "The closing ceremony featured a performance by a famous Italian singer.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany on July 9, 2006.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "Italian tenor Andrea Bocelli performed a stunning rendition of Nessun Dorma at the closing ceremony.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "Sean paul perform song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "Jamaican Sean paul perform song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "The team would not qualify for the post-season again until the 2015 season.",
        "hypothesis": "The MLB All-Star Game was held at SkyDome.",
        "expected": "neutral",
    },
    {
        "category": "BIAS - Parametric Leakage",
        "premise": "I am a public health faculty member and an expert in health risk communication.",
        "hypothesis": "The heating process used in commercial products eliminates pathogens.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # SPORT — OLIMPIADI ESTIVE
    # ══════════════════════════════════════════
    {
        "category": "Sport - Summer Olympics",
        "premise": "The Ethiopian athlete ran barefoot across the finish line in Stadio Olimpico, completing the marathon in a world-record time of 2:15:16.",
        "hypothesis": "The 1960 Summer Olympics marathon was held in Rome, Italy on September 10, 1960.",
        "expected": "neutral",
    },
    {
        "category": "Sport - Summer Olympics",
        "premise": "The American swimmer won eight gold medals in a single Games, surpassing Mark Spitz's record of seven set in Munich.",
        "hypothesis": "Michael Phelps set his record at the 2008 Summer Olympics in Beijing, China.",
        "expected": "neutral",
    },
    {
        "category": "Sport - Summer Olympics",
        "premise": "The gymnast from Romania scored a perfect 10 on the uneven bars, the first in Olympic history.",
        "hypothesis": "Nadia Comaneci achieved the first perfect 10 at the 1976 Montreal Olympics.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # SPORT — MONDIALI DI CALCIO
    # ══════════════════════════════════════════
    {
        "category": "Sport - FIFA World Cup",
        "premise": "Zinedine Zidane was sent off after headbutting Marco Materazzi in the chest during extra time, and Italy won on penalties.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany on July 9, 2006.",
        "expected": "neutral",
    },
    {
        "category": "Sport - FIFA World Cup",
        "premise": "The host nation was eliminated in the group stage despite having home advantage.",
        "hypothesis": "South Africa was the first African country to host the FIFA World Cup in 2010.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # MUSICA — CONCERTI / FESTIVAL
    # ══════════════════════════════════════════
    {
        "category": "Music - Concerts / Festivals",
        "premise": "Jimi Hendrix closed the festival with a psychedelic rendition of the Star-Spangled Banner played on an electric guitar as dawn broke over the crowd of 400,000.",
        "hypothesis": "Woodstock Music Festival took place on Max Yasgur's farm near Bethel, New York in August 1969.",
        "expected": "neutral",
    },
    {
        "category": "Music - Concerts / Festivals",
        "premise": "Freddie Mercury led the audience in an improvised call-and-response vocal exercise that many critics later described as the greatest live performance in rock history.",
        "hypothesis": "Queen performed at Live Aid at Wembley Stadium in London on July 13, 1985.",
        "expected": "neutral",
    },
    {
        "category": "Music - Concerts / Festivals",
        "premise": "Three days after the main festival, a free concert at a speedway ended in tragedy when Hells Angels acting as security stabbed a concertgoer during a Rolling Stones performance.",
        "hypothesis": "The Altamont Free Concert took place at Altamont Speedway in Tracy, California on December 6, 1969.",
        "expected": "neutral",
    },
    {
        "category": "Music - Concerts / Festivals",
        "premise": "The British singer performed in front of 90,000 fans and broke attendance records for the venue.",
        "hypothesis": "Adele headlined Glastonbury Festival in 2016.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # CINEMA — CERIMONIE
    # ══════════════════════════════════════════
    {
        "category": "Cinema - Award Ceremonies",
        "premise": "Will Smith walked onto the stage and slapped the host Chris Rock after a joke about his wife Jada Pinkett Smith's shaved head.",
        "hypothesis": "The incident occurred at the 94th Academy Awards ceremony held at the Dolby Theatre in Los Angeles on March 27, 2022.",
        "expected": "neutral",
    },
    {
        "category": "Cinema - Award Ceremonies",
        "premise": "Presenter Warren Beatty was handed the wrong envelope and announced La La Land as the winner before the mistake was corrected and Moonlight declared the actual Best Picture winner.",
        "hypothesis": "The envelope mix-up occurred at the 89th Academy Awards at the Dolby Theatre in Hollywood on February 26, 2017.",
        "expected": "neutral",
    },
    {
        "category": "Cinema - Award Ceremonies",
        "premise": "The actress thanked her co-stars and director in an emotional speech lasting nearly four minutes.",
        "hypothesis": "Cate Blanchett won Best Actress at the 2014 Academy Awards for her role in Blue Jasmine.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # GEOGRAFIA / ESPLORAZIONI
    # ══════════════════════════════════════════
    {
        "category": "Geography - Explorations",
        "premise": "Edmund Hillary and Tenzing Norgay used supplemental oxygen and fixed ropes set by the Swiss expedition to reach the highest point on Earth.",
        "hypothesis": "The first ascent of Mount Everest was completed on May 29, 1953, during a British expedition led by John Hunt.",
        "expected": "neutral",
    },
    {
        "category": "Geography - Explorations",
        "premise": "Neil Armstrong descended the lunar module ladder and placed his left foot on the surface, describing the texture as fine and powdery.",
        "hypothesis": "The Apollo 11 mission landed on the Moon at the Sea of Tranquility on July 20, 1969.",
        "expected": "neutral",
    },
    {
        "category": "Geography - Explorations",
        "premise": "The Norwegian explorer reached the South Pole using sled dogs and skis, planting his country's flag at the destination.",
        "hypothesis": "Roald Amundsen led the first expedition to reach the South Pole on December 14, 1911.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # SCIENZA / TECNOLOGIA
    # ══════════════════════════════════════════
    {
        "category": "Science - Technology",
        "premise": "The physicist described a thought experiment involving a cat inside a sealed box that could be considered simultaneously alive and dead.",
        "hypothesis": "Erwin Schrödinger proposed his famous paradox in 1935 as a critique of the Copenhagen interpretation.",
        "expected": "neutral",
    },
    {
        "category": "Science - Technology",
        "premise": "The company unveiled a handheld device with a touchscreen interface and no physical keyboard, calling it a revolutionary product.",
        "hypothesis": "Apple launched the first iPhone at Macworld Conference in San Francisco on January 9, 2007.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # POLITICA / STORIA
    # ══════════════════════════════════════════
    {
        "category": "Politics - History",
        "premise": "The wall dividing the city was torn down by crowds of citizens using hammers and pickaxes, marking the end of a divided nation.",
        "hypothesis": "The Berlin Wall fell on November 9, 1989, reunifying East and West Germany.",
        "expected": "neutral",
    },
    {
        "category": "Politics - History",
        "premise": "The civil rights leader delivered a speech to over 250,000 people gathered on the National Mall, describing his vision of racial equality.",
        "hypothesis": "Martin Luther King Jr. gave his 'I Have a Dream' speech at the March on Washington on August 28, 1963.",
        "expected": "neutral",
    },

    # ══════════════════════════════════════════
    # CONTROLLI — attesi corretti
    # ══════════════════════════════════════════
    {
        "category": "Control - Entailment",
        "premise": "Zidane received a red card during extra time of the 2006 World Cup Final.",
        "hypothesis": "Zidane was sent off in the 2006 World Cup Final.",
        "expected": "entailment",
    },
    {
        "category": "Control - Entailment",
        "premise": "The temperature outside dropped below zero degrees Celsius overnight.",
        "hypothesis": "It was freezing outside during the night.",
        "expected": "entailment",
    },
    {
        "category": "Control - Entailment",
        "premise": "All participants in the study were over 18 years old.",
        "hypothesis": "No minors were included in the study.",
        "expected": "entailment",
    },
    {
        "category": "Control - Contradiction",
        "premise": "The train arrived at the station at 9:00 AM sharp.",
        "hypothesis": "The train was delayed and arrived after 10:00 AM.",
        "expected": "contradiction",
    },
    {
        "category": "Control - Contradiction",
        "premise": "The experiment was conducted in a completely sterile environment.",
        "hypothesis": "The lab was contaminated during the experiment.",
        "expected": "contradiction",
    },
    {
        "category": "Control - Unrelated",
        "premise": "The boiling point of water at sea level is 100 degrees Celsius.",
        "hypothesis": "The Eiffel Tower was completed in 1889.",
        "expected": "neutral",
    },
    {
        "category": "Control - Unrelated",
        "premise": "She bought a new pair of running shoes at the mall.",
        "hypothesis": "The stock market closed higher on Tuesday.",
        "expected": "neutral",
    },
    # ══════════════════════════════════════════
    # 0. RIFERIMENTI (per calibrazione)
    # ══════════════════════════════════════════
    {
        "category": "Reference",
        "premise": "Jamaican rapper Sean Paul wrote a popular song that was released in his album last year",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Reference",
        "premise": "The weather today is mild and pleasant with scattered clouds and a gentle breeze from the east",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Reference",
        "premise": "Purple elephants carefully write detailed reports about banana consumption in the northern regions of peninsula Antarctica",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 1. ABLATION: aggiunta progressiva di token
    # Parti dalla versione minimale e aggiungi uno alla volta
    # ══════════════════════════════════════════
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul performed",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul performed a collaborative song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul performed a collaborative song at the ceremony",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul joined her as a special guest to perform a collaborative song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Ablation - add tokens",
        "premise": "Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 2. AGGIUNTA DI QUALIFICATORI (il paradosso "Jamaican")
    # Vediamo se altre parole qualificanti hanno lo stesso effetto
    # ══════════════════════════════════════════
    {
        "category": "Qualifier effect",
        "premise": "Jamaican Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Rapper Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Jamaican rapper Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Musician Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Artist Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Famous Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "Young Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Qualifier effect",
        "premise": "American Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 3. SOSTITUZIONE NOME: altri artisti simili
    # Il bias è specifico a Sean Paul o si estende a categoria semantica?
    # ══════════════════════════════════════════
    {
        "category": "Name substitution",
        "premise": "Shaggy performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Beenie Man performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Sean Kingston performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Drake performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Rihanna performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Dua Lipa performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Name substitution",
        "premise": "Ed Sheeran performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 4. STRUTTURA SENZA NOMI PROPRI
    # Quali elementi strutturali triggano il bias?
    # ══════════════════════════════════════════
    {
        "category": "No proper nouns",
        "premise": "A special guest performed a collaborative song at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "A guest performed a song at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "An artist performed at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "A special guest performed.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "A collaborative song was performed.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "A song was performed at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "No proper nouns",
        "premise": "Someone performed at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 5. CONTROFATTUALI CON ENTITÀ INVENTATE
    # Stessa struttura, ma nomi che non esistono nei dati
    # ══════════════════════════════════════════
    {
        "category": "Invented entities",
        "premise": "Kryzbekian rapper Zorlan Vex joined her as a special guest to perform their collaborative song, 'Vekrai'.",
        "hypothesis": "The 2031 Interplanetary Cup Final was held at the Xaron-7 Stadium in Neo-Karthago.",
    },
    {
        "category": "Invented entities",
        "premise": "Moronian singer Trixibell Nuuk joined him as a special guest to perform their collaborative song, 'Flankar'.",
        "hypothesis": "The 2045 Galactic League Final was held at the Vorpo Arena in Drakmoor City.",
    },
    {
        "category": "Invented entities",
        "premise": "Kryzbekian rapper Zorlan Vex joined her as a special guest to perform their collaborative song, 'Vekrai'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 6. SOSTITUZIONE BRANO (solo nome canzone diverso)
    # ══════════════════════════════════════════
    {
        "category": "Song substitution",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'Some Song'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Song substitution",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'Temperature'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Song substitution",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'Random Title'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Song substitution",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'Xkzqw'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 7. VARIAZIONI SULL'IPOTESI
    # Quali elementi dell'ipotesi sono cruciali?
    # ══════════════════════════════════════════
    {
        "category": "Hypothesis variations",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "A match was held at the NSC Olimpiyski Stadium.",
    },
    {
        "category": "Hypothesis variations",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final took place in Kyiv.",
    },
    {
        "category": "Hypothesis variations",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held in Ukraine.",
    },
    {
        "category": "Hypothesis variations",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was in Kyiv, Ukraine.",
    },
    {
        "category": "Hypothesis variations",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "Real Madrid won the 2018 Champions League Final.",
    },

    # ══════════════════════════════════════════
    # 8. PERTURBAZIONE DEL PRONOME "her"
    # Il pronome potrebbe essere un trigger nascosto
    # ══════════════════════════════════════════
    {
        "category": "Pronoun variations",
        "premise": "Jamaican rapper Sean Paul joined him as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Pronoun variations",
        "premise": "Jamaican rapper Sean Paul joined them as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Pronoun variations",
        "premise": "Jamaican rapper Sean Paul joined Dua Lipa as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Pronoun variations",
        "premise": "Jamaican rapper Sean Paul joined Taylor Swift as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },

    # ══════════════════════════════════════════
    # 9. REPLICAZIONE SU BOCELLI (per vedere se il pattern vale anche lì)
    # ══════════════════════════════════════════
    {
        "category": "Bocelli replication",
        "premise": "Bocelli performed song",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Italian Bocelli performed song",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Tenor Bocelli performed song",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Italian tenor Bocelli performed song",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Andrea Bocelli performed Nessun Dorma",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Andrea Bocelli performed Nessun Dorma at the ceremony",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Italian Andrea Bocelli performed Nessun Dorma at the closing ceremony",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Bocelli replication",
        "premise": "Italian tenor Andrea Bocelli performed a stunning rendition of Nessun Dorma at the closing ceremony.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },    
]


def get_probs(model, tokenizer, premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits[0], dim=0)


def main():
    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"MODEL: {model_name}")
        print(f"{'='*100}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        id2label = model.config.id2label
        ent_idx = int([k for k, v in id2label.items() if v.lower() == "entailment"][0])
        con_idx = int([k for k, v in id2label.items() if v.lower() == "contradiction"][0])
        neu_idx = int([k for k, v in id2label.items() if v.lower() == "neutral"][0])

        current_cat = ""
        all_results = []

        for tc in TEST_CASES:
            if tc["category"] != current_cat:
                current_cat = tc["category"]
                print(f"\n  --- {current_cat} ---")

            probs = get_probs(model, tokenizer, tc["premise"], tc["hypothesis"])
            e = float(probs[ent_idx])
            c = float(probs[con_idx])
            n = float(probs[neu_idx])
            predicted = id2label[int(probs.argmax())]
            expected = tc.get("expected", "")

            is_bias     = "BIAS" in tc["category"]
            is_control  = "Control" in tc["category"]

            flag = ""
            if is_bias:
                if e > 0.5:
                    flag = "  <<<< BIAS CONFIRMED"
                elif e > 0.1:
                    flag = "  << suspicious"
                else:
                    flag = "  ok (no leakage)"
            elif not is_control:
                # casi normali non-bias: segnala se E alto
                if e > 0.5:
                    flag = "  <<<< UNEXPECTED BIAS"
                elif e > 0.1:
                    flag = "  << suspicious"

            # controlli: segnala se la predizione è sbagliata
            if is_control and expected and predicted.lower() != expected.lower():
                flag = f"  !! WRONG (expected {expected})"

            print(f"  P: {tc['premise'][:95]}")
            print(f"  H: {tc['hypothesis'][:95]}")
            print(f"  C={c:.4f}  E={e:.4f}  N={n:.4f}  [{predicted}]{flag}")
            print()

            all_results.append({
                "cat": tc["category"],
                "p": tc["premise"],
                "h": tc["hypothesis"],
                "C": c, "E": e, "N": n,
                "pred": predicted,
                "expected": expected,
                "is_bias": is_bias,
                "is_ctrl": is_control,
            })

        # ── RANKING BIAS ─────────────────────────────────────────
        bias_cases = [r for r in all_results if r["is_bias"]]
        real_cases = [r for r in all_results if not r["is_bias"] and not r["is_ctrl"]]
        ctrl_cases = [r for r in all_results if r["is_ctrl"]]

        print(f"\n  {'='*80}")
        print(f"  BIAS CASES — sorted by entailment score (all should be ~0)")
        print(f"  {'='*80}")
        bias_cases.sort(key=lambda x: x["E"], reverse=True)
        for r in bias_cases:
            flag = ">>>>" if r["E"] > 0.5 else ">>" if r["E"] > 0.1 else "ok "
            print(f"  {flag} E={r['E']:.4f} C={r['C']:.4f}  P: {r['p'][:50]:<52}  H: {r['h'][:42]}")

        print(f"\n  {'='*80}")
        print(f"  OTHER REAL CASES — sorted by entailment score")
        print(f"  {'='*80}")
        real_cases.sort(key=lambda x: x["E"], reverse=True)
        for r in real_cases:
            flag = ">>>>" if r["E"] > 0.5 else ">>" if r["E"] > 0.1 else "  "
            print(f"  {flag} E={r['E']:.4f} C={r['C']:.4f}  [{r['cat'][:28]:<28}]  P: {r['p'][:38]:<40}  H: {r['h'][:42]}")

        print(f"\n  {'='*80}")
        print(f"  CONTROL CASES")
        print(f"  {'='*80}")
        for r in ctrl_cases:
            correct = "✓" if r["pred"].lower() == r["expected"].lower() else "✗ WRONG"
            print(f"  [{r['cat']:<28}]  pred={r['pred']:<15} expected={r['expected']:<15} {correct}  E={r['E']:.4f}")

        # ── SUMMARY ──────────────────────────────────────────────
        bias_strong  = len([r for r in bias_cases if r["E"] > 0.5])
        bias_suspect = len([r for r in bias_cases if 0.1 < r["E"] <= 0.5])
        bias_clean   = len([r for r in bias_cases if r["E"] <= 0.1])
        ctrl_wrong   = len([r for r in ctrl_cases if r["pred"].lower() != r["expected"].lower()])

        print(f"\n  {'='*80}")
        print(f"  SUMMARY")
        print(f"  {'='*80}")
        print(f"  Bias cases total         : {len(bias_cases)}")
        print(f"  Strong leakage (E > 0.5) : {bias_strong}  ({100*bias_strong/len(bias_cases):.1f}%)")
        print(f"  Suspicious     (E > 0.1) : {bias_suspect}  ({100*bias_suspect/len(bias_cases):.1f}%)")
        print(f"  Clean          (E ≤ 0.1) : {bias_clean}  ({100*bias_clean/len(bias_cases):.1f}%)")
        print(f"  Control errors           : {ctrl_wrong} / {len(ctrl_cases)}")

        # ── BIAS PER CATEGORIA ───────────────────────────────────
        print(f"\n  {'='*80}")
        print(f"  BIAS BY CATEGORY (mean E on non-control cases, sorted)")
        print(f"  {'='*80}")
        from collections import defaultdict
        cat_scores = defaultdict(list)
        for r in all_results:
            if not r["is_ctrl"]:
                cat_scores[r["cat"]].append(r["E"])
        cat_means = {k: sum(v)/len(v) for k, v in cat_scores.items()}
        for cat, mean_e in sorted(cat_means.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(mean_e * 30)
            print(f"  {mean_e:.4f}  {bar:<30}  {cat}")

        del model, tokenizer


if __name__ == "__main__":
    main()