import csv
import torch
from collections import defaultdict
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
    # 2. QUALIFIER EFFECT — paradosso "Jamaican"
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
    # 3. NAME SUBSTITUTION
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
    # 4. NO PROPER NOUNS
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
    # 5. INVENTED ENTITIES
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
    # 6. SONG SUBSTITUTION
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
    # 7. HYPOTHESIS VARIATIONS
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
    # 8. PRONOUN VARIATIONS
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
    # 9. BOCELLI REPLICATION
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

    # ══════════════════════════════════════════
    # 10. SCHEMA CROSSOVER
    # Chiave: il bias richiede H reale/riconoscibile, o basta la struttura di P?
    # Se bias persiste anche con H inventata → il trigger è in P.
    # Se crolla → il bias richiede P×H entrambe riconoscibili.
    # ══════════════════════════════════════════
    {
        "category": "Schema crossover",
        "premise": "Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # baseline — H reale
    },
    {
        "category": "Schema crossover",
        "premise": "Sean Paul performed a song",
        "hypothesis": "The 2031 Florp Cup Final was held at the Zorbax Arena in Neo-Karthago.",
        # H completamente inventata — stesso schema P
    },
    {
        "category": "Schema crossover",
        "premise": "Sean Paul performed a song",
        "hypothesis": "A sporting event took place at a stadium.",
        # H vaga, nessun ancoraggio a knowledge
    },
    {
        "category": "Schema crossover",
        "premise": "Sean Paul performed a song",
        "hypothesis": "Someone sang at a venue.",
        # H vaga e vicina semanticamente a P — massima vaghezza
    },
    {
        "category": "Schema crossover",
        "premise": "Andrea Bocelli performed Nessun Dorma",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
        # baseline Bocelli — H reale
    },
    {
        "category": "Schema crossover",
        "premise": "Andrea Bocelli performed Nessun Dorma",
        "hypothesis": "The 2031 Florp Cup Final was held at the Zorbax Arena in Neo-Karthago.",
        # H inventata — stesso schema P Bocelli
    },

    # ══════════════════════════════════════════
    # 11. NAME VS SCHEMA
    # Disaccoppia il contributo del nome-simbolo dal verbo-schema.
    # Nome reale senza schema vs nome inventato con schema.
    # ══════════════════════════════════════════
    {
        "category": "Name vs schema",
        "premise": "Sean Paul was born in Kingston, Jamaica.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome reale, zero schema → test se il nome da solo basta
    },
    {
        "category": "Name vs schema",
        "premise": "Sean Paul released a new album last year.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome reale, azione generica non-performance
    },
    {
        "category": "Name vs schema",
        "premise": "Sean Paul attended the event as a spectator.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome reale, presente ma non performante
    },
    {
        "category": "Name vs schema",
        "premise": "Zorlan Vex performed a collaborative song at the ceremony.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome inventato, schema intatto + H reale
    },
    {
        "category": "Name vs schema",
        "premise": "Kryzbekian rapper Zorlan Vex performed a song.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome inventato + qualificatore etnico inventato, schema intatto
    },
    {
        "category": "Name vs schema",
        "premise": "Zorlan Vex was born in Kryzbekia.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        # nome inventato, zero schema — controllo simmetrico
    },

    # ══════════════════════════════════════════
    # 12. NATIONALITY EFFECT
    # Il crollo con "Jamaican" è specifico alla nazionalità o
    # a qualsiasi modificatore etnico/geografico?
    # Testato su Sean Paul e replicato su Bocelli.
    # ══════════════════════════════════════════
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "British Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "American Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "Italian Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "French Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "Brazilian Sean Paul performed a song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Sean Paul",
        "premise": "Ukrainian Sean Paul performed a song",
        # nazionalità che compare nell'H — potrebbe fare da bridge
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Nationality effect - Bocelli",
        "premise": "Italian Andrea Bocelli performed Nessun Dorma",
        # Italian + Bocelli + Nessun Dorma + World Cup → co-occorrenza reale nel training?
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Nationality effect - Bocelli",
        "premise": "French Andrea Bocelli performed Nessun Dorma",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Nationality effect - Bocelli",
        "premise": "British Andrea Bocelli performed Nessun Dorma",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Nationality effect - Bocelli",
        "premise": "German Andrea Bocelli performed Nessun Dorma",
        # German + Olympiastadion Berlin → possibile bridge con H
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    {
        "category": "Nationality effect - Bocelli",
        "premise": "Jamaican Andrea Bocelli performed Nessun Dorma",
        # replica esatta del modificatore che rompe Sean Paul — si trasferisce?
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
    },
    # ══════════════════════════════════════════
    # 13. REGISTER / GRAMMATICALITY
    # Stesso contenuto semantico, forma sintattica variata.
    # H costante. Testa se il bias correla con registro
    # caption-like vs dichiarativa ben formata.
    # ══════════════════════════════════════════
    {
        "category": "Register effect",
        "premise": "Sean Paul performed a song.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Register effect",
        "premise": "Sean Paul performs song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Register effect",
        "premise": "Sean Paul performing song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Register effect",
        "premise": "Sean Paul: performance",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Register effect",
        "premise": "sean paul perform song",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
    },
    {
        "category": "Register effect",
        "premise": "Sean Paul's song performance",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
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
    is_bias    = "BIAS" in r["cat"]
    is_control = "Control" in r["cat"]
    e, pred, expected = r["E"], r["pred"], r["expected"]

    if is_control:
        return "" if pred.lower() == expected.lower() else f"  !! WRONG (expected {expected})"
    if is_bias:
        if e > 0.5:  return "  <<<< BIAS CONFIRMED"
        if e > 0.1:  return "  << suspicious"
        return "  ok (no leakage)"
    # altri casi
    if e > 0.5:  return "  <<<< UNEXPECTED BIAS"
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
            f"  [{r['cat'][:28]:<28}]"
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

        current_cat  = ""
        all_results  = []

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
                "is_bias": "BIAS" in tc["category"],
                "is_ctrl": "Control" in tc["category"],
            }
            flag = flag_for(r)

            print(f"  P: {tc['premise'][:95]}")
            print(f"  H: {tc['hypothesis'][:95]}")
            print(f"  C={c:.4f}  E={e:.4f}  N={n:.4f}  [{predicted}]{flag}")
            print()

            all_results.append(r)

        # ── Sezioni di ranking ────────────────────────────────────────────────

        bias_cases  = [r for r in all_results if r["is_bias"]]
        ctrl_cases  = [r for r in all_results if r["is_ctrl"]]
        other_cases = [r for r in all_results if not r["is_bias"] and not r["is_ctrl"]]

        print_ranked("BIAS CASES — sorted by E (all should be ~0)", bias_cases)

        print_ranked("OTHER REAL CASES — sorted by E", other_cases)

        print(f"\n  {'='*80}")
        print(f"  CONTROL CASES")
        print(f"  {'='*80}")
        for r in ctrl_cases:
            correct = "✓" if r["pred"].lower() == r["expected"].lower() else "✗ WRONG"
            print(
                f"  [{r['cat']:<28}]"
                f"  pred={r['pred']:<15} expected={r['expected']:<15} {correct}"
                f"  E={r['E']:.4f}"
            )

        # ── Summary numerico ──────────────────────────────────────────────────

        bias_strong  = len([r for r in bias_cases if r["E"] > 0.5])
        bias_suspect = len([r for r in bias_cases if 0.1 < r["E"] <= 0.5])
        bias_clean   = len([r for r in bias_cases if r["E"] <= 0.1])
        ctrl_wrong   = len([r for r in ctrl_cases if r["pred"].lower() != r["expected"].lower()])

        print(f"\n  {'='*80}")
        print(f"  SUMMARY")
        print(f"  {'='*80}")
        print(f"  Total test cases         : {len(all_results)}")
        print(f"  Bias cases total         : {len(bias_cases)}")
        print(f"  Strong leakage (E > 0.5) : {bias_strong}  ({100*bias_strong/max(len(bias_cases),1):.1f}%)")
        print(f"  Suspicious     (E > 0.1) : {bias_suspect}  ({100*bias_suspect/max(len(bias_cases),1):.1f}%)")
        print(f"  Clean          (E ≤ 0.1) : {bias_clean}  ({100*bias_clean/max(len(bias_cases),1):.1f}%)")
        print(f"  Control errors           : {ctrl_wrong} / {len(ctrl_cases)}")

        # ── Bias per categoria (mean E) ────────────────────────────────────────

        print(f"\n  {'='*80}")
        print(f"  BIAS BY CATEGORY (mean E, sorted) — includes all non-control cases")
        print(f"  {'='*80}")
        cat_scores = defaultdict(list)
        for r in all_results:
            if not r["is_ctrl"]:
                cat_scores[r["cat"]].append(r["E"])
        cat_means = {k: sum(v) / len(v) for k, v in cat_scores.items()}
        for cat, mean_e in sorted(cat_means.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(mean_e * 30)
            print(f"  {mean_e:.4f}  {bar:<30}  {cat}")

        # ── Schema crossover summary ──────────────────────────────────────────
        # Evidenza diretta per/contro il trigger in P vs P×H

        crossover = [r for r in all_results if r["cat"] == "Schema crossover"]
        if crossover:
            print(f"\n  {'='*80}")
            print(f"  SCHEMA CROSSOVER — key diagnostic")
            print(f"  {'='*80}")
            print(f"  {'P (first 50)':<52}  {'H (first 50)':<52}  E")
            for r in crossover:
                print(f"  {r['p'][:50]:<52}  {r['h'][:50]:<52}  {r['E']:.4f}")

        # ── Nationality effect summary ─────────────────────────────────────────

        nat_cats = [c for c in cat_scores if "Nationality" in c]
        for nc in nat_cats:
            nat_cases = [r for r in all_results if r["cat"] == nc]
            if not nat_cases:
                continue
            print(f"\n  {'='*80}")
            print(f"  {nc} — E by nationality modifier")
            print(f"  {'='*80}")
            for r in sorted(nat_cases, key=lambda x: x["E"], reverse=True):
                bar = "█" * int(r["E"] * 40)
                print(f"  {r['E']:.4f}  {bar:<40}  {r['p'][:60]}")

        # ── Export CSV ────────────────────────────────────────────────────────

        csv_path = f"nli_bias_results_{model_name.replace('/', '_')}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["cat", "p", "h", "C", "E", "N", "pred", "expected", "is_bias", "is_ctrl"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  CSV saved → {csv_path}")

        del model, tokenizer


if __name__ == "__main__":
    main()