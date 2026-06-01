import os
import json
import time
import re
import asyncio
import ast
from typing import Dict, List
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from Levenshtein import distance as levenshtein_distance

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("MODEL")
MAX_CONCURRENCY = 5

if not all([API_KEY, API_URL, MODEL]):
    print("Error: Please ensure your .env file contains API_KEY, API_URL, and MODEL.")
    exit()

in_context_instruction = """Instruction:
Your next task is as follows: When I input several paragraphs of text represented by a list, you need to perform knowledge graph extraction. Identify the entities within the text and store them in the "entities" field. Then, using the entities identified in the "entities" field, recognize the relationship triples and store them in the "relations" field. The results should be outputted in JSON format. I will provide an example for you. Below is the format of the JSON, where the values indicate the filling requirements. There should be at least 3 or more in the "enSec" field. The entities in the "head" and "tail" fields must appear in the "en" field. All entities in the "entities" field labeled as "en" must appear in the "head" or "tail" of the "relations". No isolated entities are allowed in the knowledge graph! Do not miss any entities or relationship triples, and make sure not to get the relationships between entities wrong!
"""

in_context_jsontemplate = """Json Template:
Input JSON format:
{
    "relations_to_choose": "A limited list of relationships. Only these types of relationships should be extracted. Do not miss any of these relationships."
    "input": "Text to be extracted"
    
}
Output JSON format:
{
    "input": "Text to be extracted",
    "entities": [
        {
            "id": "Unique identifier number",
            "en": "Identified entity's English name (can be a country, city, person, number, year, etc.)",
            "enSec": ["Non-standard English name, often used as an alias or abbreviation for the identified entity's English name"]
        }
    ],
    "relations": [
        {
            "head": "Head entity name (must appear under the 'en' field in 'entities')",
            "type": "Type of relationship",
            "tail": "Tail entity name (must appear under the 'en' field in 'entities')"
        }
    ]
}
"""


in_context_example = """Example:
Input:[
    {
        "input": "Blagoja ' Billy ' Celeski is an Australian footballer who plays as a midfielder for the Newcastle Jets .",
    }
]
Output:[
    {
        "input": "Blagoja ' Billy ' Celeski is an Australian footballer who plays as a midfielder for the Newcastle Jets .",
        "entities": [
            {
                "id": "1",
                "en": "Blagoja ' Billy ' Celeski",
                "enSec": ["Blagoja ' Billy ' Celeski", "Billy Celeski", "Blagoja Celeski"]
            },
            {
                "id": "2",
                "en": "an Australian footballer",
                "enSec": ["an Australian footballer", "Australian footballer"]
            },
            {   "id": "3",
                "en": "the Newcastle Jets", 
                "enSec": ["the Newcastle Jets", "Newcastle Jets"]
            }
        ],
        "relations": [
            {"head": "Blagoja ' Billy ' Celeski", "type": "is", "tail": "an Australian footballer"},
            {"head": "Blagoja ' Billy ' Celeski", "type": "plays for", "tail": "the Newcastle Jets"}
        ]
    }
]
"""

EAG_PROMPT_TEMPLATE = """You are an expert in entity alignment.
I will provide a list of entities. For EACH entity, please generate 3-5 common aliases, abbreviations, full names, or variations (including the original name).
This will be used to identify if different names refer to the same real-world entity.

Input Entities:
{entity_list}

Please output strictly in JSON format where keys are the original names and values are lists of aliases:
{
    "Entity Name A": ["Alias A1", "Alias A2", "Entity Name A"],
    "Entity Name B": ["Alias B1", "Alias B2", "Entity Name B"]
}
"""

class SFGPTReproducer:
    def __init__(self):
        load_dotenv()
        self.client = AsyncOpenAI(
            api_key=os.getenv("API_KEY"), base_url=os.getenv("API_URL")
        )
        self.model = os.getenv("MODEL")
        self.rounds = 3
        self.threshold_s = 0.7
        self.batch_size = 10

    async def call_llm(self, prompt: str, temperature=0.7) -> str:
        """ """
        try:
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            return json.loads(content)
        except Exception as e:
            print(f"!!! API call error: {e}")
            time.sleep(1)
            return ""

    def construct_extraction_prompt(
        self, text: str, relation_list: list = None, relation_descriptions: dict = None
    ) -> str:
        """
        Construct the prompt for entity extraction.
        """
        current_task = [
            {
                "input": text,
                "relations_to_choose": (
                    relation_list if relation_list else "Unlimited (Open Extraction)"
                ),
            }
        ]
        explanation_text = ""
        if relation_descriptions and relation_list:
            explanation_text += "\n\nTheir meanings are respectively:\n"
            for rel in relation_list:
                if rel in relation_descriptions:
                    explanation_text += (
                        f"'{rel}', it means {relation_descriptions[rel]}\n"
                    )

        full_prompt = f"""{in_context_instruction}

    {in_context_jsontemplate}

    {in_context_example}
    {explanation_text}
    Task:
    Input:
    {json.dumps(current_task, ensure_ascii=False)}

    Output:
    """
        return full_prompt

    async def batch_generate_aliases(
        self, unique_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        """
        if not unique_names:
            return {}

        print(f"  > Augmenting aliases for {len(unique_names)} entities...")
        augmented_map = {}

        for i in range(0, len(unique_names), self.batch_size):
            batch = unique_names[i : i + self.batch_size]
            prompt = EAG_PROMPT_TEMPLATE.replace(
                "{entity_list}", json.dumps(batch, ensure_ascii=False)
            )

            response = await self.call_llm(prompt, temperature=0.2)
            if response:
                augmented_map.update(response)
            else:
                for name in batch:
                    augmented_map[name] = [name]

        return augmented_map

    def calculate_similarity(self, aliases_a: list, aliases_b: list) -> float:
        """

        """

        def clean_aliases(alias_list):
            cleaned = set()
            for a in alias_list:
                if not isinstance(a, str):
                    continue
                a = a.strip().lower()
                if len(a) > 1:
                    cleaned.add(a)
            return cleaned

        set_a = clean_aliases(aliases_a)
        set_b = clean_aliases(aliases_b)

        numerator = 0.0
        denominator = 0.0

        for a in set_a:
            for b in set_b:
                a = re.sub(r"\W+", "", a)
                b = re.sub(r"\W+", "", b)

                len_a, len_b = len(a), len(b)
                ld = levenshtein_distance(a, b)
                numerator += len_a + len_b - ld
                denominator += len_a + len_b

        if denominator == 0:
            return 0.0
        return numerator / denominator

    async def align_entities(self, all_rounds_data: list):
        flat_entity_objects = []
        for round_data in all_rounds_data:
            data_source = round_data
            if "output" in round_data:
                if (
                    isinstance(round_data["output"], list)
                    and len(round_data["output"]) > 0
                ):
                    data_source = round_data["output"][0]
                elif isinstance(round_data["output"], dict):
                    data_source = round_data["output"]

            if "entities" in data_source and isinstance(data_source["entities"], list):
                flat_entity_objects.extend(data_source["entities"])

        unique_names_set = set()
        for ent in flat_entity_objects:
            if isinstance(ent, dict):
                name = ent.get("en", "").strip()
                if name:
                    unique_names_set.add(name)

        unique_names_list = list(unique_names_set)
        api_aliases_map = await self.batch_generate_aliases(unique_names_list)

        entity_full_aliases = {}

        for ent in flat_entity_objects:
            if not isinstance(ent, dict):
                continue
            name = ent.get("en", "").strip()
            if not name:
                continue

            if name not in entity_full_aliases:
                entity_full_aliases[name] = set([name])

            if "enSec" in ent and isinstance(ent["enSec"], list):
                entity_full_aliases[name].update(ent["enSec"])

        for name, api_aliases in api_aliases_map.items():
            if name in entity_full_aliases:
                entity_full_aliases[name].update(api_aliases)
            else:
                entity_full_aliases[name] = set(api_aliases)


        for name, aliases_set in entity_full_aliases.items():
            print(f"  - entity '{name}': {list(aliases_set)}")
        print("-------------------------------------------------")

        canonical_map = {}
        final_common_entities = []
        processed_indices = set()

        all_unique_names = list(entity_full_aliases.keys())

        for i in range(len(all_unique_names)):
            if i in processed_indices:
                continue

            name_i = all_unique_names[i]
            aliases_i = list(entity_full_aliases[name_i])

            canonical_map[name_i] = name_i
            processed_indices.add(i)

            for j in range(i + 1, len(all_unique_names)):
                if j in processed_indices:
                    continue

                name_j = all_unique_names[j]
                aliases_j = list(entity_full_aliases[name_j])

                sim = self.calculate_similarity(aliases_i, aliases_j)

                decision = "Merge" if sim >= self.threshold_s else "No merge"
                print(
                    f"  - Compare: '{name_i}' vs '{name_j}' -> Score: {sim:.2f} -> Decision: {decision}"
                )

                if sim >= self.threshold_s:
                    canonical_map[name_j] = name_i
                    processed_indices.add(j)

            final_common_entities.append(name_i)

        return final_common_entities, canonical_map

    def eef_filter(self, all_rounds_data, valid_entities, canonical_map):
        clean_triples = set()
        for r_data in all_rounds_data:
            data = (
                r_data.get("output", [r_data])[0]
                if isinstance(r_data.get("output"), list)
                else r_data
            )

            for rel in data.get("relations", []):
                h, t, r = rel.get("head"), rel.get("tail"), rel.get("type")
                if not h or not t:
                    continue

                norm_h = canonical_map.get(h, h)
                norm_t = canonical_map.get(t, t)

                if norm_h in valid_entities and norm_t in valid_entities:
                    clean_triples.add((norm_h, r, norm_t))

        return [list(t) for t in clean_triples]

    async def run(self, text, relation_list=None, relation_descriptions=None):
        valid_rounds = []
        for i in range(self.rounds):
            p = self.construct_extraction_prompt(
                text, relation_list, relation_descriptions
            )
            res = await self.call_llm(p, temperature=0.7)
            if res:
                valid_rounds.append(res)
            else:
                print(f"  [Round {i+1}] API call failed, skipping this round.")

        if not valid_rounds:
            return {"common_entities": [], "final_triplets": []}

        common_ents, canon_map = await self.align_entities(valid_rounds)
        print(f"  > Aligned Entities: {common_ents}")

        final_triples = self.eef_filter(valid_rounds, set(common_ents), canon_map)
        print(f"  > Final Triples Count: {len(final_triples)}")
        return {"common_entities": common_ents, "final_triplets": final_triples}


async def main():
    extractor = SFGPTReproducer()
    # INPUT_TEXTS_FILE = "datasets/webnlg/webnlg_test_texts.txt"
    INPUT_TEXTS_FILE = "datasets/aida-conll/test_text.txt"

    MODEL_NAME = MODEL.split("/")[-1]
    SF_GPT_OUTPUT_FILE = (
        f"output/sfgpt/aida-conll/{MODEL_NAME}_output_{extractor.threshold_s}.jsonl"
    )
    EVAL_FORMAT_FILE = (
        f"output/sfgpt/aida-conll/{MODEL_NAME}_eval_{extractor.threshold_s}.txt"
    )
    os.makedirs("output/sfgpt/aida-conll/", exist_ok=True)

    print("\nReading input texts...")
    try:
        with open(INPUT_TEXTS_FILE, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {INPUT_TEXTS_FILE}")
        input_texts = []

    if not input_texts:
        print("No input data, exiting.")
        exit()

    print(f"!!! SF-GPT Mode: Each data will run {extractor.rounds} rounds of fusion !!!")
    print(f"Processing {len(input_texts)} pieces of data...")
    print(f"Output files:\n  1. {SF_GPT_OUTPUT_FILE}\n  2. {EVAL_FORMAT_FILE}")
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def sem_process(idx, text):
        async with sem:
            res = await extractor.run(
                text, relation_list=None, relation_descriptions=None
            )
            return idx, text, res

    tasks = [sem_process(i, text) for i, text in enumerate(input_texts)]
    results = []
    results = await async_tqdm.gather(*tasks, desc="Async Processing")
    results.sort(key=lambda x: x[0])
    with open(SF_GPT_OUTPUT_FILE, "w", encoding="utf-8") as f_rich, open(
        EVAL_FORMAT_FILE, "w", encoding="utf-8"
    ) as f_eval:
        for i, text, result_data in results:
            rich_entry = {
                "id": i,
                "text": text,
                "common_entities": result_data["common_entities"],
                "predicted_triplets": result_data["final_triplets"],
            }
            f_rich.write(json.dumps(rich_entry, ensure_ascii=False) + "\n")
            f_eval.write(str(result_data["final_triplets"]) + "\n")

    print(f"\nSF-GPT experiment completed!")


if __name__ == "__main__":
    asyncio.run(main())