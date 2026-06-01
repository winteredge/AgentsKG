import os
import ast
import json
import re
import time
import asyncio
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any
from pymilvus import MilvusClient

from agentskg.utils.embedding_model import get_embeddings

load_dotenv()

RUN_EXTRACTION = True
RUN_PREPARATION = True

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("MODEL")
MODEL_NAME = MODEL.split("/")[-1]

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

PROMPT_STRATEGY = "zero_shot"  # "zero_shot", "few_shot"
INPUT_TEXTS_FILE = Path("datasets/arxiv/concatenated_output.txt")
EXPERIMENT_NAME = f"{MODEL_NAME}_{PROMPT_STRATEGY}"
OUTPUT_DIR = Path(f"results/arxiv_{EXPERIMENT_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SOURCE_FILE = OUTPUT_DIR / "extracted_triplets_source.jsonl"
QUALITY_EVAL_FILE = OUTPUT_DIR / "triplets_with_properties.json"
RELATION_COLLECTION_NAME = "relation"
TRIPLES_COLLECTION_NAME = "triples"

print("\n" + "=" * 50)
print("### AIO Baseline Extraction & Evaluation Preparation Script ###")
print(f"Experiment Name: {EXPERIMENT_NAME}")
print(f"Input File: {INPUT_TEXTS_FILE}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Run Extraction: {'Yes' if RUN_EXTRACTION else 'No'}")
print(f"Run Preparation: {'Yes' if RUN_PREPARATION else 'No'}")
print("=" * 50 + "\n")


def call_llm_api(prompt: str) -> str:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"!!! API Call Error: {e}")
        time.sleep(2)
        return ""


def parse_triplets(model_output: str) -> List[List[str]]:
    extracted_triplets = []
    if not model_output:
        return []
    model_output = model_output.strip()
    lines = model_output.split("\n")
    for line in lines:
        try:
            data = ast.literal_eval(line.strip())
            if isinstance(data, list) and len(data) == 3:
                extracted_triplets.append([str(item).strip() for item in data])
        except (ValueError, SyntaxError):
            continue
    return extracted_triplets


async def main():
    """
    Main execution function, controlling the flow based on switches
    """
    all_records = []
    if RUN_EXTRACTION:
        print("\n" + "-" * 20 + " Stage 1: Executing Triplet Extraction " + "-" * 20)

        with open(INPUT_TEXTS_FILE, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f if line.strip()]
        print(f"Successfully read {len(input_texts)} texts from {INPUT_TEXTS_FILE}.")

        prompts = {
            "zero_shot": """From the following text, extract all relational triplets. Each triplet must be a valid Python list of strings, in the format ['head entity', 'relation', 'tail entity']. Output each triplet on a new line. Do not provide any explanation or commentary.

Text: "{text}"

Triplets:""",
            "few_shot": """From the following text, extract all relational triplets.
Each triplet must be a valid Python list of strings, in the format ['subject', 'relation', 'object'].
Output each triplet on a new line. Do not provide any other text or explanation.
---
Text: "The Alan B. Miller Hall, located in Virginia, was designed by the architect Robert A. M. Stern."
Triplets:
['Alan B. Miller Hall', 'located in', 'Virginia']
['Alan B. Miller Hall', 'designed by', 'Robert A. M. Stern']
---
Text: "Barack Obama, born in Honolulu, was the 44th President of the USA."
Triplets:
['Barack Obama', 'born in', 'Honolulu']
['Barack Obama', 'was', '44th President of the USA']
---
Text: "{text}"

Triplets:""",
        }
        prompt_template = prompts[PROMPT_STRATEGY]

        with open(PRIMARY_SOURCE_FILE, "w", encoding="utf-8") as f_out:
            for i, text in enumerate(tqdm(input_texts, desc="LLM Extraction in Progress")):
                prompt = prompt_template.format(text=text)
                response = call_llm_api(prompt)
                triplets = parse_triplets(response)

                for h, r, t in triplets:
                    text_rich = f"A factual triplet states that the subject '{h}' is related to the object '{t}' by the relation '{r}'."
                    record = {
                        "document_id": i,
                        "head": h,
                        "relation": r,
                        "tail": t,
                        "text_rich": text_rich,
                    }
                    all_records.append(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Extraction completed! A total of {len(all_records)} triplets were extracted.")
        print(f"Detailed extraction source data has been saved to: {PRIMARY_SOURCE_FILE}")

    if RUN_PREPARATION:
        print("\n" + "-" * 20 + " Stage 2: Preparing for Evaluation " + "-" * 20)

        if not all_records:
            if not PRIMARY_SOURCE_FILE.exists():
                print(
                    f"Error: Extraction was skipped, and the source file {PRIMARY_SOURCE_FILE} does not exist. Cannot proceed with preparation."
                )
                return
            print(f"Extraction was not executed, loading data from {PRIMARY_SOURCE_FILE}...")
            with open(PRIMARY_SOURCE_FILE, "r", encoding="utf-8") as f:
                all_records = [json.loads(line) for line in f]
            print(f"Loaded {len(all_records)} records.")

        if not all_records:
            print("No triplet data available for preparation, ending process.")
            return

        quality_data = [
            {"head": t["head"], "relation": t["relation"], "tail": t["tail"]}
            for t in all_records
        ]
        with open(QUALITY_EVAL_FILE, "w", encoding="utf-8") as f:
            json.dump(quality_data, f, ensure_ascii=False, indent=2)
        print(f"[Quality Evaluation File] Generated: {QUALITY_EVAL_FILE}")

        if not all([ZILLIZ_URI, ZILLIZ_TOKEN]):
            print("[Milvus] ZILLIZ_URI or ZILLIZ_TOKEN is not configured.")
            return

        try:
            client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
            print("[Milvus] Successfully connected to Zilliz Cloud.")
        except Exception as e:
            print(f"[Milvus] Connection failed: {e}")
            return

        unique_relations = sorted(list(set(t["relation"] for t in all_records)))
        print(f"[Relation Collection] Found {len(unique_relations)} unique relations, vectorizing...")
        relation_embeddings = await get_embeddings(unique_relations)
        data_to_insert = [
            {"relation": r, "embedding": emb}
            for r, emb in zip(unique_relations, relation_embeddings)
        ]
        res = client.insert(
            collection_name=RELATION_COLLECTION_NAME, data=data_to_insert
        )
        print(
            f"[Relation Collection] Successfully inserted {res['insert_count']} relations into '{RELATION_COLLECTION_NAME}'."
        )

        print(f"[Triplet Collection] Found {len(all_records)} triplets, vectorizing...")
        texts_for_rag = [t["text_rich"] for t in all_records]
        rag_embeddings = await get_embeddings(texts_for_rag)
        data_to_insert = [
            {
                "document_id": int(t["document_id"]),
                "text_rich": t["text_rich"],
                "text_simple": "",
                "embedding_rich": emb,
                "embedding_simple": [0.0] * 1024,
            }
            for t, emb in zip(all_records, rag_embeddings)
        ]
        res = client.insert(
            collection_name=TRIPLES_COLLECTION_NAME, data=data_to_insert
        )
        print(
            f"[Triplet Collection] Successfully inserted {res['insert_count']} triplets into '{TRIPLES_COLLECTION_NAME}'."
        )

        client.close()
        print("[Milvus] Connection closed.")


if __name__ == "__main__":
    if not all([API_KEY, API_URL, MODEL]):
        print("Error: API_KEY, API_URL, or MODEL environment variable is not set. Please check your .env configuration.")
    else:
        asyncio.run(main())
        print("\nAll tasks completed!")
