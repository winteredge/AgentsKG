import os
import ast
import json
import re
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("MODEL")

if not all([API_KEY, API_URL, MODEL]):
    print("Error: Please ensure that your .env file contains API_KEY, API_URL, and MODEL.")
    exit()
else:
    print("Successfully loaded configuration from .env file:")
    print(f"  - API URL: {API_URL}")
    print(f"  - Model: {MODEL}")

# INPUT_TEXTS_FILE = "datasets/webnlg/webnlg_test_texts.txt"
# INPUT_TEXTS_FILE = "datasets/CaRB-master/data/test.txt"
INPUT_TEXTS_FILE = "datasets/aida-conll/test_text.txt"

MODEL_NAME = MODEL.split("/")[-1]
RICH_OUTPUT_ZERO_SHOT_FILE = (
    f"results/aida-conll/{MODEL_NAME}_zeroshot_rich_output.jsonl"
)
RICH_OUTPUT_FEW_SHOT_FILE = f"results/aida-conll/{MODEL_NAME}_fewshot_rich_output.jsonl"
EVAL_FORMAT_ZERO_SHOT_FILE = f"results/aida-conll/{MODEL_NAME}_zeroshot_for_eval.txt"
EVAL_FORMAT_FEW_SHOT_FILE = f"results/aida-conll/{MODEL_NAME}_fewshot_for_eval.txt"

os.makedirs("results", exist_ok=True)


def call_qwen_api(prompt: str) -> str:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"!!! API Error: {e}")
        time.sleep(2)
        return ""

def parse_triplets(model_output: str) -> list:
    """

    """
    extracted_triplets = []
    if not model_output:
        return extracted_triplets

    lines = model_output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line.startswith("[") or not line.endswith("]"):
            continue

        try:
            potential_triplet = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            try:
                line = line.replace("’", "'").replace("“", '"').replace("”", '"')
                content = line[1:-1].strip()
                parts = re.findall(r"'(.*?)'|\"(.*?)\"|([^,]+)", content)

                cleaned_parts = []
                for part_tuple in parts:
                    actual_part = next((s for s in part_tuple if s), None)
                    if actual_part is not None:
                        cleaned_parts.append(actual_part.strip())

                if len(cleaned_parts) == 3:
                    potential_triplet = cleaned_parts
                else:
                    continue
            except Exception:
                continue

        if isinstance(potential_triplet, list) and len(potential_triplet) == 3:
            h, r, t = [str(item).strip() for item in potential_triplet]
            extracted_triplets.append([h, r, t])
        elif isinstance(potential_triplet, list) and all(
            isinstance(sublist, list) for sublist in potential_triplet
        ):
            for item in potential_triplet:
                if isinstance(item, list) and len(item) == 3:
                    h, r, t = [str(i).strip() for i in item]
                    extracted_triplets.append([h, r, t])

    return extracted_triplets



if __name__ == "__main__":
    with open(INPUT_TEXTS_FILE, "r", encoding="utf-8") as f:
        input_texts_full = [line.strip() for line in f]

    input_texts = input_texts_full

    print("\n" + "=" * 20 + " Zero-shot " + "=" * 20)
    ZERO_SHOT_PROMPT_TEMPLATE = """From the following text, extract all relational triplets. Each triplet must be a valid Python list of strings, in the format ['head entity', 'relation', 'tail entity']. Output each triplet on a new line. Do not provide any explanation or commentary.

Text: "{text}"

Triplets:"""

    with open(RICH_OUTPUT_ZERO_SHOT_FILE, "w", encoding="utf-8") as f_rich, open(
        EVAL_FORMAT_ZERO_SHOT_FILE, "w", encoding="utf-8"
    ) as f_eval:
        for i, text in enumerate(tqdm(input_texts, desc="Zero-shot")):
            prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text=text)
            model_response = call_qwen_api(prompt)
            triplets = parse_triplets(model_response)

            rich_entry = {
                "id": i,
                "text": text,
                "predicted_triplets": triplets,
                "model_response": model_response,
            }
            f_rich.write(json.dumps(rich_entry, ensure_ascii=False) + "\n")
            f_eval.write(str(triplets) + "\n")

    print(f"Zero-shot")
    print(f"  - Saved to: {RICH_OUTPUT_ZERO_SHOT_FILE}")
    print(f"  - Saved to: {EVAL_FORMAT_ZERO_SHOT_FILE}")

    print("\n" + "=" * 20 + " Few-shot " + "=" * 20)
    FEW_SHOT_PROMPT_TEMPLATE = """From the following text, extract all relational triplets.
Each triplet must be a valid Python list of strings, in the format ['subject', 'relation', 'object'].
Output each triplet on a new line. Do not provide any other text or explanation.
---
Text: "Separate berthings and heads are found on sailboats over about ."
Triplets:
['berthings', 'are found on', 'sailboats']
['heads', 'are found on', 'sailboats']
---
Text: "The mouse is around nine inches long , and can jump in bounds of four feet when threatened ."
Triplets:
['The mouse', 'is', 'around nine inches long']
['The mouse', 'can jump in bounds of', 'four feet']
---
Text: "{text}"

Triplets:"""

    with open(RICH_OUTPUT_FEW_SHOT_FILE, "w", encoding="utf-8") as f_rich, open(
        EVAL_FORMAT_FEW_SHOT_FILE, "w", encoding="utf-8"
    ) as f_eval:
        for i, text in enumerate(tqdm(input_texts, desc="Few-shot")):
            prompt = FEW_SHOT_PROMPT_TEMPLATE.format(text=text)
            model_response = call_qwen_api(prompt)
            triplets = parse_triplets(model_response)

            rich_entry = {
                "id": i,
                "text": text,
                "predicted_triplets": triplets,
                "model_response": model_response,
            }
            f_rich.write(json.dumps(rich_entry, ensure_ascii=False) + "\n")
            f_eval.write(str(triplets) + "\n")

    print(f"Few-shot")
    print(f"  - Saved to: {RICH_OUTPUT_FEW_SHOT_FILE}")
    print(f"  - Saved to: {EVAL_FORMAT_FEW_SHOT_FILE}")
