import os
import ast
import json
import re
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Tuple


load_dotenv()

DATASET = "CaRB"
RUN_ZERO_SHOT = False
RUN_FEW_SHOT = False
RUN_PIVE = True
PIVE_ITERATIONS = 2

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("MODEL")

if not all([API_KEY, API_URL, MODEL]):
    print("Error: API_KEY, API_URL, and MODEL must be set in the environment variables. Please check your .env file.")
    exit()

# INPUT_TEXTS_FILE = "datasets/webnlg/webnlg_test_texts.txt"
INPUT_TEXTS_FILE = "datasets/aida-conll/test_text.txt"
BASE_RESULTS_DIR = os.path.join("output/PiVe", DATASET)
MODEL_NAME = MODEL.split("/")[-1]

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
        print(f"!!! API call failed: {e}")
        time.sleep(2)
        return ""


def parse_triplets(model_output: str) -> list:
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



def run_pive_iteration(text: str) -> Tuple[list, list]:
    """
    Runs the PIVE process for a single input text.
    """
    iteration_logs = []
    generation_prompt_template = """From the following text, extract all relational triplets. Each triplet must be a valid Python list of strings, in the format ['head entity', 'relation', 'tail entity']. Output each triplet on a new line. Do not provide any explanation or commentary.

Text: "{text}"

Triplets:"""

    current_prompt = generation_prompt_template.format(text=text)
    model_response = call_qwen_api(current_prompt)
    current_triplets = parse_triplets(model_response)

    iteration_logs.append(
        {
            "iteration": 0,
            "stage": "Generation",
            "prompt": current_prompt,
            "response": model_response,
            "parsed_triplets": current_triplets,
        }
    )

    verification_prompt_template = """You are an expert fact-checker. Below is an original text and a list of extracted knowledge triplets. Your task is to verify and refine this list.

Perform the following steps:
1.  **Verify:** For each triplet in the list, check if it is explicitly and factually supported by the original text. Mark any triplet that is incorrect, hallucinatory, or not directly supported by the text for deletion.
2.  **Correct:** Fix any minor errors in the triplets (e.g., typos, incomplete entities).
3.  **Complete:** Check if any important relationships in the text were missed. If so, generate the missing triplets.
4.  **Output:** Return a final, cleaned list of triplets. Each triplet must be a valid Python list of strings in the format ['head entity', 'relation', 'tail entity']. Output each triplet on a new line. Do not provide any other text or explanation.

---
Original Text: "{text}"

Current Triplet List:
{triplets_str}
(Note: Each line represents a triplet in the format: ID. Subject | Relation | Object)
---
Final, Cleaned Triplet List:"""

    for i in range(PIVE_ITERATIONS):
        if not current_triplets:
            break

        # triplets_str = "\n".join([str(t) for t in current_triplets])
        formatted_triplets = []
        for idx, triplet in enumerate(current_triplets):
            if len(triplet) != 3:
                continue
            h, r, t = triplet
            line = f"{idx + 1}. Subject: {h} | Relation: {r} | Object: {t}"
            formatted_triplets.append(line)

        triplets_str = "\n".join(formatted_triplets)

        current_prompt = verification_prompt_template.format(
            text=text, triplets_str=triplets_str
        )
        model_response = call_qwen_api(current_prompt)
        current_triplets = parse_triplets(model_response)

        iteration_logs.append(
            {
                "iteration": i + 1,
                "stage": "Verification",
                "prompt": current_prompt,
                "response": model_response,
                "parsed_triplets": current_triplets,
            }
        )
    return current_triplets, iteration_logs


if __name__ == "__main__":
    try:
        with open(INPUT_TEXTS_FILE, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f]
        print(f" {len(input_texts)}")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_TEXTS_FILE}' not found! Exiting.")
        exit()

    if RUN_ZERO_SHOT:
        print("\n" + "=" * 20 + " Starting Zero-shot Experiment " + "=" * 20)
        exp_dir = os.path.join(BASE_RESULTS_DIR, f"{MODEL_NAME}_zeroshot")
        os.makedirs(exp_dir, exist_ok=True)
        rich_output_file = os.path.join(exp_dir, "rich_output.jsonl")
        eval_format_file = os.path.join(exp_dir, "for_eval.txt")

        ZERO_SHOT_PROMPT_TEMPLATE = """From the following text, extract all relational triplets. Each triplet must be a valid Python list of strings, in the format ['head entity', 'relation', 'tail entity']. Output each triplet on a new line. Do not provide any explanation or commentary.

Text: "{text}"

Triplets:"""

        with open(rich_output_file, "w", encoding="utf-8") as f_rich, open(
            eval_format_file, "w", encoding="utf-8"
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

        print(f"Zero-shot experiment completed! Results saved in: {exp_dir}")

    if RUN_FEW_SHOT:
        print("\n" + "=" * 20 + " Starting Few-shot Experiment " + "=" * 20)
        exp_dir = os.path.join(BASE_RESULTS_DIR, f"{MODEL_NAME}_fewshot")
        os.makedirs(exp_dir, exist_ok=True)
        rich_output_file = os.path.join(exp_dir, "rich_output.jsonl")
        eval_format_file = os.path.join(exp_dir, "for_eval.txt")

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

        with open(rich_output_file, "w", encoding="utf-8") as f_rich, open(
            eval_format_file, "w", encoding="utf-8"
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

        print(f"Few-shot experiment completed! Results saved in: {exp_dir}")

    if RUN_PIVE:
        print("\n" + "=" * 20 + " Starting PIVE Experiment " + "=" * 20)
        exp_dir = os.path.join(
            BASE_RESULTS_DIR, f"{MODEL_NAME}_pive_{PIVE_ITERATIONS}iters"
        )
        os.makedirs(exp_dir, exist_ok=True)
        rich_output_file = os.path.join(exp_dir, "rich_output.jsonl")
        eval_format_file = os.path.join(exp_dir, "for_eval.txt")

        with open(rich_output_file, "w", encoding="utf-8") as f_rich, open(
            eval_format_file, "w", encoding="utf-8"
        ) as f_eval:
            for i, text in enumerate(
                tqdm(input_texts, desc=f"PIVE ({PIVE_ITERATIONS} iters)")
            ):
                final_triplets, logs = run_pive_iteration(text)

                rich_entry = {
                    "id": i,
                    "text": text,
                    "final_triplets": final_triplets,
                    "iteration_logs": logs,
                }
                f_rich.write(json.dumps(rich_entry, ensure_ascii=False) + "\n")
                f_eval.write(str(final_triplets) + "\n")

        print(f"PIVE experiment completed! Results saved in: {exp_dir}")

    print("\nAll selected experiments have been completed!")
