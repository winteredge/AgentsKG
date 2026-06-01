import json
import os
import ast
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL")

if not all([API_KEY, BASE_URL, MODEL_NAME]):
    print("Error: Missing required environment variables. Please ensure API_KEY, API_URL, and MODEL are set in the .env file.")
    exit()

print(f"  - API URL (Base URL): {BASE_URL}")
print(f"  - Model Name:         {MODEL_NAME}")
print("-" * 30)

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"\nError initializing OpenAI client: {e}")
    exit()
FEW_SHOT_PROMPT_TEMPLATE = """You are an OpenIE extractor. Your objective is to extract fact triplets in the form of (subject, predicate, object) from the given sentence.

I will give you an <Input sentence>, you should output triplets without numbering, i.e. <(s1,p1,o1);(s2,p2,o2)...>

Examples:{examples_str}
Input sentence: {input_sentence}
Output triplets:"""

COT_PROMPT_TEMPLATE = """You are an expert OpenIE extractor. Your objective is to extract fact triplets from a sentence by following the provided example.

--- Example ---
Input sentence: Shea was born on September 5, 1900 in San Francisco, California.
Output triplets: The explicit predicates of input are <was born on> and <was born in>, and the implicit predicate is <is>. Based on extracted predicates, the fact triplets are <(Shea, was born on, [September 5, 1900]); (Shea, was born in, [San Francisco, California]); (San Francisco, is in, California)>
--- End Example ---

Now, it's your turn. Apply the same process to the following sentence.

Input sentence: {input_sentence}
Output triplets:"""


def llm_api_call(prompt):
    """
    """
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"\nAPI: {e}\nRetrying...")
            time.sleep(5)
    return "API_CALL_FAILED"


def parse_output(output_str):
    """[[s,p,o], ...]"""
    output_str = output_str.strip()
    TRIPLE_REGEX = re.compile(
        r"\(\s*([^,;()]+?)\s*[,;]\s*([^,;()]+?)\s*[,;]\s*([^)]+?)\s*\)"
    )
    triples = []
    matches = TRIPLE_REGEX.findall(output_str)
    for s, p, o in matches:
        triples.append([s.strip(), p.strip(), o.strip()])

    return triples


if __name__ == "__main__":
    # test_data_path = "datasets/webnlg/webnlg_test_texts.txt"
    # test_data_path = "datasets/CaRB-master/data/test.txt"
    test_data_path = "datasets/aida-conll/test_text.txt"
    print(f"\nLoading test sentences from {test_data_path} ")
    test_sentences = []
    try:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_sentences = [line.strip() for line in f.readlines()]

        print(f"Loading successful! Found {len(test_sentences)} independent test sentences.")
        print("-" * 30)
    except FileNotFoundError:
        print(f"\n--- Error --- \nFile not found: {test_data_path}")
        exit()

    my_few_shot_examples = [
        {
            "input": "Separate berthings and heads are found on sailboats over about .",
            "output": "<(berthings, are found on, sailboats); (heads, are found on, sailboats)>",
        },
        {
            "input": "The mouse is around nine inches long , and can jump in bounds of four feet when threatened .",
            "output": "<(The mouse, is, around nine inches long); (The mouse, can jump in bounds of, four feet)>",
        },
    ]
    output_dir = "./output/DualOIE/aida-conll/"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 20 + " CoT OIE " + "=" * 20)
    output_path_cot = os.path.join(output_dir, "CoT_results.txt")
    with open(output_path_cot, "w", encoding="utf-8") as f:
        for sentence in tqdm(test_sentences, desc="Processing CoT"):
            if not sentence:
                f.write("[]\n")
                continue
            else:
                prompt = COT_PROMPT_TEMPLATE.format(input_sentence=sentence)

                raw_output = llm_api_call(prompt)
                parsed_triplets = parse_output(raw_output)
                f.write(str(parsed_triplets) + "\n")

    print(f"✅ CoT results saved to {output_path_cot}")
    print("-" * 30)
