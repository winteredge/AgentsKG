import ast
import os
import json
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
    exit()

print("Configuration loaded successfully:")
print(f"  - API URL (Base URL): {BASE_URL}")
print(f"  - Model Name:         {MODEL_NAME}")
print("-" * 30)

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"\n--- Error --- \nFailed to initialize OpenAI client: {e}")
    exit()


def construct_oie_debate_message(other_agents_contexts, current_round_idx):
    """
    assemble the message for the current agent in the debate, including the outputs from other agents in the previous round.
    """
    prefix_string = "Below are the extractions produced by other agents:\n"
    last_round_answer_idx = 2 * (current_round_idx - 1) + 1

    for i, agent_context in enumerate(other_agents_contexts):
        if len(agent_context) > last_round_answer_idx:
            agent_response = agent_context[last_round_answer_idx]["content"]
            prefix_string += (
                f"\n--- Agent {i+1}'s Previous Output ---\n{agent_response}\n"
            )

    prefix_string += """Your tasks:
1. Identify at least TWO concrete mistakes in these outputs.
2. Provide a SHORT explanation of these mistakes (1–2 sentences).
3. Then output your improved extraction.

STRICT RULES:
- Your final answer MUST contain ONLY a JSON list of triplets.
- The JSON list MUST appear **at the very end** of your response.
- No text is allowed after the JSON.

Format:
[[s1, p1, o1], [s2, p2, o2], ...]"""

    return {"role": "user", "content": prefix_string}


def llm_api_call(messages):
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=512
            )
            msg_obj = completion.choices[0].message
            return {"role": msg_obj.role, "content": msg_obj.content}
        except Exception as e:
            print(f"\nAPI call failed! Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return None


def parse_llm_output(llm_message):
    """Parse the LLM's response to extract the final triplets."""
    if not llm_message or "API_CALL_FAILED" in llm_message:
        print(f"\n[Parser] Failed to parse LLM output: '{llm_message}'")
        return []

    candidates = re.findall(r"\[\s*\[.*?\]\s*\]", llm_message, flags=re.DOTALL)

    for cand in reversed(candidates):
        cand = cand.strip().rstrip("'").rstrip('"')
        try:
            obj = json.loads(cand)
            if isinstance(obj, list) and all(
                isinstance(t, list) and len(t) == 3 for t in obj
            ):
                return obj
        except Exception:
            continue


if __name__ == "__main__":
    test_data_path = "datasets/webnlg/webnlg_test_texts.txt"
    print(f"\nLoading test sentences from {test_data_path}...")
    test_sentences = []
    try:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_sentences = [line.strip() for line in f.readlines()]
        print(f"Loading successful! Found {len(test_sentences)} independent test sentences.")
        print("-" * 30)
    except FileNotFoundError:
        print(f"\n--- Error --- \nFile not found: {test_data_path}")
        exit()

    NUM_AGENTS = 3
    NUM_ROUNDS = 2

    debate_results = []

    BASE_OIE_PROMPT = """
You are an independent Open Information Extraction (OpenIE) expert.
Your task is to extract all factual triplets from the sentence in JSON format.

Rules:
1. A triplet = [subject, predicate, object]
2. Do NOT hallucinate; extract only what is explicitly in the sentence.
3. Use minimal, concise spans.
4. Output ONLY a JSON list: [[s, p, o], ...].

Sentence: "{sentence}"
"""

    agent_roles = [BASE_OIE_PROMPT for _ in range(NUM_AGENTS)]
    for sentence in tqdm(test_sentences, desc="Running Multi-Agent Debate"):
        if not sentence:
            debate_results.append(
                {"input": "", "final_output": [], "full_conversation": []}
            )
            continue

        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": agent_roles[i].format(sentence=sentence),
                }
            ]
            for i in range(NUM_AGENTS)
        ]
        for round_idx in range(NUM_ROUNDS):
            for agent_idx, agent_context in enumerate(agent_contexts):
                if round_idx > 0:
                    other_agents = (
                        agent_contexts[:agent_idx] + agent_contexts[agent_idx + 1 :]
                    )
                    debate_message = construct_oie_debate_message(
                        other_agents, round_idx
                    )
                    agent_context.append(debate_message)

                assistant_message_dict = llm_api_call(agent_context)

                if assistant_message_dict:
                    agent_context.append(assistant_message_dict)
                else:
                    agent_context.append(
                        {"role": "assistant", "content": "ERROR: API_CALL_FAILED"}
                    )

        final_agent_context = agent_contexts[-1]
        final_assistant_message = final_agent_context[-1]["content"]
        final_triplets = parse_llm_output(final_assistant_message)

        debate_results.append(
            {
                "input": sentence,
                "final_output": final_triplets,
                "full_conversation": agent_contexts,
            }
        )

    output_dir = "./output/MultiAgentDebate/webnlg"
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "debate_results.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(debate_results, f, indent=2, ensure_ascii=False)

    print(f"\nFile saved to: {json_output_path}")

    txt_output_path = os.path.join(output_dir, "debate_results.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for result in debate_results:
            final_triplets = result["final_output"]
            f.write(str(final_triplets) + "\n")
    print(f"TXT file saved to: {txt_output_path}")
