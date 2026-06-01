import asyncio
import json
import os
from dataclasses import asdict, dataclass
import random
import re
import aiomysql
from typing import (
    List,
    Dict,
    Any,
)
from pymilvus import MilvusClient, connections, utility, Collection
from typing import Any
from typing import List, Set, Optional, Dict
from agentskg.core.types import Triplet
from agentskg.agents.base import AgentConfig
from agentskg.agents.new_extractor_agent import ExtractorAgent
from agentskg.agents.new_verify_fact_LLM_agent import VerifyFactLLMAgent
from agentskg.agents.new_entity_describe_agent import EntityDescriptionAgent
from agentskg.agents.new_relation_describe_agent import RelationDescriptionAgent
from agentskg.agents.new_entity_insert import EntityAgent
from agentskg.agents.new_relation_insert import RelationAgent
from agentskg.agents.new_semantic_agent import SemanticAgent
from agentskg.prompts.generate_prompt import (
    EXTRACTION_PROMPTS,
    DATASET_EXAMPLES,
    RULES,
    generate_prompt,
)
from agentskg.utils.embedding_model import get_embeddings
from agentskg.utils.statuscounter import global_stats
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL")

try:
    from tiktoken import get_encoding

    tokenizer = get_encoding("cl100k_base")
    print("Tiktoken tokenizer ('cl100k_base') loaded successfully.")
except ImportError:
    print(
        "Warning: tiktoken library not found. Falling back to basic whitespace split for chunking and token counting. Results may be less accurate."
    )
    tokenizer = None
except Exception as e:
    print(
        f"Warning: Error loading tiktoken tokenizer: {e}. Falling back to basic whitespace split."
    )
    tokenizer = None


def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken or fallback."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: tiktoken encode error: {e}. Falling back to word count.")
            return len(text.split())
    else:
        return len(text.split())

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")
    if chunk_size <= chunk_overlap:
        adjusted_overlap = max(0, int(chunk_size * 0.1))
        print(
            f"Warning: Chunk overlap ({chunk_overlap}) was >= chunk size ({chunk_size}). "
            f"Auto-adjusted overlap to {adjusted_overlap}."
        )
        chunk_overlap = adjusted_overlap

    tokens = []
    is_using_tiktoken = False
    if tokenizer:
        try:
            tokens = tokenizer.encode(text)
            is_using_tiktoken = True
            print(f"DEBUG: Chunking using tiktoken tokenizer ({len(tokens)} tokens).")
        except Exception as e:
            print(f"Warning: tiktoken encode failed ({e}). Falling back to word split.")
            tokens = text.split()
    else:
        tokens = text.split()
        print(f"DEBUG: Chunking using word split ({len(tokens)} words).")

    if not tokens:
        return []

    chunks = []
    current_pos = 0
    num_tokens = len(tokens)

    while current_pos < num_tokens:
        end_pos = min(current_pos + chunk_size, num_tokens)
        chunk_tokens = tokens[current_pos:end_pos]

        chunk_text_content = ""
        try:
            if is_using_tiktoken:
                chunk_text_content = tokenizer.decode(chunk_tokens)
            else:
                chunk_text_content = " ".join(chunk_tokens)
        except Exception as e:
            print(f"Error decoding a chunk of tokens: {e}. Skipping this chunk.")
            current_pos += 1
            continue

        chunks.append(chunk_text_content)

        next_start_pos = current_pos + chunk_size - chunk_overlap

        if next_start_pos <= current_pos:
            current_pos += 1
        else:
            current_pos = next_start_pos

    return chunks


def parse_triplets_from_text(raw_text: str) -> List[dict]:
    """
    Parses triplets from the raw text output of the LLM.
    """
    text = raw_text.strip()
    results = []

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    json_objects = re.finditer(r"\{.*?\}", text, re.DOTALL)

    for match in json_objects:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                results.append(obj)
        except:
            continue

    if not results:
        json_lists = re.finditer(r"\[.*?\]", text, re.DOTALL)
        for match in json_lists:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, list):
                    results.append(obj)
            except:
                continue

    return results


def normalize_to_triplet(item: Any, doc_id: Any) -> Optional[Triplet]:
    """
    Data cleaning function: Uniformly convert various formats of item (dict or list) into Triplet objects.
    """
    s, p, o = None, None, None
    if isinstance(item, dict):
        s = item.get("subject") or item.get("s") or item.get("head")
        p = item.get("predicate") or item.get("p") or item.get("relation")
        o = item.get("object") or item.get("o") or item.get("tail")

    elif isinstance(item, list) and len(item) >= 3:
        s, p, o = item[0], item[1], item[2]

    if s and p and o:
        s_str = str(s).strip()
        p_str = str(p).strip()
        o_str = str(o).strip()

        if s_str and p_str and o_str:
            return Triplet(
                subject=s_str, predicate=p_str, object=o_str, document_id=doc_id
            )
    return None


async def process_item(
    item_id: Any,
    item_contents: str,
    generate_agent: ExtractorAgent,
    context_length: int,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    dataset_name: str,
) -> Set[Triplet]:
    print(f"Processing Item ID: {item_id} ...")
    triples: Set[Triplet] = set()
    content = item_contents.strip()
    if not content:
        print(f"Warning: Item ID {item_id} has empty content. Skipping.")
        return triples

    content_tokens = count_tokens(content)
    text_chunks = [content]

    safe_context_limit = context_length * 0.85
    if content_tokens > safe_context_limit:
        try:
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            if chunks:
                text_chunks = chunks
        except Exception as e:
            print(f"Chunking error on {item_id}: {e}")

    for i, chunk in enumerate(text_chunks):
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens == 0:
            continue

        try:
            prompt = generate_prompt(
                strategy=strategy, dataset_name=dataset_name, text_chunk=chunk
            )
        except Exception as e:
            print(f"❌ Error generating prompt: {e}")
            continue

        agent_role = "You are a Knowledge Graph construction expert."
        raw_response_text = await generate_agent.execute(prompt, agent_role)
        if raw_response_text:
            raw_items = parse_triplets_from_text(raw_response_text)
            count_before = len(triples)
            for item in raw_items:
                triplet_obj = normalize_to_triplet(item, item_id)
                if triplet_obj:
                    triples.add(triplet_obj)

            added = len(triples) - count_before
            if added > 0:
                print(f"   Chunk {i+1}: Successfully extracted {added} triplets.")
            else:
                print(
                    f"   ⚠️ Chunk {i+1}: No triplets found. Raw: {raw_response_text[:50]}..."
                )
                pass

        else:
            print(f"LLM returned no response or an error occurred (chunk {i+1}).")
        await asyncio.sleep(0.5)

    print(f"Extracted a total of {len(triples)} unique triplets.")
    return triples


async def extract(
    initial_items_to_process: List[Dict[str, Any]],
    output_json_file: str,
    strategy: str,
    dataset_name: str,
):
    """
    Extracts triplets from a given list of data items and saves them to the specified output file.
    This function is refactored to be a reusable processing module.
    """

    previous_results = {}
    completed_ids = set()

    # Check for and load existing results
    if os.path.exists(output_json_file):
        print(f"Found existing results file: {output_json_file}. Attempting to resume.")
        try:
            with open(output_json_file, "r", encoding="utf-8") as f:
                previous_results = json.load(f)
                # Extract IDs of already completed items. JSON keys are strings.
                completed_ids = set(previous_results.get("results_by_id", {}).keys())
                print(f"Found {len(completed_ids)} previously completed items.")
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"Warning: Could not read existing results file properly ({e}). Starting from scratch."
            )
            previous_results = {}
            completed_ids = set()

    # Filter out completed tasks
    items_to_process = [
        item
        for item in initial_items_to_process
        if str(item["id"]) not in completed_ids
    ]
    total_initial_items = len(initial_items_to_process)
    print(
        f"Skipping {len(completed_ids)} completed items. New items to process: {len(items_to_process)}."
    )

    if not items_to_process:
        print("All items have already been processed successfully. Nothing to do.")
        return

    # --- Configuration ---
    try:
        context_length = int(os.getenv("CONTEXT_LENGTH", 32768))
        default_chunk_size = min(8000, int(context_length * 0.25))
        chunk_size = int(os.getenv("CHUNK_SIZE", default_chunk_size))
        default_chunk_overlap = min(500, int(chunk_size * 0.1))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", default_chunk_overlap))

        print("--- Configuration ---")
        print(f"Chunk Size (Target Input): {chunk_size}")
        print(f"Chunk Overlap: {chunk_overlap}")
        print("--------------------")
    except ValueError as e:
        print(
            f"Error: Invalid configuration parameter type: {e}. Please check environment variables."
        )
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # --- Initialize ExtractorAgent ---
    try:
        extractor_config = AgentConfig(
            name="ExtractorAgent",
            model_name=MODEL,
            temperature=0.3,
            max_tokens=2048,
        )
        extractor_agent = ExtractorAgent(config=extractor_config)
    except Exception as e:
        print(f"Error initializing ExtractorAgent: {e}")
        return

    # --- Process Items ---
    all_results: Dict[Any, Set[Triplet]] = {}
    print(f"Processing {len(items_to_process)} items...")

    concurrency_limit = int(os.getenv("CONCURRENCY_LIMIT", 5))
    semaphore = asyncio.Semaphore(concurrency_limit)
    print(f"Concurrency limit set to: {concurrency_limit}")

    async def process_item_wrapper(item_data: Dict[str, Any]):
        """Wrapper to manage semaphore and pass arguments for a single JSONL item."""
        item_id = item_data["id"]
        item_contents = item_data["contents"]
        async with semaphore:
            result_triplets = await process_item(
                item_id,
                item_contents,
                extractor_agent,
                context_length,
                chunk_size,
                chunk_overlap,
                strategy=strategy,
                dataset_name=dataset_name,
            )
            return item_id, result_triplets

    # Create all processing tasks based on the input list
    tasks = [process_item_wrapper(item) for item in items_to_process]

    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result_or_exc in enumerate(task_results):
        processed_item_id = items_to_process[i]["id"]
        if isinstance(result_or_exc, Exception):
            print(f"Task for item ID {processed_item_id} failed: {result_or_exc}")
            all_results[processed_item_id] = set()
        else:
            returned_id, item_triplets_result = result_or_exc
            if returned_id != processed_item_id:
                print(
                    f"Warning: Task returned ID ({returned_id}) does not match expected ID ({processed_item_id})."
                )
            all_results[processed_item_id] = item_triplets_result

    if hasattr(extractor_agent, "close"):
        await extractor_agent.close()

    # --- Summarize and save results ---
    print("\n\n--- Overall Summary ---")
    final_results_by_id = previous_results.get("results_by_id", {})
    newly_processed_results = {
        str(item_id): [t.model_dump() for t in triplets]
        for item_id, triplets in all_results.items()
        if triplets and not isinstance(triplets, Exception)
    }
    final_results_by_id.update(newly_processed_results)
    total_successful_items = len(final_results_by_id)
    total_triplets_extracted = sum(len(trips) for trips in final_results_by_id.values())

    print(f"Total items in source: {total_initial_items}.")
    print(
        f"Total successfully processed items (across all runs): {total_successful_items}"
    )
    print(
        f"Total unique triplets extracted (across all runs): {total_triplets_extracted}"
    )
    output_data = {
        "config": {
            "model_name": extractor_config.model_name,
            "temperature": extractor_config.temperature,
            "max_tokens": extractor_config.max_tokens,
            "context_length": context_length,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "concurrency_limit": concurrency_limit,
        },
        "summary": {
            "items_attempted": total_initial_items,
            "items_successful": total_successful_items,
            "total_triplets_extracted": total_triplets_extracted,
        },
        "results_by_id": final_results_by_id,
    }

    try:
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"All results have been saved to: {output_json_file}")
    except Exception as e:
        print(f"Error saving results to {output_json_file}: {e}")


async def verify(
    input_json_file: str,
    output_json_file: str,
    dataset_config: dict,
    verify_mode=True,  # True or False
):
    """Fact verification section""" 
    try:
        print(f"Starting to read extraction results: {input_json_file}...")
        with open(input_json_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        results_by_id = input_data.get("results_by_id", {})
        total_items = len(results_by_id)
        print(f"Total items read: {total_items}")

        if not verify_mode:
            print("Ablation study: no_verify mode. Injecting default passing values (1) for all triplets...")
            for item_id, triplets in results_by_id.items():
                for triplet in triplets:
                    triplet["verification"] = {"votes": [1, 1, 1], "final_result": 1}
        else:
            print("Starting to load the original dataset for verification context...")
            raw_data_items = dataset_config["loader"](dataset_config["input_path"])
            id_to_text_map = {
                str(item["id"]): item["contents"] for item in raw_data_items
            }
            print(f"Successfully loaded {len(id_to_text_map)} original texts for对照 verification.")

            verify_config = AgentConfig(
                model_name=MODEL,
                name="VerifyFactLLMAgent",
                max_tokens=1024,
                temperature=0,
            )
            verify_fact_LLM_agent = VerifyFactLLMAgent(verify_config, enabled=True)
            batch_size = 5

            for file_idx, (item_id, triplets) in enumerate(results_by_id.items(), 1):
                original_text = id_to_text_map.get(str(item_id))
                if not original_text:
                    print(f"Warning: Original text not found for ID {item_id}, skipping verification.")
                    continue

                if not triplets:
                    continue

                print(f"Processing file {file_idx}")
                for i in range(0, len(triplets), batch_size):
                    batch_triplets = triplets[i : i + batch_size]
                    verification_results = await verify_fact_LLM_agent.execute(
                        triplets=batch_triplets, original_text=original_text
                    )

                    if verification_results and len(verification_results) == 3:
                        for j, triplet in enumerate(batch_triplets):
                            votes = [results[j] for results in verification_results]
                            final_vote = 1 if sum(votes) >= 2 else 0
                            triplet["verification"] = {
                                "votes": votes,
                                "final_result": final_vote,
                            }

        output_data = {
            "config": input_data.get("config", {}),
            "summary": input_data.get("summary", {}),
            "results_by_id": results_by_id,
        }

        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Verification results saved to: {output_json_file}")

    except Exception as e:
        import traceback

        print(f"An error occurred during processing: {e}")
        traceback.print_exc()


async def describe(
    input_file: str, output_file: str, enable_llm_description: bool = True
):
    print("Starting to read JSON file...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found. Please ensure the previous step has been run successfully.")
        return
    except Exception as e:
        print(f"An error occurred while reading or parsing the file {input_file}: {e}")
        return

    print("Starting to process triplets...")
    results_by_id = data.get("results_by_id", {})

    if enable_llm_description:
        entity_config = AgentConfig(
            model_name=MODEL,
            name="EntityDescriptionAgent",
            max_tokens=1024,
            temperature=0,
        )
        relation_config = AgentConfig(
            model_name=MODEL,
            name="RelationDescriptionAgent",
            max_tokens=1024,
            temperature=0,
        )
        entity_description_agent = EntityDescriptionAgent(entity_config)
        relation_description_agent = RelationDescriptionAgent(relation_config)
        semaphore = asyncio.Semaphore(2)
    else:
        semaphore = None

    async def execute_with_retry(
        agent, context: str, prompt: str, max_retries: int = 3
    ):
        """
        Executes an agent with a retry mechanism. If the agent call fails, it will retry up to max_retries times with exponential backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return await agent.execute(context, prompt)
            except Exception as e:
                error_msg = str(e)

                if attempt == max_retries:
                    print(
                        f"  [Failed] Failed after {attempt}/{max_retries} retries: {error_msg[:100]}..."
                    )
                    raise e
                else:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    print(
                        f"  [Warning] Request failed (Attempt {attempt}/{max_retries}), retrying in {wait_time:.2f} seconds... Error: {error_msg[:50]}..."
                    )
                    await asyncio.sleep(wait_time)

    async def process_triplet(triplet):
        """
        Processes a single triplet: checks format, optionally enriches with LLM descriptions, and returns the enriched triplet.
        """
        if not all(k in triplet for k in ["subject", "predicate", "object"]):
            print(f"Warning: Skipping an incorrectly formatted triplet: {triplet}")
            return None

        if not enable_llm_description:
            triplet["descriptions"] = {
                "head_entity": "",
                "tail_entity": "",
                "relation": "",
            }
            return triplet

        async with semaphore:
            try:
                context = f"Triples: {triplet['subject']} - {triplet['predicate']} - {triplet['object']}"
                head_entity_desc, tail_entity_desc, relation_desc = (
                    await asyncio.gather(
                        execute_with_retry(
                            entity_description_agent,
                            context,
                            f"entity: {triplet['subject']}",
                        ),
                        execute_with_retry(
                            entity_description_agent,
                            context,
                            f"entity: {triplet['object']}",
                        ),
                        execute_with_retry(
                            relation_description_agent,
                            context,
                            f"relation: {triplet['predicate']}",
                        ),
                    )
                )
                triplet["descriptions"] = {
                    "head_entity": head_entity_desc,
                    "tail_entity": tail_entity_desc,
                    "relation": relation_desc,
                }

                await asyncio.sleep(1)
                return triplet

            except Exception as e:
                print(f"An error occurred while processing triplet {triplet.get('subject')}: {e}")
                return None

    print("\n--- Starting to process data from all source IDs ---")
    final_enriched_results = {}

    for item_id, triplets in results_by_id.items():
        verified_triplets_for_id = [
            t for t in triplets if t.get("verification", {}).get("final_result") == 1
        ]
        if not verified_triplets_for_id:
            print(f"ID {item_id} has no verified triplets, skipping.")
            final_enriched_results[item_id] = []
            continue

        tasks = [process_triplet(triplet) for triplet in verified_triplets_for_id]
        enriched_results_for_id = await asyncio.gather(*tasks)

        successful_results = [r for r in enriched_results_for_id if r is not None]
        final_enriched_results[item_id] = successful_results

        print(
            f"[Complete] Source ID: {item_id}. Successfully enriched {len(successful_results)} triplets."
        )
        await asyncio.sleep(2)

    print("\n--- Saving final results ---")
    total_enriched_triplets = sum(len(ts) for ts in final_enriched_results.values())
    output_data = {
        "config": data.get("config", {}),
        "summary": {
            "items_processed": len(results_by_id),
            "items_with_enriched_triplets": sum(
                1 for ts in final_enriched_results.values() if ts
            ),
            "total_enriched_triplets": total_enriched_triplets,
            "original_summary": data.get("summary", {}),
        },
        "results_by_id": final_enriched_results,
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nAll enriched triplets have been saved to: {output_file}")
        print(f"Total successfully processed triplets: {total_enriched_triplets}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")


async def insert(input_file: str, output_file: str):
    print(f"Starting to read JSON file: {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data_from_file = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Input file {input_file} not found. Please ensure the previous step 'describe' has been run successfully."
        )
        return
    except Exception as e:
        print(f"Error occurred while reading or parsing file {input_file}: {e}")
        return

    results_by_id = data_from_file.get("results_by_id", {})
    all_triplets = []
    for triplets_list in results_by_id.values():
        all_triplets.extend(triplets_list)

    if not all_triplets:
        print("No triplets to process. This might be due to the previous step not generating valid output.")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"Created empty output file: {output_file}")
        except Exception as e:
            print(f"Error occurred while creating empty output file: {e}")
        return

    print(f"Selected {len(all_triplets)} triplets for ID assignment.")

    entity_config = AgentConfig(
        model_name=MODEL,
        name="EntityAgent",
        max_tokens=128,
        temperature=0,
    )
    relation_config = AgentConfig(
        model_name=MODEL,
        name="RelationAgent",
        max_tokens=128,
        temperature=0,
    )
    entity_agent = EntityAgent(entity_config)
    relation_agent = RelationAgent(relation_config)

    print("Starting to check and load Milvus Collections...")
    try:
        target_collections = ["entity", "relation"]
        for col_name in target_collections:
            if utility.has_collection(col_name):
                print(f"  - Loading '{col_name}'...")
                Collection(col_name).load()
            else:
                print(f"Warning: Collection '{col_name}' does not exist, skipping load.")
        print("Milvus Collections loaded successfully.")

    except Exception as e:
        print(f"Error occurred while loading Collection (might not affect subsequent steps if run for the first time): {e}")

    result_with_entity_ids = await entity_agent.execute(all_triplets)
    result = await relation_agent.execute(result_with_entity_ids)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results have been saved to: {output_file}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")


async def semantic(input_file: str, output_file: str):
    semantic_config = AgentConfig(
        model_name=MODEL,
        name="SemanticAgent",
        max_tokens=128,
        temperature=0,
    )
    semantic_agent = SemanticAgent(semantic_config)

    print(f"Starting to read JSON file: {input_file}...")
    all_triplets_from_file = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            all_triplets_from_file = json.load(f)
        print(f"Successfully read {len(all_triplets_from_file)} triplets.")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found. Please ensure the file path is correct.")
        await semantic_agent.close_client()
        return
    except json.JSONDecodeError:
        print(f"Error: Input file {input_file} is not valid JSON.")
        await semantic_agent.close_client()
        return

    output_triplets = []
    processed_relation_cache = {}
    llm_call_count = 0
    cache_hit_count = 0

    temp_output_file = output_file + ".temp"
    last_save_count = 0
    save_interval = 100

    print("\nStarting to process triplets and determine relation characteristics...")
    try:
        for i, triplet_data in enumerate(all_triplets_from_file):
            predicate_id = triplet_data.get("predicate_id")
            if predicate_id is None:
                output_triplets.append(triplet_data)
                continue

            properties_list = None
            is_new_query = False
            if predicate_id in processed_relation_cache:
                properties_list = processed_relation_cache[predicate_id]
                cache_hit_count += 1
            else:
                is_new_query = True
                relation_description = triplet_data.get("descriptions", {}).get(
                    "relation"
                )
                if not relation_description:
                    print(
                        f"  Warning: predicate_id: {predicate_id} (original index: {i+1}) is missing relation description, using default properties."
                    )
                    properties_list = [0, 0, 0, 0, 0, 0, 0]
                else:
                    llm_call_count += 1
                    properties_list = await semantic_agent.execute(relation_description)
                    if properties_list is None:
                        print(
                            f"  Warning: predicate_id: {predicate_id} (index: {i+1}) semantic analysis failed, using default properties [0,0,0,0,0,0,0]."
                        )
                        properties_list = [0, 0, 0, 0, 0, 0, 0]

                processed_relation_cache[predicate_id] = properties_list

            output_triplet = triplet_data.copy()
            output_triplet["relation_properties_list"] = properties_list
            output_triplets.append(output_triplet)

            if (i + 1) % 50 == 0 or (i + 1) == len(all_triplets_from_file):
                print(
                    f"  Progress: {i + 1}/{len(all_triplets_from_file)} | LLM Calls: {llm_call_count} | Cache Hits: {cache_hit_count}"
                )

            if len(output_triplets) - last_save_count >= save_interval:
                print(f"\nPerforming periodic save, {len(output_triplets)} items processed...")
                with open(temp_output_file, "w", encoding="utf-8") as f:
                    json.dump(output_triplets, f, ensure_ascii=False, indent=2)
                print(f"Periodic save successful: {temp_output_file}")
                last_save_count = len(output_triplets)

            if is_new_query:
                await asyncio.sleep(0.2)

        print(f"\nProcessing complete. Starting to write results to file: {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_triplets, f, ensure_ascii=False, indent=2)
        print(f"Successfully wrote {len(output_triplets)} processed items to {output_file}")
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        print("Attempting to save progress from temporary file to final output file...")
        if os.path.exists(temp_output_file):
            try:
                os.rename(temp_output_file, output_file)
                print(f"Successfully renamed temporary file to: {output_file}")
            except OSError as oe:
                print(f"Failed to rename temporary file: {oe}")
    finally:
        await semantic_agent.close_client()
        print("\n--- Semantic Analysis Finished ---")


async def clear_sql(pool_or_config, table_name, force=False):
    """
    Clears all data from the specified SQL table using TRUNCATE.
    """
    if not force:
        confirm = input(f"Confirm deletion of {table_name}? [y/n]: ")
        if confirm != "y":
            return

    print(f" [System Auto] Currently truncating table '{table_name}' ...")

    conn = await aiomysql.connect(**pool_or_config)
    async with conn.cursor() as cursor:
        await cursor.execute(f"TRUNCATE TABLE {table_name};")
    conn.close()

    print(f"Table '{table_name}' has been truncated.")


def clear_milvus_collections(target_collections):
    """
    Clears all data from the specified Milvus collections by deleting all entities.
    """
    print(f"\n🌊 [Milvus] Preparing to clear the following collections: {target_collections}")

    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    if not uri or not token:
        print("Error: ZILLIZ_URI or ZILLIZ_TOKEN not found in .env.")
        return

    try:
        if not connections.has_connection("default"):
            connections.connect("default", uri=uri, token=token)

        for name in target_collections:
            if not utility.has_collection(name):
                print(f"⚠️ Collection '{name}' does not exist, skipping.")
                continue

            collection = Collection(name)
            pk_name = None
            for field in collection.schema.fields:
                if field.is_primary:
                    pk_name = field.name
                    break

            if not pk_name:
                print(f"'{name}' does not have a primary key defined, cannot perform deletion.")
                continue

            expr = f"{pk_name} > 0"
            collection.load()
            current_count = collection.num_entities

            if current_count == 0:
                print(f"'{name}' is already empty, no need to clear.")
                collection.release()
                continue

            print(f"   Clearing '{name}'...")
            print(f"   - Primary key name: {pk_name}")
            print(f"   - Delete condition: {expr}")
            print(f"   - Estimated deletions: {current_count} items")

            collection.delete(expr)
            collection.release()
            print(f"Cleanup command for '{name}' has been sent.")

    except Exception as e:
        print(f"An error occurred during Milvus operation: {e}")


async def create_triples_table_if_not_exists_aiomysql(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS triples (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        subject VARCHAR(767) NOT NULL,
                        predicate VARCHAR(767) NOT NULL,
                        object TEXT NOT NULL, 
                        subject_id BIGINT NOT NULL,
                        predicate_id BIGINT NOT NULL,
                        object_id BIGINT NOT NULL,
                        semantic_source VARCHAR(100) DEFAULT 'original',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_triples_ids (subject_id, predicate_id, object_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
                )
                print("Table 'triples' has been ensured to exist or has been created (aiomysql).")


                indexes = {
                    "idx_triples_subject_id": "CREATE INDEX idx_triples_subject_id ON triples(subject_id);",
                    "idx_triples_predicate_id": "CREATE INDEX idx_triples_predicate_id ON triples(predicate_id);",
                    "idx_triples_object_id": "CREATE INDEX idx_triples_object_id ON triples(object_id);",
                    "idx_triples_pred_obj_ids": "CREATE INDEX idx_triples_pred_obj_ids ON triples(predicate_id, object_id);",
                }
                for index_name, index_sql in indexes.items():
                    await cursor.execute(
                        f"SHOW INDEX FROM triples WHERE Key_name = '{index_name}'"
                    )
                    if not await cursor.fetchone():
                        try:
                            await cursor.execute(index_sql)
                            print(f"Index '{index_name}' has been created (aiomysql).")
                        except aiomysql.MySQLError as idx_err:
                            if idx_err.args[0] == 1071:
                                print(
                                    f"Warning: Failed to create index '{index_name}', key is too long. SQL: {index_sql}. Error: {idx_err}"
                                )
                            else:
                                raise
                    else:
                        print(f"Index '{index_name}' already exists (aiomysql).")

                print("Database table and indexes initialized (aiomysql).")

            except aiomysql.MySQLError as e:
                print(f"An error occurred during database initialization: {e}")
                raise


async def check_triple_exists_aiomysql(cursor, triple_info_dict):
    query = "SELECT 1 FROM triples WHERE subject_id = %s AND predicate_id = %s AND object_id = %s LIMIT 1"
    await cursor.execute(
        query,
        (
            triple_info_dict["subject_id"],
            triple_info_dict["predicate_id"],
            triple_info_dict["object_id"],
        ),
    )
    return await cursor.fetchone() is not None


async def insert_triple_to_db_aiomysql(cursor, triple_info_dict, source="original"):
    if await check_triple_exists_aiomysql(
        cursor, triple_info_dict
    ):  # await aync function
        return False

    query = """
        INSERT INTO triples 
        (subject, predicate, object, subject_id, predicate_id, object_id, semantic_source) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        await cursor.execute(
            query,
            (
                triple_info_dict["subject"],
                triple_info_dict["predicate"],
                triple_info_dict["object"],
                triple_info_dict["subject_id"],
                triple_info_dict["predicate_id"],
                triple_info_dict["object_id"],
                source,
            ),
        )
        return True

    except aiomysql.MySQLError as e:
        if e.args[0] == 1062:
            return False
        print(
            f"  Failed to insert triple: {e} (S_ID:{triple_info_dict['subject_id']}, P_ID:{triple_info_dict['predicate_id']}, O_ID:{triple_info_dict['object_id']})"
        )
        raise


async def triples_embedding_insert(data_to_process: list[dict], collection_name: str):
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_token = os.getenv("ZILLIZ_TOKEN")
    milvus_client = None
    try:
        milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
        print("Successfully connected to Zilliz Cloud.")

        print(f"Validating Milvus collection '{collection_name}'...")
        if not milvus_client.has_collection(collection_name):
            error_msg = (
                f"Error: The Milvus collection '{collection_name}' for triple embeddings does not exist."
            )
            print(error_msg)
            raise ValueError(error_msg)

        print(f"Collection '{collection_name}' exists.")

        if not data_to_process:
            print("No text data needs to be processed for embeddings.")
            return

        texts_for_simple_embedding = [item["text_simple"] for item in data_to_process]
        texts_for_rich_embedding = [item["text_rich"] for item in data_to_process]

        print(
            f"Generating simple and rich embedding vectors for {len(data_to_process)} triples..."
        )
        simple_embeddings, rich_embeddings = await asyncio.gather(
            get_embeddings(texts_for_simple_embedding),
            get_embeddings(texts_for_rich_embedding),
        )

        data_to_insert = []
        failed_count = 0
        for i in range(len(data_to_process)):
            if simple_embeddings[i] is not None and rich_embeddings[i] is not None:
                original_item = data_to_process[i]

                data_to_insert.append(
                    {
                        "embedding_simple": simple_embeddings[i],
                        "embedding_rich": rich_embeddings[i],
                        "text_simple": original_item["text_simple"],
                        "text_rich": original_item["text_rich"],
                        "document_id": original_item.get("document_id", -1),
                    }
                )
            else:
                failed_count += 1
                print(
                    f"Warning: Failed to retrieve embedding vectors for triple '{data_to_process[i]['text_simple']}', skipping this record."
                )

        if not data_to_insert:
            print("No valid embedding pairs retrieved, no data to insert into Milvus.")
            return

        print(
            f"Preparing to insert {len(data_to_insert)} valid records into Milvus. ({failed_count} failed)"
        )

        insert_result = milvus_client.insert(
            collection_name=collection_name, data=data_to_insert
        )
        print(f"Successfully inserted: {insert_result['insert_count']}")

    except Exception as e:
        print(f"Critical error occurred during triple embedding insertion: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if milvus_client:
            milvus_client.close()
            print("\nMilvus connection has been closed.")


async def save_list_to_file(data_list, filename):
    """
    Saves a list of data to a file, one item per line. Tries to use asynchronous file writing with aiofiles for better performance, but falls back to synchronous writing if aiofiles is not available.
    """
    print(f"\nPreparing to write {len(data_list)} records to file '{filename}'...")
    try:
        import aiofiles
    except ImportError:
        print(
            "Warning: 'aiofiles' is not installed, will use synchronous way to write to file. Consider running 'pip install aiofiles'."
        )
        with open(filename, "w", encoding="utf-8") as f:
            if not data_list:
                f.write("Is Empty!\n")
            else:
                for item in data_list:
                    f.write(str(item) + "\n")
        print(f"Synchronous writing completed. Total {len(data_list)} records.")
        return

    async with aiofiles.open(filename, "w", encoding="utf-8") as f:
        if not data_list:
            await f.write("Is Empty!\n")
        else:
            await f.writelines(f"{item}\n" for item in data_list)
    print(f"Asynchronous writing completed. Total {len(data_list)} records.")


async def sqlinsert_async(input_data_list, db_pool):
    """
    This function processes a list of triple information dictionaries, checks for consistency based on semantic properties, and inserts valid triples into a SQL database using aiomysql. It also prepares data for embedding insertion into Milvus and provides detailed logging throughout the process.
    """
    if not input_data_list:
        print("Input data list is empty, no content to process.")
        return {"total_processed": 0, "summary": []}

    print("Creating mappings from entity and relation IDs to descriptions...")
    entity_id_to_desc = {}
    relation_id_to_desc = {}
    for item in input_data_list:
        s_id, o_id = item.get("subject_id"), item.get("object_id")
        s_desc = item.get("descriptions", {}).get("head_entity")
        o_desc = item.get("descriptions", {}).get("tail_entity")
        if s_id and s_desc:
            entity_id_to_desc[s_id] = s_desc
        if o_id and o_desc:
            entity_id_to_desc[o_id] = o_desc

        p_id = item.get("predicate_id")
        p_desc = item.get("descriptions", {}).get("relation")
        if p_id and p_desc:
            relation_id_to_desc[p_id] = p_desc
    print("Mappings created successfully.")

    def format_texts(s, p, o, s_id, o_id, p_id):
        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = re.sub(r"Target word:|Output:|Entity:|Relation:", "", text, flags=re.IGNORECASE)
            return " ".join(text.split()).strip()

        s_clean, p_clean, o_clean = clean_text(s), clean_text(p), clean_text(o)
        text_simple = f"{s_clean}-{p_clean}-{o_clean}"
        s_desc = entity_id_to_desc.get(s_id, s_clean)
        o_desc = entity_id_to_desc.get(o_id, o_clean)
        p_desc = relation_id_to_desc.get(p_id, p_clean)

        text_rich = (
            f"Triplet: {s_clean} - {p_clean} - {o_clean}. "
            f"Subject Description: {clean_text(s_desc)}. "
            f"Object Description: {clean_text(o_desc)}. "
            f"Relation Description: {clean_text(p_desc)}."
        )

        return {"text_simple": text_simple, "text_rich": text_rich}

    triples_for_embedding_insertion = []
    processed_results_summary = []
    total_items_in_list = len(input_data_list)
    items_processed_count = 0

    print(f"\nStarting asynchronous processing of {total_items_in_list} input records and inserting into database...")
    for item_index, current_triple_info in enumerate(input_data_list):
        items_processed_count += 1
        subject_text = current_triple_info.get("subject")
        predicate_text = current_triple_info.get("predicate")
        object_text = current_triple_info.get("object")
        subject_id = current_triple_info.get("subject_id")
        predicate_id = current_triple_info.get("predicate_id")
        object_id = current_triple_info.get("object_id")
        semantic_01_list = current_triple_info.get("relation_properties_list")
        document_id = current_triple_info.get("document_id")
        triple_display = f"S:'{subject_text}'({subject_id}), P:'{predicate_text}'({predicate_id}), O:'{object_text}'({object_id})"

        if not all(
            [
                subject_text is not None,
                predicate_text is not None,
                object_text is not None,
                subject_id is not None,
                predicate_id is not None,
                object_id is not None,
                isinstance(semantic_01_list, list),
                len(semantic_01_list) == 7,
            ]
        ):
            print(
                f"Warning (Item {item_index + 1}/{total_items_in_list}): Skipping invalid data item."
            )
            processed_results_summary.append(
                {"triple": triple_display, "status": "skipped_invalid_data"}
            )
            continue

        try:
            numbers = list(map(int, semantic_01_list))
            (
                functional,
                inverse_functional,
                transitive,
                symmetric,
                asymmetric,
                reflexive,
                irreflexive,
            ) = numbers
        except ValueError:
            print(
                f"Warning (Item {item_index + 1}/{total_items_in_list}): Skipping {triple_display}, semantic_01_list contains non-integer values."
            )
            processed_results_summary.append(
                {"triple": triple_display, "status": "skipped_invalid_semantic_list"}
            )
            continue

        print(
            f"\n--- Processing (Item {item_index + 1}/{total_items_in_list}): {triple_display} ---"
        )

        is_consistent = True
        violation_reason = ""

        async with db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    if functional == 1:
                        query_func = "SELECT object_id FROM triples WHERE subject_id = %s AND predicate_id = %s AND object_id != %s LIMIT 1"
                        await cursor.execute(
                            query_func, (subject_id, predicate_id, object_id)
                        )
                        if await cursor.fetchone():
                            is_consistent = False
                            violation_reason = "Functional property violation (ID)"

                    if is_consistent and inverse_functional == 1:
                        query_inv_func = "SELECT subject_id FROM triples WHERE predicate_id = %s AND object_id = %s AND subject_id != %s LIMIT 1"
                        await cursor.execute(
                            query_inv_func, (predicate_id, object_id, subject_id)
                        )
                        if await cursor.fetchone():
                            is_consistent = False
                            violation_reason = (
                                "Inverse Functional property violation (ID)"
                            )

                    if is_consistent and asymmetric == 1:
                        asymmetric_check_dict = {
                            "subject_id": object_id,
                            "predicate_id": predicate_id,
                            "object_id": subject_id,
                        }
                        if await check_triple_exists_aiomysql(
                            cursor, asymmetric_check_dict
                        ):
                            is_consistent = False
                            violation_reason = "Asymmetric property violation (ID)"

                    if is_consistent and irreflexive == 1:
                        if subject_id == object_id:
                            is_consistent = False
                            violation_reason = (
                                "Irreflexive property violation (S_id == O_id)"
                            )

                    if is_consistent:
                        await conn.begin()
                        main_inserted = await insert_triple_to_db_aiomysql(
                            cursor, current_triple_info, source="original"
                        )

                        if main_inserted:
                            processed_results_summary.append(
                                {
                                    "triple": triple_display,
                                    "status": "inserted_original",
                                }
                            )
                            texts = format_texts(
                                subject_text,
                                predicate_text,
                                object_text,
                                subject_id,
                                object_id,
                                predicate_id,
                            )
                            triples_for_embedding_insertion.append(
                                {**texts, "document_id": document_id}
                            )
                        else:
                            processed_results_summary.append(
                                {
                                    "triple": triple_display,
                                    "status": "existed_original (ID)",
                                }
                            )

                        if transitive == 1:
                            q_prev = "SELECT subject_id, subject FROM triples WHERE object_id = %s AND predicate_id = %s"
                            await cursor.execute(q_prev, (subject_id, predicate_id))
                            async for prev_s_id, prev_s_text in cursor:
                                transitive_triple1_info = {
                                    "subject": prev_s_text,
                                    "predicate": predicate_text,
                                    "object": object_text,
                                    "subject_id": prev_s_id,
                                    "predicate_id": predicate_id,
                                    "object_id": object_id,
                                }
                                if await insert_triple_to_db_aiomysql(
                                    cursor, transitive_triple1_info, source="transitive"
                                ):
                                    texts = format_texts(
                                        prev_s_text,
                                        predicate_text,
                                        object_text,
                                        prev_s_id,
                                        object_id,
                                        predicate_id,
                                    )
                                    triples_for_embedding_insertion.append(
                                        {**texts, "document_id": document_id}
                                    )
                                    processed_results_summary.append(
                                        {
                                            "triple": f"S:'{prev_s_text}'({prev_s_id}), P:'{predicate_text}'({predicate_id}), O:'{object_text}'({object_id})",
                                            "status": "inserted_transitive",
                                        }
                                    )

                            q_next = "SELECT object_id, object FROM triples WHERE subject_id = %s AND predicate_id = %s"
                            await cursor.execute(q_next, (object_id, predicate_id))
                            async for next_o_id, next_o_text in cursor:
                                transitive_triple2_info = {
                                    "subject": subject_text,
                                    "predicate": predicate_text,
                                    "object": next_o_text,
                                    "subject_id": subject_id,
                                    "predicate_id": predicate_id,
                                    "object_id": next_o_id,
                                }
                                if await insert_triple_to_db_aiomysql(
                                    cursor, transitive_triple2_info, source="transitive"
                                ):
                                    texts = format_texts(
                                        subject_text,
                                        predicate_text,
                                        next_o_text,
                                        subject_id,
                                        next_o_id,
                                        predicate_id,
                                    )
                                    triples_for_embedding_insertion.append(
                                        {**texts, "document_id": document_id}
                                    )
                                    processed_results_summary.append(
                                        {
                                            "triple": f"S:'{subject_text}'({subject_id}), P:'{predicate_text}'({predicate_id}), O:'{next_o_text}'({next_o_id})",
                                            "status": "inserted_transitive",
                                        }
                                    )

                        if reflexive == 1:
                            reflexive_s_info = {
                                "subject": subject_text,
                                "predicate": predicate_text,
                                "object": subject_text,
                                "subject_id": subject_id,
                                "predicate_id": predicate_id,
                                "object_id": subject_id,
                            }
                            if await insert_triple_to_db_aiomysql(
                                cursor, reflexive_s_info, source="reflexive"
                            ):
                                texts = format_texts(
                                    subject_text,
                                    predicate_text,
                                    subject_text,
                                    subject_id,
                                    subject_id,
                                    predicate_id,
                                )
                                triples_for_embedding_insertion.append(
                                    {**texts, "document_id": document_id}
                                )
                                processed_results_summary.append(
                                    {
                                        "triple": f"S:'{subject_text}'({subject_id}), P:'{predicate_text}'({predicate_id}), O:'{subject_text}'({subject_id})",
                                        "status": "inserted_reflexive",
                                    }
                                )
                            if object_id != subject_id:
                                reflexive_o_info = {
                                    "subject": object_text,
                                    "predicate": predicate_text,
                                    "object": object_text,
                                    "subject_id": object_id,
                                    "predicate_id": predicate_id,
                                    "object_id": object_id,
                                }
                                if await insert_triple_to_db_aiomysql(
                                    cursor, reflexive_o_info, source="reflexive"
                                ):
                                    texts = format_texts(
                                        object_text,
                                        predicate_text,
                                        object_text,
                                        object_id,
                                        object_id,
                                        predicate_id,
                                    )
                                    triples_for_embedding_insertion.append(
                                        {**texts, "document_id": document_id}
                                    )
                                    processed_results_summary.append(
                                        {
                                            "triple": f"S:'{object_text}'({object_id}), P:'{predicate_text}'({predicate_id}), O:'{object_text}'({object_id})",
                                            "status": "inserted_reflexive",
                                        }
                                    )

                        if symmetric == 1:
                            symmetric_info = {
                                "subject": object_text,
                                "predicate": predicate_text,
                                "object": subject_text,
                                "subject_id": object_id,
                                "predicate_id": predicate_id,
                                "object_id": subject_id,
                            }
                            if await insert_triple_to_db_aiomysql(
                                cursor, symmetric_info, source="symmetric"
                            ):
                                texts = format_texts(
                                    object_text,
                                    predicate_text,
                                    subject_text,
                                    object_id,
                                    subject_id,
                                    predicate_id,
                                )
                                triples_for_embedding_insertion.append(
                                    {**texts, "document_id": document_id}
                                )
                                processed_results_summary.append(
                                    {
                                        "triple": f"S:'{object_text}'({object_id}), P:'{predicate_text}'({predicate_id}), O:'{subject_text}'({subject_id})",
                                        "status": "inserted_symmetric",
                                    }
                                )

                        await conn.commit()

                    else:
                        print(
                            f"  Semantic check failed for {triple_display}, skipping database write. Reason: {violation_reason}"
                        )
                        processed_results_summary.append(
                            {
                                "triple": triple_display,
                                "status": "skipped_semantic_violation",
                                "reason": violation_reason,
                            }
                        )

                except aiomysql.MySQLError as db_err_item:
                    print(f"Error occurred while processing {triple_display}: {db_err_item}")
                    try:
                        await conn.rollback()
                    except aiomysql.MySQLError as rb_err:
                        print(f"  Error occurred while rolling back transaction: {rb_err}")
                    processed_results_summary.append(
                        {
                            "triple": triple_display,
                            "status": "error_db_operation",
                            "reason": str(db_err_item),
                        }
                    )
                except Exception as general_err_item:
                    print(f"Error occurred while processing {triple_display}: {general_err_item}")
                    try:
                        await conn.rollback()
                    except aiomysql.MySQLError as rb_err:
                        print(f"  Error occurred while rolling back transaction: {rb_err}")
                    processed_results_summary.append(
                        {
                            "triple": triple_display,
                            "status": "error_unknown",
                            "reason": str(general_err_item),
                        }
                    )

    print(
        f"\n\n--- All {items_processed_count}/{total_items_in_list} records have been attempted for processing ---"
    )

    triples_collection_name = "triples"
    # await save_list_to_file(
    #     triples_for_embedding_insertion, "triples_for_embedding.txt"
    # )

    await triples_embedding_insert(
        triples_for_embedding_insertion, triples_collection_name
    )
    return {
        "total_processed": items_processed_count,
        "summary": processed_results_summary,
    }


async def sqlinsert(input_file_path: str, db_config):
    pool = None
    try:
        pool = await aiomysql.create_pool(minsize=1, maxsize=10, **db_config)
        print("Successfully created the aiomysql connection pool.")

        await create_triples_table_if_not_exists_aiomysql(pool)

        data_to_process = []
        if os.path.exists(input_file_path):
            print(f"Starting to read JSON file from '{input_file_path}'...")
            try:
                with open(input_file_path, "r", encoding="utf-8") as f:
                    data_to_process = json.load(f)
                print(f"Successfully read {len(data_to_process)} records.")
            except Exception as e:
                print(f"Error occurred while reading file '{input_file_path}': {e}")
                data_to_process = []
        else:
            print(f"Error: Input file '{input_file_path}' not found. Cannot proceed.")

        if data_to_process:
            results = await sqlinsert_async(data_to_process, pool)
            print("\n--- SQL Insertion Finished ---")
            print(f"Total items processed: {results.get('total_processed', 'N/A')}")
        else:
            print("No data available for processing, skipping database insertion.")

    except aiomysql.MySQLError as e_db_setup:
        print(f"Error occurred while setting up aiomysql connection pool: {e_db_setup}")
        raise
    except Exception as e_global:
        print(f"Global error occurred: {e_global}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        if pool:
            pool.close()
            await pool.wait_closed()
            print("aiomysql connection pool has been closed.")


def load_text_lines_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a plain text file, where each line is an item."""
    print(f"--- Loading data from {file_path} using Text Lines loader ---")
    items_to_process = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                content = line.strip()
                if content:
                    items_to_process.append({"id": line_num, "contents": content})
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        return []
    print(f"Successfully loaded {len(items_to_process)} items.")
    return items_to_process


async def main():
    """Main entry point for the KG extraction pipeline."""
    DATASET_CONFIGS = {
        "webnlg": {
            "loader": load_text_lines_data,
            "input_path": "datasets/webnlg/webnlg_test_texts.txt",
        },
        "CaRB": {
            "loader": load_text_lines_data,
            "input_path": "datasets/CaRB-master/data/test.txt",
        },
        "aida-conll": {
            "loader": load_text_lines_data,
            "input_path": "datasets/aida-conll/test_text.txt",
        },
        "casestudy": {
            "loader": load_text_lines_data,
            "input_path": "datasets/casestudy/turn me on text.txt",
        },
    }

    DATASET_TO_RUN = "aida-conll"  # "webnlg", "CaRB", "aida-conll"
    PROMPT_STRATEGY = "few_shot"  # "zero_shot", "few_shot"

    EXECUTE_INIT = True
    EXECUTE_EXTRACTION = False
    EXECUTE_VERIFICATION = False
    EXECUTE_DESCRIBE = True
    EXECUTE_INSERT = False
    EXECUTE_SEMANTIC = False
    EXECUTE_SQLINSERT = False

    print(f"\n{'='*20} Starting Pipeline {'='*20}")

    if DATASET_TO_RUN not in DATASET_CONFIGS:
        print(f"Error: Dataset '{DATASET_TO_RUN}' is not defined in DATASET_CONFIGS.")
        return
    if PROMPT_STRATEGY not in EXTRACTION_PROMPTS:
        print(f"Error: Prompt strategy '{PROMPT_STRATEGY}' is not defined in EXTRACTION_PROMPTS.")
        return

    modelname = MODEL.split("/")[-1]
    experiment_folder_name = f"{modelname}_{PROMPT_STRATEGY}"
    output_dir_for_run = os.path.join(
        "output", "AgentsKG", DATASET_TO_RUN, experiment_folder_name
    )

    os.makedirs(output_dir_for_run, exist_ok=True)

    file_paths = {
        "extracted": os.path.join(output_dir_for_run, "extracted_triplets.json"),
        "verified": os.path.join(output_dir_for_run, "verified_triplets.json"),
        "described": os.path.join(output_dir_for_run, "described_triplets.json"),
        "insert_result": os.path.join(output_dir_for_run, "insert_result.json"),
        "with_properties": os.path.join(
            output_dir_for_run, "triplets_with_properties.json"
        ),
    }

    print(f"Dataset: {DATASET_TO_RUN}")
    print(f"Prompt Strategy: {PROMPT_STRATEGY}")
    print(f"All output files for this run will be saved in: {output_dir_for_run}")
    print(f"{'='*58}\n")

    config = DATASET_CONFIGS[DATASET_TO_RUN]

    if EXECUTE_INIT:
        print("--- Step 0: Initializing Database Environment ---")
        MILVUS_COLLECTIONS_TO_CLEAN = ["entity", "relation", "triples"]
        clear_milvus_collections(MILVUS_COLLECTIONS_TO_CLEAN)
        print("--- Milvus Cleaning Finished ---\n")

    if EXECUTE_EXTRACTION:
        print("--- Step 1: Starting Triplet Extraction ---")
        initial_items_to_process = config["loader"](config["input_path"])
        if initial_items_to_process:
            await extract(
                initial_items_to_process=initial_items_to_process,
                output_json_file=file_paths["extracted"],
                strategy=PROMPT_STRATEGY,
                dataset_name=DATASET_TO_RUN,
            )
        print("--- Triplet Extraction Finished ---\n")

    if EXECUTE_VERIFICATION:
        print("--- Step 2: Starting Fact Verification ---")
        await verify(
            file_paths["extracted"],
            file_paths["verified"],
            DATASET_CONFIGS[DATASET_TO_RUN],
        )
        print("--- Fact Verification Finished ---\n")

    if EXECUTE_DESCRIBE:
        print("--- Step 3: Starting Description Generation ---")
        await describe(file_paths["verified"], file_paths["described"])
        print("--- Description Generation Finished ---\n")

    if EXECUTE_INSERT:
        print("--- Step 4: Starting Entity/Relation Insertion ---")
        await insert(file_paths["described"], file_paths["insert_result"])
        print("--- Entity/Relation Insertion Finished ---\n")

    if EXECUTE_SEMANTIC:
        print("--- Step 5: Starting Semantic Analysis ---")
        await semantic(file_paths["insert_result"], file_paths["with_properties"])
        print("--- Semantic Analysis Finished ---\n")

    if EXECUTE_SQLINSERT:
        print("--- Step 6a: Starting Triple Insertion to SQL DB ---")
        try:
            db_config = {
                "host": os.getenv("DB_HOST").strip(),
                "port": int(os.getenv("DB_PORT", 3306)),
                "user": os.getenv("DB_USER").strip(),
                "password": os.getenv("DB_PASSWORD").strip(),
                "db": os.getenv("DB_NAME").strip(),
            }
            await clear_sql(db_config, "triples", force=True)

        except Exception as e:
            print(f"Error occurred while initializing database: {e}")
            return
        print("--- Database Initialized ---\n")
        await sqlinsert(file_paths["with_properties"], db_config)
        print("--- Triple Insertion to SQL DB Finished ---\n")


async def ablation():
    """Main entry point for the KG extraction pipeline."""
    DATASET_CONFIGS = {
        "webnlg": {
            "loader": load_text_lines_data,
            "input_path": "datasets/webnlg/webnlg_test_texts.txt",
        },
        "CaRB": {
            "loader": load_text_lines_data,
            "input_path": "datasets/CaRB-master/data/test.txt",
        },
        "aida-conll": {
            "loader": load_text_lines_data,
            "input_path": "datasets/aida-conll/test_text.txt",
        },
    }

    DATASET_TO_RUN = "aida-conll"  # "webnlg", "CaRB"
    PROMPT_STRATEGY = "few_shot"  # "zero_shot", "few_shot"
    VERIFY_MODE = False  # True, False
    DESCRIPTION_MODE = False  # True, False
    EXECUTE_INIT = True
    EXECUTE_EXTRACTION = True
    EXECUTE_VERIFICATION = True
    EXECUTE_DESCRIBE = True
    EXECUTE_INSERT = True
    EXECUTE_SEMANTIC = False
    EXECUTE_SQLINSERT = False

    print(f"\n{'='*20} Starting Pipeline {'='*20}")

    if DATASET_TO_RUN not in DATASET_CONFIGS:
        print(f"Error: Dataset '{DATASET_TO_RUN}' is not defined in DATASET_CONFIGS.")
        return
    if PROMPT_STRATEGY not in EXTRACTION_PROMPTS:
        print(f"Error: Prompt strategy '{PROMPT_STRATEGY}' is not defined in EXTRACTION_PROMPTS.")
        return

    modelname = MODEL.split("/")[-1]
    experiment_folder_name = f"{modelname}_{PROMPT_STRATEGY}_{DESCRIPTION_MODE}"
    output_dir_for_run = os.path.join(
        "ablation", "none", DATASET_TO_RUN, experiment_folder_name
    )

    os.makedirs(output_dir_for_run, exist_ok=True)

    file_paths = {
        "extracted": os.path.join(output_dir_for_run, "extracted_triplets.json"),
        "verified": os.path.join(output_dir_for_run, "verified_triplets.json"),
        "described": os.path.join(output_dir_for_run, "described_triplets.json"),
        "insert_result": os.path.join(output_dir_for_run, "insert_result.json"),
        "with_properties": os.path.join(
            output_dir_for_run, "triplets_with_properties.json"
        ),
    }

    print(f"Dataset: {DATASET_TO_RUN}")
    print(f"Prompt Strategy: {PROMPT_STRATEGY}")
    print(f"All output files for this run will be saved in: {output_dir_for_run}")
    print(f"{'='*58}\n")

    config = DATASET_CONFIGS[DATASET_TO_RUN]

    if EXECUTE_INIT:
        print("--- Step 0: Initializing Database Environment ---")
        MILVUS_COLLECTIONS_TO_CLEAN = ["entity", "relation", "triples"]
        clear_milvus_collections(MILVUS_COLLECTIONS_TO_CLEAN)
        print("--- Milvus Cleaning Finished ---\n")

    if EXECUTE_EXTRACTION:
        print("--- Step 1: Starting Triplet Extraction ---")
        initial_items_to_process = config["loader"](config["input_path"])
        if initial_items_to_process:
            await extract(
                initial_items_to_process=initial_items_to_process,
                output_json_file=file_paths["extracted"],
                strategy=PROMPT_STRATEGY,
                dataset_name=DATASET_TO_RUN,
            )
        print("--- Triplet Extraction Finished ---\n")

    if EXECUTE_VERIFICATION:
        print("--- Step 2: Starting Fact Verification ---")
        await verify(
            file_paths["extracted"],
            file_paths["verified"],
            DATASET_CONFIGS[DATASET_TO_RUN],
            verify_mode=VERIFY_MODE,
        )
        print("--- Fact Verification Finished ---\n")

    if EXECUTE_DESCRIBE:
        print("--- Step 3: Starting Description Generation ---")
        await describe(
            file_paths["verified"],
            file_paths["described"],
            enable_llm_description=DESCRIPTION_MODE,
        )
        print("--- Description Generation Finished ---\n")

    if EXECUTE_INSERT:
        print("--- Step 4: Starting Entity/Relation Insertion ---")
        await insert(file_paths["described"], file_paths["insert_result"])
        print("--- Entity/Relation Insertion Finished ---\n")

    if EXECUTE_SEMANTIC:
        print("--- Step 5: Starting Semantic Analysis ---")
        await semantic(file_paths["insert_result"], file_paths["with_properties"])
        print("--- Semantic Analysis Finished ---\n")

    if EXECUTE_SQLINSERT:
        print("--- Step 6a: Starting Triple Insertion to SQL DB ---")
        try:
            db_config = {
                "host": os.getenv("DB_HOST").strip(),
                "port": int(os.getenv("DB_PORT", 3306)),
                "user": os.getenv("DB_USER").strip(),
                "password": os.getenv("DB_PASSWORD").strip(),
                "db": os.getenv("DB_NAME").strip(),
            }
            await clear_sql(db_config, "triples", force=True)

        except Exception as e:
            print(f"Error occurred while initializing database: {e}")
            return
        print("--- Database Initialized ---\n")
        await sqlinsert(file_paths["with_properties"], db_config)
        print("--- Triple Insertion to SQL DB Finished ---\n")


async def alignment_study():
    """
    Main entry point for the alignment sensitivity study, which runs a 3x3 grid search over Tau and Radius parameters.
    """
    DATASET_TO_RUN = "aida-conll"
    PROMPT_STRATEGY = "few_shot"

    input_base_path = (
        r"analysis\grid_search\aida-conll\Qwen3-30B-A3B-Instruct-2507_zero_shot"
    )
    source_described_file = os.path.join(input_base_path, "described_triplets.json")

    if not os.path.exists(source_described_file):
        print(f"Error: Source file {source_described_file} not found.")
        return

    TAU_VALUES = [0.05, 0.10, 0.15]
    RADIUS_VALUES = [0.4, 0.6, 0.8]
    total_runs = len(TAU_VALUES) * len(RADIUS_VALUES)
    current_run = 0

    print(f"\n{'='*20} Starting Alignment Sensitivity Study {'='*20}")
    print(f"Source Data: {source_described_file}")

    for tau in TAU_VALUES:
        for radius in RADIUS_VALUES:
            current_run += 1
            print(
                f"\n[Progress {current_run}/{total_runs}] Testing combination: Tau={tau}, Radius={radius}"
            )

            print(f"Clearing Milvus environment...")
            clear_milvus_collections(["entity", "relation"])
            await asyncio.sleep(5)

            os.environ["EXP_TAU"] = str(tau)
            os.environ["EXP_RADIUS"] = str(radius)
            global_stats.reset()

            output_folder = f"analysis/results/tau_{tau}_radius_{radius}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, "insert_result.json")

            await insert(source_described_file, output_file)

            metadata = {
                "tau": tau,
                "radius": radius,
                "llm_calls": global_stats.llm_calls,
                "output_file": output_file,
            }
            with open(os.path.join(output_folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"LLM calls: {global_stats.llm_calls}")

    print(f"\n{'='*20} 3x3 Grid Search Finished Total 9 Runs {'='*20}")


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(ablation())
    # asyncio.run(alignment_study())

    print("Pipeline execution finished.")
