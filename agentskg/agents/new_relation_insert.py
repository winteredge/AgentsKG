import asyncio
import json
import os
import httpx
import time
from agentskg.agents.base import AgentConfig, BaseAgent
from pymilvus import MilvusClient
from agentskg.utils.embedding_model import get_embeddings
from agentskg.prompts.merge_prompt import MERGE_RELATION_PROMPT
from agentskg.utils.statuscounter import global_stats


class RelationAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        init_start_time = time.time()
        self.zilliz_uri = os.getenv("ZILLIZ_URI")
        self.zilliz_token = os.getenv("ZILLIZ_TOKEN")
        self.collection_name = "relation"

        try:
            self.milvus_client = MilvusClient(
                uri=self.zilliz_uri, token=self.zilliz_token
            )
            print("Successfully connected to Zilliz Cloud.")
            self._verify_milvus_collection()
            print("Milvus collection verification completed.")
        except Exception as e:
            print(f"Failed to connect to Zilliz Cloud: {e}")
            raise

    def _verify_milvus_collection(self):
        """Verify that the Milvus collection exists and the configuration generally meets expectations."""
        print(f"Verifying Milvus collection '{self.collection_name}'...")
        if not self.milvus_client.has_collection(self.collection_name):
            error_msg = f"Error: Milvus collection '{self.collection_name}' does not exist. Please create it manually on Zilliz Cloud and ensure its Schema (Auto_id, embedding, text) and index (embedding_index on 'embedding' field, COSINE) are correct."
            print(error_msg)
            raise ValueError(error_msg)

        print(f"Collection '{self.collection_name}' exists.")

    async def _search_similar_relation_in_milvus(self, vector: list):
        if (
            not vector
            or not isinstance(vector, list)
            or not all(isinstance(x, (float, int)) for x in vector)
        ):
            return []

        search_start_time = time.time()
        try:
            results = await asyncio.to_thread(
                self.milvus_client.search,
                collection_name=self.collection_name,
                data=[vector],
                limit=5,
                search_params={"metric_type": "COSINE", "params": {"radius": 0.8}},
                output_fields=["text"],
            )
            search_end_time = time.time()
            # print(
            #     f"  [TIME] _search_similar_relation_in_milvus Time taken: {search_end_time - search_start_time:.4f} 秒"
            # )

            if results and results[0]:
                return results[0]
            return []
        except Exception as e:
            search_end_time = time.time()
            print(
                f"Failed to search in Milvus: {e} (Time taken: {search_end_time - search_start_time:.4f} seconds)"
            )
            return []

    async def _insert_relation_to_milvus(self, description: str, vector: list):
        if (
            not vector
            or not isinstance(vector, list)
            or not all(isinstance(x, (float, int)) for x in vector)
        ):
            return None

        insert_total_start_time = time.time()
        data_to_insert = [
            {
                "text": description,
                "embedding": vector,
            }
        ]
        new_relation_id = None
        try:
            mutation_result = await asyncio.to_thread(
                self.milvus_client.insert,
                collection_name=self.collection_name,
                data=data_to_insert,
            )
            new_relation_id = mutation_result["ids"][0]
            # print(f"  [Milvus Insert Client] Successfully inserted. New relation ID: {new_relation_id}")
            insert_total_end_time = time.time()
            # print(
            #     f"  [TIME] _insert_relation_to_milvus Total time taken: {insert_total_end_time - insert_total_start_time:.4f} seconds"
            # )
            return new_relation_id

        except Exception as e:
            insert_total_end_time = time.time()
            print(
                f"Failed to insert into Milvus (Description: '{description[:30]}...'): {e} (Time taken: {insert_total_end_time - insert_total_start_time:.4f} seconds)"
            )
            return None

    async def _merge_relation(
        self, relation: str, description: str, milvus_hit: list[object]
    ) -> int:
        if not relation or not description or not milvus_hit:
            return 0
        global_stats.increment()
        merge_total_start_time = time.time()
        candidate_relations_string_parts = []
        for i, hit_dict in enumerate(milvus_hit):
            try:
                # print(hit_dict)
                hit_entity_data = hit_dict.get("entity", {})
                hit_desc = (
                    hit_entity_data.get("text", "N/A") if hit_entity_data else "N/A"
                )
                hit_id = hit_dict.get("Auto_id", "N/A")
                hit_distance = hit_dict.get("distance", "N/A")
                candidate_relations_string_parts.append(
                    f"{i+1}. Description: {hit_desc} (ID: {hit_id}, Similarity: {hit_distance:.4f})"
                )
            except AttributeError:
                print(
                    f"  [LLM Merge] Warning: Candidate hit at index {i} has unexpected structure. Skipping."
                )
                candidate_relations_string_parts.append(f"{i+1}. [Data format error]")
        candidate_relations_formatted_string = "\n".join(
            candidate_relations_string_parts
        )

        if not candidate_relations_formatted_string.strip():
            print(
                "  [LLM Merge] No valid candidate relations to present to LLM after formatting."
            )
            return 0

        final_prompt = MERGE_RELATION_PROMPT.format(
            relation=relation,
            description=description,
            other_relations=candidate_relations_formatted_string,
        )
        api_url = os.getenv("API_URL")
        api_key = os.getenv("API_KEY")
        if not api_url or not api_key:
            print("Error: _merge_relation failed to load API_URL or API_KEY.")
            return 0

        api_url = f"{api_url.rstrip('/')}/chat/completions"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        timeout_config = httpx.Timeout(30.0, connect=5.0)
        res = 0

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                res_content = response_data["choices"][0]["message"]["content"].strip()
                res_content = res_content.splitlines()[0].strip()
                res = int(res_content)

                merge_total_end_time = time.time()
                print(
                    f"  [TIME] _merge_relation (Success) Total time taken: {merge_total_end_time - merge_total_start_time:.4f} seconds"
                )
                return res

            except (httpx.HTTPStatusError, httpx.RequestError, Exception) as e:
                error_msg_prefix = "RelationAgent - "
                if isinstance(e, httpx.HTTPStatusError):
                    error_msg = f"{error_msg_prefix}{e}"
                elif isinstance(e, httpx.RequestError):
                    error_msg = f"{error_msg_prefix}Request error: {e.request.url!r} - {e}"
                else:
                    error_msg = f"{error_msg_prefix}: {e}"

                print(f"{error_msg}")

                merge_total_end_time = time.time()
                print(
                    f"  [TIME] _merge_relation (Failed) Total time taken: {merge_total_end_time - merge_total_start_time:.4f} seconds"
                )
                return 0

    async def execute(self, triples_input_list: list[dict]):
        execute_total_start_time = time.time()
        print(f"[TIME] RelationAgent.execute Starting execution...")

        step1_start_time = time.time()
        unique_descriptions_map = {}
        all_relation_occurrences = []
        for i, triple_data in enumerate(triples_input_list):
            name, desc = (
                triple_data["predicate"],
                triple_data["descriptions"]["relation"],
            )
            all_relation_occurrences.extend(
                [
                    {
                        "triple_idx": i,
                        "role": "predicate",
                        "name": name,
                        "description": desc,
                        "id": None,
                    }
                ]
            )
            if desc not in unique_descriptions_map:
                unique_descriptions_map[desc] = []
            unique_descriptions_map[desc].append(
                {"name": name, "context": f"Triple {i}, predicate"}
            )
        step1_end_time = time.time()
        # print(
        #     f"  [TIME] Step 1 (Preparing unique descriptions) took: {step1_end_time - step1_start_time:.4f} seconds"
        # )

        list_of_unique_descriptions = list(unique_descriptions_map.keys())
        total_unique_descriptions = len(list_of_unique_descriptions)
        if not list_of_unique_descriptions:
            execute_total_end_time = time.time()
            print(
                f"[TIME] RelationAgent.execute (No unique descriptions) Total time taken: {execute_total_end_time - execute_total_start_time:.4f} seconds"
            )
            return triples_input_list

        step2_start_time = time.time()
        embedding_vectors = await get_embeddings(list_of_unique_descriptions)
        step2_end_time = time.time()
        print(
            f"  [TIME] Step 2 (Fetching {total_unique_descriptions} embedding vectors) took: {step2_end_time - step2_start_time:.4f} seconds"
        )

        if len(embedding_vectors) != len(list_of_unique_descriptions) or not all(
            v is not None for v in embedding_vectors if isinstance(v, list)
        ):
            print("[Error] Failed to fetch embedding vectors or some vectors are invalid.")
            execute_total_end_time = time.time()
            print(
                f"[TIME] RelationAgent.execute (Embedding fetch failed) Total time taken: {execute_total_end_time - execute_total_start_time:.4f} seconds"
            )
            return [dict(t, predicate_id=None) for t in triples_input_list]

        step3_total_start_time = time.time()
        description_to_milvus_id_cache = {}
        for i, description in enumerate(list_of_unique_descriptions):
            loop_iter_start_time = time.time()
            print(
                f"    Processing unique relation description {i+1}/{total_unique_descriptions}..."
            )
            vector = embedding_vectors[i]
            representative_name = unique_descriptions_map[description][0]["name"]
            if not vector:
                description_to_milvus_id_cache[description] = None
                print(
                    f"      Skipping due to no vector. Iteration time: {time.time() - loop_iter_start_time:.4f} 秒"
                )
                continue

            similar_hits_list = await self._search_similar_relation_in_milvus(vector)
            milvus_id_for_this_description = None

            if similar_hits_list:
                which_merge = await self._merge_relation(
                    representative_name, description, similar_hits_list
                )
                print("which merge:", which_merge)
                if which_merge is not None:
                    if 1 <= which_merge <= len(similar_hits_list):
                        selected_hit_object = similar_hits_list[which_merge - 1]
                        milvus_id_for_this_description = selected_hit_object.get(
                            "Auto_id", None
                        )
                        print(
                            f"  [Execute] (Milvus ID: {milvus_id_for_this_description})"
                        )
                    else:
                        print(
                            f"  [Execute] Candidates available: {len(similar_hits_list)}. Not merging."
                        )
                else:
                    print(
                        f"  [Execute] LLM decided NOT to merge for '{description[:50]}...' (Returned: {which_merge})"
                    )

            if milvus_id_for_this_description is None:
                new_id = await self._insert_relation_to_milvus(description, vector)
                milvus_id_for_this_description = new_id
                await asyncio.sleep(0.4)

            description_to_milvus_id_cache[description] = milvus_id_for_this_description
            print(
                f"      Iteration {i+1}/{total_unique_descriptions} processed. 耗时: {time.time() - loop_iter_start_time:.4f} 秒"
            )

        step3_total_end_time = time.time()
        print(
            f"  [TIME] Step 3 (Processing {total_unique_descriptions} unique descriptions for relation linking) took: {step3_total_end_time - step3_total_start_time:.4f} seconds"
        )

        step4_start_time = time.time()
        output_triples_with_ids = [triple.copy() for triple in triples_input_list]
        for relation_info in all_relation_occurrences:
            triple_idx, role, desc = (
                relation_info["triple_idx"],
                relation_info["role"],
                relation_info["description"],
            )
            relation_milvus_id = description_to_milvus_id_cache.get(desc)
            output_triples_with_ids[triple_idx]["predicate_id"] = relation_milvus_id

        step4_end_time = time.time()
        print(
            f"  [TIME] Step 4 (Post-processing data) took: {step4_end_time - step4_start_time:.4f} seconds"
        )

        execute_total_end_time = time.time()
        print(
            f"[TIME] RelationAgent.execute Total time taken: {execute_total_end_time - execute_total_start_time:.4f} seconds"
        )
        return output_triples_with_ids

    async def validate(self):
        pass
