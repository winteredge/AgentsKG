import asyncio
import json
import os
from pathlib import Path
import re
import numpy as np
from collections import defaultdict
from pymilvus import MilvusClient, connections, utility, Collection
import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from agentskg.utils.embedding_model import get_embeddings

# ==============================================================================
# 1. 全局配置
# ==============================================================================

load_dotenv()
RUN_INTERNAL_QUALITY_EVAL = True
RUN_RAG_EVALUATION = True


# --- Zilliz Cloud 配置 ---
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

# --- LLM 配置 ---
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
EVALUATION_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# --- 文件配置 ---
EXPERIMENT_DIR = Path("output/arxiv_Qwen2.5-72B-Instruct_few_shot")
# EXPERIMENT_DIR = Path("results/arxiv_Qwen2.5-72B-Instruct_zero_shot")

INPUT_TRIPLE_FILE = EXPERIMENT_DIR / "triplets_with_properties.json"
RELATION_COLLECTION_NAME = "relation"
QA_DATASET_FILE = Path("datasets/arxiv/output_qa_pairs.jsonl")
TRIPLES_COLLECTION_NAME = "triples"
RAG_SEARCH_VECTOR_FIELD = "embedding_rich"

INTERNAL_QUALITY_REPORT_FILE = EXPERIMENT_DIR / "internal_quality_report.json"
RAG_RESULTS_FILE = EXPERIMENT_DIR / "rag_evaluation_results.json"
RAG_SCORED_RESULTS_FILE = EXPERIMENT_DIR / "rag_evaluation_with_scores.json"


# ==============================================================================
# 2. 关系冗余度评估模块
# ==============================================================================


async def evaluate_relation_redundancy(collection_name: str) -> float:
    """
    连接到 Milvus，计算并返回关系集合的冗余度指标。
    指标定义：每个关系与其最相似的另一个关系的平均余弦相似度。
    返回值：平均相似度得分，若失败则返回 -1.0。
    """
    print("\n" + "=" * 20 + " 阶段 1: 关系冗余度评估 " + "=" * 20)
    try:
        connections.connect(alias="relation_eval", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        print(f"成功连接到 Zilliz Cloud (alias: relation_eval)。")

        if not utility.has_collection(collection_name, using="relation_eval"):
            print(f"错误: Milvus 集合 '{collection_name}' 不存在。")
            return -1.0

        collection = Collection(collection_name, using="relation_eval")
        total_entities = collection.num_entities
        print(f"成功获取集合: '{collection_name}', 总关系数: {total_entities}")

        if total_entities < 2:
            print("集合中实体少于2个，无法计算冗余度。")
            return 0.0

        collection.load()
        print("集合加载完成。")

        print("正在从集合中分页查询所有关系向量...")
        all_embeddings = []
        offset = 0
        while True:
            batch_results = collection.query(
                expr="", output_fields=["embedding"], limit=1000, offset=offset
            )
            if not batch_results:
                break
            all_embeddings.extend(
                np.array([item["embedding"] for item in batch_results])
            )
            offset += len(batch_results)
            if offset >= total_entities:
                break

        print(f"成功查询到 {len(all_embeddings)} 个向量。")
        all_embeddings = np.array(all_embeddings)

        print("正在分批进行相似度搜索...")
        search_batch_size = 10  # Zilliz Cloud Serverless 限制 nq <= 10
        closest_similarities = []
        for i in range(0, len(all_embeddings), search_batch_size):
            batch_vectors = all_embeddings[i : i + search_batch_size]

            search_params = {"metric_type": "COSINE", "params": {}}
            batch_search_results = collection.search(
                data=batch_vectors,
                anns_field="embedding",
                param=search_params,
                limit=2,
            )

            for hits in batch_search_results:
                if len(hits) > 1:
                    closest_similarities.append(hits[1].distance)

        collection.release()

        if not closest_similarities:
            print("未能计算出任何相似度得分。")
            return -1.0

        average_similarity = np.mean(closest_similarities)
        print(f"\n--- 关系冗余度评估结果 ---")
        print(f"平均最近邻余弦相似度: {average_similarity:.4f}")
        print("（此值越接近1，表示关系定义之间的相似度越高，可能存在越多语义冗余）")
        return average_similarity

    except Exception as e:
        print(f"在评估关系冗余度时发生错误: {e}")
        return -1.0
    finally:
        if "relation_eval" in connections.list_connections():
            connections.disconnect("relation_eval")
            print("已断开与 Zilliz Cloud 的连接 (alias: relation_eval)。")


# ==============================================================================
# 3. 三元组质量评估模块
# ==============================================================================

EVALUATION_LLM_PROMPT = """## Role
You are an expert **Knowledge Graph Architect**. Your primary task is to evaluate the quality of a knowledge triplet based on its suitability for building a clean, scalable, and reusable knowledge graph.

## Core Principle: Schema-Adherence and Normalization
Your evaluation must prioritize **structural quality and adherence to a schema** over mere literal faithfulness to the source text. The ideal triplet is one that has been correctly normalized and can be seamlessly integrated into a formal ontology.

## Detailed Scoring Criteria (1-5)

- **5 (Excellent / Production-Ready):**
  - **Factually Accurate**: The statement is correct.
  - **Perfectly Normalized**: The predicate is a **concise, schema-like property** (e.g., `birthDate`, `location`, `manager`). Verbs and conversational phrases have been correctly converted into properties.
  - **Fully Atomic**: Subject and object are clean, resolved entities.
  - *This is the gold standard for a clean knowledge graph.*
  *(Example: ['Microsoft', 'headquartersLocation', 'Redmond'])*

- **4 (Good / Needs Minor Refinement):**
  - **Factually Accurate**: The statement is correct.
  - **Good Normalization Attempt**: The predicate is close to a schema property but could be better (e.g., using `establishedIn` instead of the more standard `creationDate`).
  - **Mostly Atomic**: Entities are mostly clean.
  *(Example: ['ALCO RS-3', 'powerType', 'Diesel-electric transmission'])*

- **3 (Acceptable / Raw but Correct):**
  - **Factually Accurate**: The statement is correct.
  - **Poor Normalization (Verb-based)**: The predicate is a **raw verb phrase directly from the text** (e.g., 'was born in', 'is managed by'). While factually correct, this is **undesirable** for a structured KG and requires significant post-processing.
  - *This score represents a failure in normalization.*
  *(Example: ['Barack Obama', 'was born in', 'Honolulu'])*

- **2 (Poor / Questionable Accuracy):**
  - **Factual Accuracy is Questionable or Likely Incorrect**: You have strong doubts about the fact's validity.
  - Normalization quality is irrelevant if the fact is wrong.
  *(Example: ['all birds', 'can', 'fly'])*

- **1 (Unusable / False):**
  - **Demonstrably False or Nonsensical**: The statement is clearly wrong.
  *(Example: ['the moon', 'is made of', 'cheese'])*

## Task
Evaluate the following triplets and for each, return a JSON object with "score" and "justification". Your justification should be brief and **mention why it meets (or fails) the normalization standard.**

### Example Input 1:
Triplet: ('Barack Obama', 'was born in', 'Honolulu')
### Your Expected Output 1:
{"score": 3, "justification": "Factually correct, but the predicate 'was born in' is a raw verb phrase and has not been normalized to a schema property like 'birthPlace'."}

### Example Input 2:
Triplet: ('Microsoft', 'birthPlace', 'Albuquerque')
### Your Expected Output 2:
{"score": 5, "justification": "Excellent. The predicate 'birthPlace' is a perfect, schema-like property."}


## Begin Assessment - Input Triplets:
"""


class TripletQualityAgent:
    """使用LLM对三元组进行直接打分的代理。"""

    def __init__(self, concurrency_limit: int = 5):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        self.client = httpx.AsyncClient(timeout=180.0, limits=limits)

    async def evaluate_batch(self, triplets_batch):
        if not triplets_batch:
            return []
        input_texts = [f"Triplet: {str(tuple(t.values())[:3])}" for t in triplets_batch]
        full_prompt = EVALUATION_LLM_PROMPT + "\n" + "\n".join(input_texts)

        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
            "model": EVALUATION_MODEL,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        api_url = f"{API_URL.rstrip('/')}/chat/completions"
        try:
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            content = (
                response.json()["choices"][0].get("message", {}).get("content", "")
            )
            return self._parse_batch_response(content, len(triplets_batch))

        except Exception as e:
            print(
                f"\n[!] LLM批量评估API调用失败 (批次大小: {len(triplets_batch)}): {e}"
            )
            return [
                self._default_error_response("API Call Failed") for _ in triplets_batch
            ]

    def _parse_batch_response(self, content: str, count: int) -> list:
        """【新】健壮的批量JSON解析器。"""
        results = []
        if not content:
            while len(results) < count:
                results.append(self._default_error_response("Empty LLM Response"))
            return results[:count]

        json_pattern = re.compile(r"\{.*?\}")
        lines = content.strip().split("\n")
        for line in lines:
            match = json_pattern.search(line)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if "score" in parsed and "justification" in parsed:
                        results.append(parsed)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行内JSON: '{match.group(0)}'")

        while len(results) < count:
            results.append(self._default_error_response("Parsing Mismatch"))

        return results[:count]

    def _default_error_response(self, reason: str = "评估失败"):
        return {"score": 0, "justification": f"{reason}，默认为0分"}

    async def close(self):
        await self.client.aclose()


async def evaluate_triplet_quality(input_file: str, output_file: str) -> dict:
    """
    加载三元组文件，使用LLM进行打分，并返回评估摘要（修改为逐一评估）。
    """
    print("\n" + "=" * 20 + " 阶段 2: 三元组质量评估 " + "=" * 20)
    if not API_KEY or not API_URL:
        print("错误: 缺少 API_KEY 或 API_URL 环境变量，无法进行LLM评估。")
        return {}

    # --- 1. 数据加载 ---
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return {}
    if isinstance(data, dict):
        all_triplets = [t for triplets in data.values() for t in triplets]
    elif isinstance(data, list):
        all_triplets = data
    else:
        print(f"错误: 输入文件 '{input_file}' 的内容格式不正确。")
        return {}
    if not all_triplets:
        print("文件中没有找到三元组。")
        return {}

    # --- 2. 逐一评估 ---
    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    agent = TripletQualityAgent()
    batch_size = 1

    print(f"共找到 {len(all_triplets)} 个三元组，开始逐一评估...")

    batches = [
        all_triplets[i : i + batch_size]
        for i in range(0, len(all_triplets), batch_size)
    ]

    async def run_batch_with_semaphore(batch):
        async with semaphore:
            return await agent.evaluate_batch(batch)

    tasks = [run_batch_with_semaphore(batch) for batch in batches]
    results_from_batches = await tqdm.gather(*tasks, desc="三元组质量评分中")
    all_eval_results = [item for sublist in results_from_batches for item in sublist]

    for triplet, evaluation in zip(all_triplets, all_eval_results):
        triplet["llm_quality_evaluation"] = evaluation

    await agent.close()

    # --- 3. 报告与统计 ---
    score_counts = defaultdict(int)
    valid_scores = []
    for triplet in all_triplets:
        score = 0
        evaluation = triplet.get("llm_quality_evaluation")
        if isinstance(evaluation, dict):
            score_val = evaluation.get("score")
            if isinstance(score_val, int):
                score = score_val
        if 1 <= score <= 5:
            valid_scores.append(score)
        score_counts[score] += 1

    average_score = np.mean(valid_scores) if valid_scores else 0

    summary = {
        "total_triplets": len(all_triplets),
        "evaluated_count": len(valid_scores),
        "average_score": float(f"{average_score:.2f}"),
        "score_distribution": dict(
            sorted(score_counts.items(), key=lambda item: item[0], reverse=True)
        ),
    }

    print("\n--- 三元组质量评估结果 ---")
    print(f"平均质量得分: {summary['average_score']:.2f} / 5.00")
    print("分数分布:")
    for score, count in summary["score_distribution"].items():
        percentage = (count / len(all_triplets)) * 100
        print(f"  - {score}分: {count:>5} 个 ({percentage:.2f}%)")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=2)
    print(f"详细评估报告已保存到: {output_file}")

    return summary


# ==============================================================================
# 4. QA能力评估模块
# ==============================================================================


async def ask_llm_for_rag_answer_batch(
    questions: list[str],
    contexts: list[list[str]],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """使用 LLM 基于上下文批量生成答案"""
    if not all([API_KEY, API_URL]):
        return ["LLM API config missing." for _ in questions]

    api_url = f"{API_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "You are a rigorous Q&A bot. Base your answers only on the provided context."
    )

    async def single_request(question, context):
        async with semaphore:
            context_str = "\n".join(context) if context else "No context provided."
            user_prompt = f"""Please answer the question based only on the 'Context Information' provided below. Do not use any of your own knowledge. If the context is insufficient to answer, reply exactly with: 'Based on the provided context, the question cannot be answered.'
    --- Context Information ---
    {context_str}
    --- Question ---
    {question}"""
            payload = {
                "model": EVALUATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            }

            try:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(
                    f"RAG answer generation error for question '{question[:30]}...': {e}"
                )
                return f"Error generating answer: {type(e).__name__}"

    tasks = [single_request(q, c) for q, c in zip(questions, contexts)]
    from tqdm.asyncio import tqdm

    return await tqdm.gather(*tasks, desc="批量生成RAG答案")


async def score_rag_answer_batch(
    questions: list[str],
    ground_truths: list[str],
    generated_answers: list[str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list[int]:
    """【新】使用 LLM 作为裁判批量评估答案质量"""
    if not all([API_KEY, API_URL]):
        return [-1 for _ in questions]

    api_url = f"{API_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    system_prompt = "You are an extremely rigorous, objective, and detail-oriented AI QA evaluation expert. Your task is to compare a 'Model's Answer' to a 'Ground Truth Answer' and provide an integer score. You must strictly follow the scoring criteria and output only a single number (0, 1, or 2), with no additional text or explanations."

    async def single_request(question, ground_truth, generated_answer):
        async with semaphore:
            user_prompt = f"""Please evaluate the consistency and completeness of the 'Model's Answer' against the 'Ground Truth Answer'.
    --- Scoring Criteria ---
    - **Score 2 (Fully Correct):** The 'Model's Answer' is factually identical to the 'Ground Truth Answer' and covers all its key information points...
    - **Score 1 (Partially Correct):** The 'Model's Answer' contains some of the key information...
    - **Score 0 (Incorrect):** The core facts of the 'Model's Answer' contradict the 'Ground Truth Answer'...
    --- Content to Evaluate ---
    [Question]: {question}
    [Ground Truth Answer]: {ground_truth}
    [Model's Answer]: {generated_answer}
    --- Your Score (Return only 0, 1, or 2) ---"""
            payload = {
                "model": EVALUATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 5,
                "temperature": 0.0,
            }

            try:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                score_str = "".join(filter(str.isdigit, content))
                return int(score_str) if score_str in ["0", "1", "2"] else -1
            except Exception as e:
                print(f"RAG scoring error for question '{question[:30]}...': {e}")
                return -1

    tasks = [
        single_request(q, gt, ga)
        for q, gt, ga in zip(questions, ground_truths, generated_answers)
    ]
    from tqdm.asyncio import tqdm

    return await tqdm.gather(*tasks, desc="批量评分RAG答案")


# --- RAG 评估的主函数 ---
async def evaluate_rag_performance(
    qa_file: Path,
    triples_collection: str,
    vector_field: str,
    output_raw_file: Path,
    output_scored_file: Path,
) -> dict:
    """完整的 RAG 评估流程（修改为批量异步）。"""
    print("\n" + "=" * 20 + " 阶段 3: RAG 问答评估 " + "=" * 20)

    # 1. 连接 Milvus 和加载数据
    try:
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        if not client.has_collection(triples_collection):
            print(f"错误: Milvus 集合 '{triples_collection}' 不存在。")
            return {}
    except Exception as e:
        print(f"连接 Zilliz Cloud 失败: {e}")
        return {}

    qa_dataset = []
    try:
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_dataset = [json.loads(line) for line in f if line.strip()]
        print(f"成功从 .jsonl 文件 '{qa_file}' 加载了 {len(qa_dataset)} 条 QA 数据。")
    except Exception as e:
        print(f"加载 QA 数据集失败: {e}")
        return {}

    all_questions = [item["question"] for item in qa_dataset]
    print("正在批量转换问题为向量...")
    all_question_vectors = await get_embeddings(all_questions)

    # 2. 批量检索
    print(f"正在分批执行向量搜索 (使用字段 '{vector_field}')...")
    all_search_results = []
    BATCH_SIZE = 10
    TOP_K = 5
    for i in tqdm(range(0, len(all_question_vectors), BATCH_SIZE), desc="向量搜索中"):
        batch_vectors = all_question_vectors[i : i + BATCH_SIZE]
        try:
            batch_results = client.search(
                collection_name=triples_collection,
                data=batch_vectors,
                limit=TOP_K,
                output_fields=["text_rich", "document_id"],
                anns_field=vector_field,
                search_params={"metric_type": "COSINE"},
            )
            all_search_results.extend(batch_results)
        except Exception as e:
            print(f"\n搜索批次失败: {e}")
            all_search_results.extend([[] for _ in batch_vectors])

    print("上下文检索完成。")
    client.close()

    # 3.批量生成答案和批量评分
    generated_results = []
    scores = []
    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with httpx.AsyncClient(timeout=120.0) as async_client:
        # 批量生成答案
        print("\n--- 开始批量生成答案 ---")
        all_contexts = [
            [hit["entity"]["text_rich"] for hit in hits] if hits else []
            for hits in all_search_results
        ]
        generated_answers = await ask_llm_for_rag_answer_batch(
            all_questions, all_contexts, async_client, semaphore
        )

        generated_results = [
            {
                "question": qa_item["question"],
                "ground_truth_answer": qa_item["answer"],
                "generated_answer": gen_ans,
                "retrieved_context": context,
                "ground_truth_document_id": qa_item.get("id"),
            }
            for qa_item, gen_ans, context in zip(
                qa_dataset, generated_answers, all_contexts
            )
        ]
        with open(output_raw_file, "w", encoding="utf-8") as f:
            json.dump(generated_results, f, ensure_ascii=False, indent=2)
        print(f"RAG 答案批量生成完毕，结果保存至: {output_raw_file}")

        # 批量评分
        print("\n--- 开始进行批量评分 ---")
        all_ground_truths = [item["ground_truth_answer"] for item in generated_results]
        scores = await score_rag_answer_batch(
            all_questions, all_ground_truths, generated_answers, async_client, semaphore
        )

    scored_results = []
    reciprocal_ranks = []
    for i, case in enumerate(generated_results):
        score = scores[i]
        search_hits = all_search_results[i]
        ground_truth_doc_id = case.get("ground_truth_document_id")
        rank = 0
        if ground_truth_doc_id is not None and search_hits:
            for rank_idx, hit in enumerate(search_hits):
                if str(hit["entity"]["document_id"]) == str(ground_truth_doc_id):
                    rank = rank_idx + 1
                    break

        reciprocal_rank = 1 / rank if rank > 0 else 0
        reciprocal_ranks.append(reciprocal_rank)

        scored_results.append(
            {
                **case,
                "llm_evaluation_score": score,
                "source_rank": rank or None,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    with open(output_scored_file, "w", encoding="utf-8") as f:
        json.dump(scored_results, f, ensure_ascii=False, indent=2)
    print(f"评分完成，结果保存至: {output_scored_file}")

    # 5. 统计总结
    valid_scores = [s for s in scores if s >= 0]
    if not valid_scores:
        return {"error": "No valid scores generated."}

    total_q = len(scored_results)
    strict_accuracy = valid_scores.count(2) / total_q
    overall_correctness = (valid_scores.count(2) + valid_scores.count(1)) / total_q
    mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    source_match_accuracy = (
        sum(1 for r in reciprocal_ranks if r > 0) / total_q if total_q > 0 else 0
    )

    summary = {
        "total_questions": total_q,
        "strict_accuracy": float(f"{strict_accuracy:.4f}"),
        "overall_correctness": float(f"{overall_correctness:.4f}"),
        f"mrr_at_{TOP_K}": float(f"{mean_reciprocal_rank:.4f}"),
        "source_match_accuracy": float(f"{source_match_accuracy:.4f}"),
    }
    return summary


# ==============================================================================
# 5. 主执行流程
# ==============================================================================


async def main():
    """统一的评估主流程"""
    if not ZILLIZ_URI or not ZILLIZ_TOKEN:
        print("错误：请确保 .env 文件中已正确配置 ZILLIZ_URI 和 ZILLIZ_TOKEN。")
        exit(1)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    final_report = {}

    if RUN_INTERNAL_QUALITY_EVAL:
        redundancy_score = await evaluate_relation_redundancy(RELATION_COLLECTION_NAME)
        quality_summary = await evaluate_triplet_quality(
            INPUT_TRIPLE_FILE, INTERNAL_QUALITY_REPORT_FILE
        )

        final_report["internal_quality"] = {
            "relation_redundancy": redundancy_score,
            **(quality_summary or {}),  # 使用 or {} 避免 quality_summary 为 None 时出错
        }

    if RUN_RAG_EVALUATION:
        rag_summary = await evaluate_rag_performance(
            QA_DATASET_FILE,
            TRIPLES_COLLECTION_NAME,
            RAG_SEARCH_VECTOR_FIELD,
            RAG_RESULTS_FILE,
            RAG_SCORED_RESULTS_FILE,
        )
        final_report["rag_performance"] = rag_summary or {}

    print("\n\n" + "#" * 30)
    print("### 最终评估总结合报告 ###")
    print(f"实验目录: {EXPERIMENT_DIR}")
    print("#" * 30)
    print(json.dumps(final_report, indent=4, ensure_ascii=False))
    print("\n评估流程执行完毕。")


if __name__ == "__main__":
    asyncio.run(main())
