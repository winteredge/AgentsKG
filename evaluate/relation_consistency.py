import json
import asyncio
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from agentskg.utils.embedding_model import get_embeddings


async def run_batch_relation_analysis(file_list):
    results = []
    for json_file_path in file_list:
        if not os.path.exists(json_file_path):
            continue

        print(f"🚀 正在分析 AgentsKG: {json_file_path}")
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取基础统计
        total_triplets = len(data)
        unique_relations = {}
        for item in data:
            p_id = item.get("predicate_id")
            p_desc = item.get("descriptions", {}).get("relation", "")
            if p_id and p_desc and p_id not in unique_relations:
                unique_relations[p_id] = p_desc

        p_texts = list(unique_relations.values())
        num_unique = len(p_texts)
        if num_unique < 2:
            continue

        embeddings_list = await get_embeddings(p_texts)
        valid_embeddings = np.array(
            [e if e is not None else [0.0] * 1024 for e in embeddings_list]
        )
        sim_matrix = cosine_similarity(valid_embeddings)

        # 指标计算：ANRS
        mask = ~np.eye(num_unique, dtype=bool)
        max_sims = [
            np.max(sim_matrix[i][mask[i]]) for i in range(num_unique) if np.any(mask[i])
        ]
        anrs = np.mean(max_sims) if max_sims else 0

        upper_tri = sim_matrix[np.triu_indices(num_unique, k=1)]

        # --- 核心修改：转化为每 100 个关系的冗余密度 (方案一) ---
        def get_density(threshold):
            abs_count = np.sum(upper_tri > threshold)
            return (abs_count / num_unique * 100) if num_unique > 0 else 0

        model_label = (
            json_file_path.split("/")[-2] if "/" in json_file_path else json_file_path
        )
        results.append(
            [
                model_label,
                total_triplets,
                num_unique,
                f"{anrs:.4f}",
                f"{get_density(0.9):.3f}",
                f"{get_density(0.8):.3f}",
                f"{get_density(0.7):.3f}",
                f"{get_density(0.6):.3f}",
            ]
        )

    return results


async def run_external_tsv_analysis(tsv_file_path):
    if not os.path.exists(tsv_file_path):
        return None
    print(f"🚀 正在分析外部基准: {os.path.basename(tsv_file_path)}")

    all_relations = []
    with open(tsv_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rel_text = parts[2].strip().strip("'").strip('"')
                if rel_text:
                    all_relations.append(rel_text)

    total_triplets = len(all_relations)
    unique_texts = list(set(all_relations))
    num_unique = len(unique_texts)
    if num_unique < 2:
        return None

    embeddings_list = await get_embeddings(unique_texts)
    valid_embeddings = np.array(
        [e if e is not None else [0.0] * 1024 for e in embeddings_list]
    )
    sim_matrix = cosine_similarity(valid_embeddings)

    mask = ~np.eye(num_unique, dtype=bool)
    max_sims = [
        np.max(sim_matrix[i][mask[i]]) for i in range(num_unique) if np.any(mask[i])
    ]
    anrs = np.mean(max_sims)

    upper_tri = sim_matrix[np.triu_indices(num_unique, k=1)]

    def get_density(threshold):
        abs_count = np.sum(upper_tri > threshold)
        return (abs_count / num_unique * 100) if num_unique > 0 else 0

    return [
        os.path.basename(tsv_file_path),
        total_triplets,
        num_unique,
        f"{anrs:.4f}",
        f"{get_density(0.9):.3f}",
        f"{get_density(0.8):.3f}",
        f"{get_density(0.7):.3f}",
        f"{get_density(0.6):.3f}",
    ]


async def main():
    headers = [
        "Model/Dir",
        "#Triplets",
        "#Unique_Rel",
        "ANRS",
        "Den>0.9",
        "Den>0.8",
        "Den>0.7",
        "Den>0.6",
    ]

    # 1. 分析 AgentsKG
    print("=== 开始分析 AgentsKG 结果 ===")
    files_to_run = [
        "output/AgentsKG/CaRB/Qwen2.5-72B-Instruct_zero_shot/insert_result.json",
        "output/AgentsKG/CaRB/Qwen2.5-72B-Instruct_few_shot/insert_result.json",
        "output/AgentsKG/CaRB/Qwen2.5-7B-Instruct_zero_shot/insert_result.json",
        "output/AgentsKG/CaRB/Qwen2.5-7B-Instruct_few_shot/insert_result.json",
    ]
    kg_results = await run_batch_relation_analysis(files_to_run)

    # 2. 分析外部基准
    print("\n=== 开始分析外部基准 ===")
    external_files = [
        "output/stanfordOIE/CaRB_results.tsv",
        "output/openie6/CaRB_for_eval.tsv",
        "output/sfgpt/CaRB/Qwen2.5-72B-Instruct_eval_0.7.tsv",
        "output/PiVe/CaRB/Qwen2.5-72B-Instruct_pive_2iters/rich_output.tsv",
        "output/DualOIE/CaRB/cot_results.tsv",
        "output/LLM/CaRB/Qwen2.5-72B-Instruct_zeroshot_rich_output.tsv",
        "output/LLM/CaRB/Qwen2.5-72B-Instruct_fewshot_rich_output.tsv",
        "output/LLM/CaRB/llama3_zeroshot_for_eval.tsv",
        "output/LLM/CaRB/llama3_fewshot_for_eval.tsv",
        "output/LLM/CaRB/deepseek-r1_zeroshot_for_eval.tsv",
        "output/LLM/CaRB/deepseek-r1_fewshot_for_eval.tsv",
    ]
    ext_results = []
    for tsv in external_files:
        res = await run_external_tsv_analysis(tsv)
        if res:
            ext_results.append(res)

    # 3. 统一打印 Markdown 表格
    all_data = kg_results + ext_results
    print("\n" + "=" * 110)
    print("📊 关系一致性实验汇总 (Den = Redundancy per 100 Relations)")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in all_data:
        print("| " + " | ".join(map(str, row)) + " |")
    print("=" * 110)


if __name__ == "__main__":
    asyncio.run(main())
