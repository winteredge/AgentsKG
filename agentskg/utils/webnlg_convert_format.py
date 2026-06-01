import json
import os


def convert_flat_json_to_eval_format(input_filepath, output_filepath):
    """
    Convert a flat list of triplets into an evaluation format aligned by document_id.
    Input format: [{ "s":..., "p":..., "o":..., "document_id": 1 }, ...]
    Output format: One JSON list per line [[s,p,o], [s,p,o]]
    """
    print(f"--- Start Conversion ---")
    print(f"Reading input file: {input_filepath}")

    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    if not isinstance(data, list):
        print("Error: Input data must be a list (List).")
        return
    grouped_data = {}
    max_doc_id = 0

    count = 0
    for item in data:
        doc_id = item.get("document_id")
        if doc_id is None:
            continue

        doc_id = int(doc_id)
        if doc_id > max_doc_id:
            max_doc_id = doc_id

        s = item.get("subject")
        p = item.get("predicate")
        o = item.get("object")

        if s and p and o:
            if doc_id not in grouped_data:
                grouped_data[doc_id] = []
            grouped_data[doc_id].append([s, p, o])
            count += 1

    print(f"Extracted {count} triplets in total, involving {len(grouped_data)} document IDs.")
    print(f"Maximum document ID: {max_doc_id}")

    with open(output_filepath, "w", encoding="utf-8") as f:

        for i in range(1, max_doc_id + 1):
            triplets = grouped_data.get(i, [])
            line = json.dumps(triplets, ensure_ascii=False)
            f.write(line + "\n")

    print(f"--- Conversion Completed ---")
    print(f"Results saved to: {output_filepath}")
    print(f"Total lines: {max_doc_id}")


def convert_json_to_eval_format(input_json_path, output_txt_path, total_lines=2155):
    """
    Convert a JSON file in results_by_id format to a WebNLG golden standard style txt file.
    Each line format: [[s, p, o], [s, p, o]]
    """
    print(f"--- Start Conversion to WebNLG Format ---")

    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    results_by_id = data.get("results_by_id", {})

    count = 0
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(1, total_lines + 1):
            str_id = str(i)
            triplets_in_doc = results_by_id.get(str_id, [])

            formatted_list = []
            for t in triplets_in_doc:
                s = t.get("subject", "")
                p = t.get("predicate", "")
                o = t.get("object", "")
                formatted_list.append([s, p, o])
                count += 1

            line = json.dumps(formatted_list, ensure_ascii=False)
            f.write(line + "\n")

    print(f"✅ Conversion completed!")
    print(f"   Total lines: {total_lines}")
    print(f"   Total triplets: {count}")
    print(f"   Output file: {output_txt_path}")


if __name__ == "__main__":
    INPUT_FILE = "ablation/description/webnlg/Qwen3-30B-A3B-Instruct-2507_few_shot_True/extracted_triplets4.json"
    OUTPUT_FILE = "ablation/description/webnlg/Qwen3-30B-A3B-Instruct-2507_few_shot_True/eval_prediction4.txt"

    # convert_flat_json_to_eval_format(INPUT_FILE, OUTPUT_FILE)
    convert_json_to_eval_format(INPUT_FILE, OUTPUT_FILE)
