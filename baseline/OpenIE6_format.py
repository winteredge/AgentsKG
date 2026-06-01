import re
import sys


def convert_to_carb_tabbed(input_file, output_file):
    """
    Sentence [TAB] Confidence [TAB] Relation [TAB] Subject [TAB] Object
    """
    print(f"OpenIE 6 files: {input_file} ...")

    count = 0
    current_sentence = None

    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"^(\d+\.\d+):\s*\((.*)\)$", line)

            if match:
                if current_sentence is None:
                    continue

                confidence = match.group(1)
                content = match.group(2)

                parts = [p.strip() for p in content.split(";")]
                if len(parts) >= 3:
                    subj = parts[0]
                    rel = parts[1]
                    obj = parts[2]

                    f_out.write(
                        f"{current_sentence}\t{confidence}\t{rel}\t{subj}\t{obj}\n"
                    )
                    count += 1
            else:
                current_sentence = line

    print(f"Conversion completed successfully! Generated {count} triplets.")
    print(f"Output file: {output_file}")
    print("You can now run the evaluation with benchmark.py --tabbed.")

INPUT_FILE = "output\openie6\webnlg_for_eval.txt"
OUTPUT_FILE = "output\openie6\webnlg_for_eval.tsv"

if __name__ == "__main__":
    convert_to_carb_tabbed(INPUT_FILE, OUTPUT_FILE)
