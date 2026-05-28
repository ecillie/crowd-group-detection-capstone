"""
@Author : Evan Cillie
@LastEdit : 05-28-26
@Purpose : Read one or more YOLO + DeepSORT group tracking result files
           and create one LaTeX table file for each input file.

Current Run Command:
python3 tagged_results.py people-walking-tagged-results.txt people-in-park-tagged-results.txt pier-walking-tagged-results.txt walk-in-park-tagged-results.txt wold-tagged-results.txt
"""

import argparse
import os

OUTPUT_FOLDER = "processed_results"

def clean_latex(text):
    """Escape characters that can break LaTeX."""
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def make_latex_table(input_file, table_number):
    with open(input_file, "r") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    headers = [clean_latex(h.strip()) for h in lines[0].split("|")]

    rows = []
    for line in lines[1:]:
        row = [clean_latex(item.strip()) for item in line.split("|")]
        rows.append(row)

    file_name = os.path.splitext(os.path.basename(input_file))[0]
    caption_name = clean_latex(file_name.replace("_", " "))

    latex = ""
    latex += "\\begin{table}[H]\n"
    latex += "    \\centering\n"
    latex += f"    \\caption{{Detected Group Summary: {caption_name}}}\n"
    latex += f"    \\label{{tab:group-summary-{table_number}}}\n"
    latex += "    \\begin{tabular}{c c c l}\n"
    latex += "        \\toprule\n"

    latex += "        "
    latex += " & ".join([f"\\textbf{{{h}}}" for h in headers])
    latex += " \\\\\n"

    latex += "        \\midrule\n"

    for row in rows:
        latex += "        "
        latex += " & ".join(row)
        latex += " \\\\\n"

    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex

    
def main():
    parser = argparse.ArgumentParser(
        description="Convert tagged YOLO + DeepSORT result text files into LaTeX tables."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Input text files to convert into LaTeX tables"
    )

    args = parser.parse_args()

    for i, input_file in enumerate(args.files, start=1):
        latex_table = make_latex_table(input_file, i)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join("processed_results", base_name + ".tex")

        with open(output_file, "w") as file:
            file.write(latex_table)



if __name__ == "__main__":
    main()