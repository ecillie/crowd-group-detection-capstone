"""
@Author : Evan Cillie
@LastEdit : 05-28-26
@Purpose : Compare auto and manual YOLO + DeepSORT group results.
"""

import os
import re


MANUAL_FOLDER = "tagged_results/processed_results"
AUTO_FOLDER = "processed_results"
OUTPUT_FOLDER = "processed_results"


manual_files = [
    "people-walking-tagged-results.tex",
    "people-in-park-tagged-results.tex",
    "pier-walking-tagged-results.tex",
    "walk-in-park-tagged-results.tex",
    "wold-tagged-results.tex"
]


auto_files = [
    "people-walking.tex",
    "people-in-park-results.tex",
    "pier-walking-results.tex",
    "walk-in-park-results.tex",
    "wold-results.tex"
]


def fix_latex_text(text):
    """Fix characters that can mess up LaTeX."""
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")

    return text


def clean_cell(text):
    """Clean one cell from a LaTeX table."""
    text = text.strip()
    text = text.replace("\\textbf{", "")
    text = text.replace("}", "")
    text = text.replace("$", "")
    text = text.replace("\\%", "%")
    text = text.replace("\\_", "_")
    text = text.strip()

    return text


def time_to_seconds(time_text):
    """Convert a time like 0:04 into seconds."""
    parts = time_text.strip().split(":")

    minutes = int(parts[0])
    seconds = int(parts[1])

    total_seconds = minutes * 60 + seconds

    return total_seconds


def get_number(text):
    """Get the first number from a string."""
    match = re.search(r"-?\d+(\.\d+)?", text)

    if match:
        return float(match.group())

    return None


def is_group_row(first_cell):
    """Check if the row starts with a group label like G1."""
    first_cell = clean_cell(first_cell)

    if re.match(r"^G\d+$", first_cell):
        return True

    return False


def read_manual_file(file_path):
    """Read a manual tagged result file."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    durations = []

    for line in lines:
        line = line.strip()

        if "&" not in line:
            continue

        line = line.replace("\\\\", "").strip()

        parts = line.split("&")
        parts = [clean_cell(part) for part in parts]

        if len(parts) != 4:
            continue

        if not is_group_row(parts[0]):
            continue

        start_time = parts[1]
        end_time = parts[2]

        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)

        duration = end_seconds - start_seconds

        durations.append(duration)

    return durations


def read_auto_file(file_path):
    """Read an auto generated result file."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    durations = []
    duration_col = None

    for line in lines:
        line = line.strip()

        if "&" not in line:
            continue

        line = line.replace("\\\\", "").strip()

        parts = line.split("&")
        parts = [clean_cell(part) for part in parts]

        for i in range(len(parts)):
            if "duration" in parts[i].lower():
                duration_col = i

        if len(parts) == 0:
            continue

        if not is_group_row(parts[0]):
            continue

        duration = None

        if duration_col is not None and duration_col < len(parts):
            duration = get_number(parts[duration_col])

        if duration is None:
            numbers = []

            for part in parts[1:]:
                number = get_number(part)

                if number is not None:
                    numbers.append(number)

            if len(numbers) > 0:
                duration = numbers[-1]

        if duration is not None:
            durations.append(duration)

    return durations


def make_video_name(file_name):
    """Make the file name look nicer for the table."""
    name = os.path.splitext(os.path.basename(file_name))[0]

    name = name.replace("-tagged-results", "")
    name = name.replace("_tagged_results", "")
    name = name.replace("-results", "")
    name = name.replace("_results", "")
    name = name.replace("-group-statistics-table", "")
    name = name.replace("_group_statistics_table", "")

    name = name.replace("-", " ")
    name = name.replace("_", " ")

    return name.title()


def summarize_video(manual_file, auto_file):
    """Get the summary numbers for one video."""
    manual_path = os.path.join(MANUAL_FOLDER, manual_file)
    auto_path = os.path.join(AUTO_FOLDER, auto_file)

    manual_durations = read_manual_file(manual_path)
    auto_durations = read_auto_file(auto_path)

    manual_groups = len(manual_durations)
    auto_groups = len(auto_durations)

    manual_total = sum(manual_durations)
    auto_total = sum(auto_durations)

    if manual_groups > 0:
        manual_avg = manual_total / manual_groups
    else:
        manual_avg = 0

    if auto_groups > 0:
        auto_avg = auto_total / auto_groups
    else:
        auto_avg = 0

    if auto_groups > 0:
        group_percent = (manual_groups / auto_groups) * 100
    else:
        group_percent = 0

    summary = {
        "video": make_video_name(auto_file),
        "auto_groups": auto_groups,
        "manual_groups": manual_groups,
        "group_percent": group_percent,
        "auto_total": auto_total,
        "manual_total": manual_total,
        "auto_avg": auto_avg,
        "manual_avg": manual_avg
    }

    return summary


def make_latex_table(summaries):
    """Make the final LaTeX comparison table."""
    table = ""

    table += "\\begin{table}[H]\n"
    table += "    \\centering\n"
    table += "    \\caption{Comparison of Automatic and Manual Group Results Across Videos}\n"
    table += "    \\label{tab:auto-manual-group-comparison}\n"
    table += "    \\resizebox{\\textwidth}{!}{%\n"
    table += "    \\begin{tabular}{l r r r r r r r}\n"
    table += "        \\toprule\n"
    table += "        \\textbf{Video} & \\textbf{Auto Groups} & \\textbf{Manual Groups} & \\textbf{Manual/Auto (\\%)} & \\textbf{Auto Total (s)} & \\textbf{Manual Total (s)} & \\textbf{Auto Avg. (s)} & \\textbf{Manual Avg. (s)} \\\\\n"
    table += "        \\midrule\n"

    for summary in summaries:
        video = fix_latex_text(summary["video"])

        table += (
            f"        {video} & "
            f"{summary['auto_groups']} & "
            f"{summary['manual_groups']} & "
            f"{summary['group_percent']:.2f}\\% & "
            f"{summary['auto_total']:.2f} & "
            f"{summary['manual_total']:.2f} & "
            f"{summary['auto_avg']:.2f} & "
            f"{summary['manual_avg']:.2f} \\\\\n"
        )

    table += "        \\bottomrule\n"
    table += "    \\end{tabular}%\n"
    table += "    }\n"
    table += "\\end{table}\n"

    return table


def main():
    """Run the comparison script."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if len(manual_files) != len(auto_files):
        print("Error: manual files and auto files do not match.")
        return

    summaries = []

    for i in range(len(manual_files)):
        summary = summarize_video(manual_files[i], auto_files[i])
        summaries.append(summary)

    latex_table = make_latex_table(summaries)

    output_path = os.path.join(OUTPUT_FOLDER, "comparison_table.tex")

    with open(output_path, "w") as file:
        file.write(latex_table)


if __name__ == "__main__":
    main()