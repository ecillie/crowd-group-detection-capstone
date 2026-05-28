"""
@Author : Evan Cillie
@LastEdit : 05-28-26
@Purpose : Compare original and tagged YOLO + DeepSORT group result files by:
           - original groups seen
           - tagged groups seen
           - percent of groups tagged
           - original total duration
           - tagged total duration
           - original average duration
           - tagged average duration
"""

import os
import re


TAGGED_FOLDER = "tagged_results/processed_results"
NON_TAGGED_FOLDER = "processed_results"
OUTPUT_FOLDER = "processed_results"


TAGGED_FILES = [
    "people-walking-tagged-results.tex",
    "people-in-park-tagged-results.tex",
    "pier-walking-tagged-results.tex",
    "walk-in-park-tagged-results.tex",
    "wold-tagged-results.tex"
]


NON_TAGGED_FILES = [
    "people-walking.tex",
    "people-in-park-results.tex",
    "pier-walking-results.tex",
    "walk-in-park-results.tex",
    "wold-results.tex"
]


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


def clean_cell(text):
    """
    Clean a LaTeX table cell so it is easier to parse.
    """

    text = text.strip()
    text = text.replace("\\textbf{", "")
    text = text.replace("}", "")
    text = text.replace("$", "")
    text = text.replace("\\%", "%")
    text = text.replace("\\_", "_")
    text = text.strip()

    return text


def time_to_seconds(time_string):
    """
    Convert a time string like 0:04 or 1:23 into seconds.
    """

    parts = time_string.strip().split(":")

    minutes = int(parts[0])
    seconds = int(parts[1])

    return minutes * 60 + seconds


def get_number_from_text(text):
    """
    Get the first number from a piece of text.
    Works for values like:
    12
    12.5
    12.50 seconds
    """

    match = re.search(r"-?\d+(\.\d+)?", text)

    if match:
        return float(match.group())

    return None


def is_group_row(first_cell):
    """
    Checks whether a row is a real group row.
    Examples:
    G1
    G2
    G15
    """

    first_cell = clean_cell(first_cell)

    return re.match(r"^G\d+$", first_cell) is not None


def parse_tagged_latex_file(input_path):
    """
    Read one tagged LaTeX table file.

    Expected tagged row format:
    G1 & 0:00 & 0:04 & Center-Downward \\
    """

    with open(input_path, "r") as file:
        lines = file.readlines()

    durations = []

    for line in lines:
        line = line.strip()

        if "&" not in line:
            continue

        line = line.replace("\\\\", "").strip()
        parts = [clean_cell(item) for item in line.split("&")]

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


def parse_non_tagged_latex_file(input_path):
    """
    Read one original/non-tagged LaTeX result table.

    This counts group rows and tries to find the duration column by looking
    for a header cell containing the word 'duration'.

    Expected general row format:
    G1 & ... & duration value & ... \\
    """

    with open(input_path, "r") as file:
        lines = file.readlines()

    durations = []
    duration_index = None

    for line in lines:
        line = line.strip()

        if "&" not in line:
            continue

        line = line.replace("\\\\", "").strip()
        parts = [clean_cell(item) for item in line.split("&")]

        for index, item in enumerate(parts):
            if "duration" in item.lower():
                duration_index = index

        if len(parts) == 0 or not is_group_row(parts[0]):
            continue

        duration = None

        if duration_index is not None and duration_index < len(parts):
            duration = get_number_from_text(parts[duration_index])

        if duration is None:
            numbers = []

            for item in parts[1:]:
                number = get_number_from_text(item)

                if number is not None:
                    numbers.append(number)

            if numbers:
                duration = numbers[-1]

        if duration is not None:
            durations.append(duration)

    return durations


def clean_video_name(file_name):
    """
    Convert a file name into a cleaner video name for the LaTeX table.
    """

    video_name = os.path.splitext(os.path.basename(file_name))[0]

    video_name = video_name.replace("-tagged-results", "")
    video_name = video_name.replace("_tagged_results", "")
    video_name = video_name.replace("-results", "")
    video_name = video_name.replace("_results", "")
    video_name = video_name.replace("-group-statistics-table", "")
    video_name = video_name.replace("_group_statistics_table", "")
    video_name = video_name.replace("-", " ")
    video_name = video_name.replace("_", " ")

    return video_name.title()


def summarize_video(tagged_file, non_tagged_file):
    """
    Create summary statistics for one video.

    Original/non-tagged file gives:
    - original groups
    - original total duration
    - original average duration

    Tagged file gives:
    - tagged groups
    - tagged total duration
    - tagged average duration

    Percent is only calculated for groups:
    tagged groups / original groups * 100
    """

    tagged_path = os.path.join(TAGGED_FOLDER, tagged_file)
    non_tagged_path = os.path.join(NON_TAGGED_FOLDER, non_tagged_file)

    tagged_durations = parse_tagged_latex_file(tagged_path)
    original_durations = parse_non_tagged_latex_file(non_tagged_path)

    tagged_groups = len(tagged_durations)
    original_groups = len(original_durations)

    tagged_total_duration = sum(tagged_durations)
    original_total_duration = sum(original_durations)

    if tagged_groups > 0:
        tagged_average_duration = tagged_total_duration / tagged_groups
    else:
        tagged_average_duration = 0

    if original_groups > 0:
        original_average_duration = original_total_duration / original_groups
    else:
        original_average_duration = 0

    if original_groups > 0:
        groups_percent = (tagged_groups / original_groups) * 100
    else:
        groups_percent = 0

    summary = {
        "video": clean_video_name(non_tagged_file),
        "original_groups": original_groups,
        "tagged_groups": tagged_groups,
        "groups_percent": groups_percent,
        "original_total_duration": original_total_duration,
        "tagged_total_duration": tagged_total_duration,
        "original_average_duration": original_average_duration,
        "tagged_average_duration": tagged_average_duration
    }

    return summary


def make_comparison_latex(summaries):
    """
    Create a LaTeX comparison table from all video summaries.
    """

    latex = ""

    latex += "\\begin{table}[H]\n"
    latex += "    \\centering\n"
    latex += "    \\caption{Comparison of Original and Tagged Group Results Across Videos}\n"
    latex += "    \\label{tab:original-tagged-group-comparison}\n"
    latex += "    \\resizebox{\\textwidth}{!}{%\n"
    latex += "    \\begin{tabular}{l r r r r r r r}\n"
    latex += "        \\toprule\n"
    latex += "        \\textbf{Video} & \\textbf{Orig. Groups} & \\textbf{Tagged Groups} & \\textbf{Groups (\\%)} & \\textbf{Orig. Total (s)} & \\textbf{Tagged Total (s)} & \\textbf{Orig. Avg. (s)} & \\textbf{Tagged Avg. (s)} \\\\\n"
    latex += "        \\midrule\n"

    for summary in summaries:
        video = clean_latex(summary["video"])

        latex += (
            f"        {video} & "
            f"{summary['original_groups']} & "
            f"{summary['tagged_groups']} & "
            f"{summary['groups_percent']:.2f}\\% & "
            f"{summary['original_total_duration']:.2f} & "
            f"{summary['tagged_total_duration']:.2f} & "
            f"{summary['original_average_duration']:.2f} & "
            f"{summary['tagged_average_duration']:.2f} \\\\\n"
        )

    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}%\n"
    latex += "    }\n"
    latex += "\\end{table}\n"

    return latex


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if len(TAGGED_FILES) != len(NON_TAGGED_FILES):
        raise ValueError("must have the same number of files in each list.")

    summaries = []

    for tagged_file, non_tagged_file in zip(TAGGED_FILES, NON_TAGGED_FILES):
        summary = summarize_video(tagged_file, non_tagged_file)
        summaries.append(summary)

    latex_table = make_comparison_latex(summaries)

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "comparison_table.tex"
    )

    with open(output_file, "w") as file:
        file.write(latex_table)


if __name__ == "__main__":
    main()