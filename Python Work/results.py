"""
@Author : Evan Cillie
@LastEdit : 05-17-26
@Purpose : Read YOLO + DeepSORT result files and make LaTeX tables.

Run Example:
python3 results.py people-in-park-results.txt people-walking.txt wold-results.txt pier-walking-results.txt walk-in-park-results.txt
"""

import os
import re
import math
from collections import Counter

import pandas as pd


OUTPUT_FOLDER = "processed_results"

FPS = 30

GROUP_MATCH_THRESHOLD = 0.35
MAX_FRAME_GAP = 10
LOCATION_MATCH_DISTANCE = 100

STATIONARY_SPEED_THRESHOLD = 1.5
SLOW_MOVING_SPEED_THRESHOLD = 5.0

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

MIN_DURATION = 2.0


def get_file_base_name(file_path):
    """Get the file name without the path or extension."""
    file_name = os.path.basename(file_path)
    file_base_name = os.path.splitext(file_name)[0]

    return file_base_name


def frame_to_seconds(frame_number):
    """Convert a frame number into seconds."""
    return frame_number / FPS


def calculate_distance(x1, y1, x2, y2):
    """Find the distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_group_overlap(ids1, ids2):
    """Check how much two groups overlap by member IDs."""
    if len(ids1) == 0 or len(ids2) == 0:
        return 0

    intersection = len(ids1.intersection(ids2))
    smaller_group_size = min(len(ids1), len(ids2))

    return intersection / smaller_group_size


def get_screen_region(x, y):
    """Turn an x and y position into a screen location."""
    if x < FRAME_WIDTH / 3:
        horizontal = "left"
    elif x < 2 * FRAME_WIDTH / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if y < FRAME_HEIGHT / 3:
        vertical = "top"
    elif y < 2 * FRAME_HEIGHT / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    return f"{vertical}-{horizontal}"


def classify_motion(avg_speed):
    """Label the group as stationary, slow moving, or fast moving."""
    if avg_speed < STATIONARY_SPEED_THRESHOLD:
        return "Stationary"
    elif avg_speed < SLOW_MOVING_SPEED_THRESHOLD:
        return "Slow Moving"
    else:
        return "Fast Moving"


def clean_latex_text(text):
    """Clean text before putting it into LaTeX."""
    return str(text).replace("_", "\\_")


def parse_member_ids(member_text):
    """Turn the member ID text into a list of numbers."""
    member_text = member_text.strip()

    if member_text == "":
        return []

    member_ids = []

    for value in member_text.split(","):
        value = value.strip()
        value = value.replace("'", "")
        value = value.replace('"', "")

        if value != "":
            member_ids.append(int(value))

    return member_ids


def parse_group_tracking_file(file_path):
    """Read one tracking result file into a DataFrame."""
    group_rows = []

    current_frame = None
    current_people_tracked = None
    current_groups_detected = None

    frame_pattern = re.compile(r"Frame (\d+)")
    people_pattern = re.compile(r"People tracked: (\d+)")
    groups_detected_pattern = re.compile(r"Groups detected: (\d+)")

    group_pattern = re.compile(
        r"Group (\d+) \| "
        r"Type: (.+?) \| "
        r"Members: \[(.*?)\] \| "
        r"Size: (\d+) \| "
        r"Center: \(([-\d.]+), ([-\d.]+)\) \| "
        r"Avg Speed: ([-\d.]+)"
    )

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            frame_match = frame_pattern.match(line)
            if frame_match:
                current_frame = int(frame_match.group(1))
            people_match = people_pattern.match(line)
            if people_match:
                current_people_tracked = int(people_match.group(1))
            groups_detected_match = groups_detected_pattern.match(line)
            if groups_detected_match:
                current_groups_detected = int(groups_detected_match.group(1))
            group_match = group_pattern.match(line)
            if group_match:
                frame_group_id = int(group_match.group(1))
                group_type = group_match.group(2)
                member_ids = parse_member_ids(group_match.group(3))
                group_size = int(group_match.group(4))
                center_x = float(group_match.group(5))
                center_y = float(group_match.group(6))
                avg_speed = float(group_match.group(7))

                group_rows.append({
                    "frame": current_frame,
                    "time_seconds": frame_to_seconds(current_frame),
                    "people_tracked": current_people_tracked,
                    "groups_detected": current_groups_detected,
                    "frame_group_id": frame_group_id,
                    "type": group_type,
                    "member_ids": member_ids,
                    "group_size": group_size,
                    "center_x": center_x,
                    "center_y": center_y,
                    "avg_speed": avg_speed,
                    "screen_region": get_screen_region(center_x, center_y)
                })
    return pd.DataFrame(group_rows)


def assign_persistent_group_ids(group_df):
    """Connect groups across frames using IDs and location."""
    group_df = group_df.copy()
    group_df["persistent_group_id"] = None
    active_groups = {}
    next_persistent_id = 1
    sorted_df = group_df.sort_values("frame")

    for index, row in sorted_df.iterrows():
        current_frame = row["frame"]
        current_ids = set(row["member_ids"])
        current_center = (row["center_x"], row["center_y"])
        best_match_id = None
        best_match_score = 0
        for persistent_id, previous_group in active_groups.items():
            previous_frame = previous_group["frame"]
            if current_frame - previous_frame > MAX_FRAME_GAP:
                continue
            previous_ids = previous_group["member_ids"]
            previous_center = previous_group["center"]
            id_overlap = calculate_group_overlap(current_ids, previous_ids)
            center_distance = calculate_distance(current_center[0], current_center[1], previous_center[0], previous_center[1])
            if id_overlap >= GROUP_MATCH_THRESHOLD:
                score = 2.0 + id_overlap
            elif center_distance < LOCATION_MATCH_DISTANCE:
                score = 1.0 - (center_distance / LOCATION_MATCH_DISTANCE)
            else:
                score = 0
            if score > best_match_score:
                best_match_score = score
                best_match_id = persistent_id
        if best_match_id is None:
            best_match_id = next_persistent_id
            next_persistent_id += 1
        group_df.at[index, "persistent_group_id"] = best_match_id
        active_groups[best_match_id] = {
            "frame": current_frame,
            "member_ids": current_ids,
            "center": current_center
        }
    group_df["persistent_group_id"] = group_df["persistent_group_id"].astype(int)
    return group_df

def build_group_lifetimes(group_df):
    """Build the final summary for each persistent group."""
    lifetime_rows = []

    for persistent_id, data in group_df.groupby("persistent_group_id"):
        data = data.sort_values("frame")
        first_frame = data["frame"].min()
        last_frame = data["frame"].max()
        start_time = frame_to_seconds(first_frame)
        end_time = frame_to_seconds(last_frame)
        duration_seconds = end_time - start_time
        main_region = Counter(data["screen_region"].tolist()).most_common(1)[0][0]
        average_size = data["group_size"].mean()
        max_size = data["group_size"].max()
        average_speed = data["avg_speed"].mean()
        motion_type = classify_motion(average_speed)
        type_counts = Counter(data["type"].tolist())
        if max_size == 1:
            if "PASSERBY" in type_counts:
                main_type = "Passerby"
            else:
                main_type = "Single Person"
        elif "LINE" in type_counts:
            main_type = "Line"
        elif max_size >= 4:
            main_type = "Crowd"
        else:
            main_type = "Small Group"
        all_member_ids = set()
        for member_list in data["member_ids"]:
            all_member_ids.update(member_list)
        lifetime_rows.append({
            "Group ID": f"G{persistent_id}",
            "persistent_group_id": persistent_id,
            "Type": main_type,
            "Average Size": round(average_size, 2),
            "Max Size": max_size,
            "Location": main_region,
            "Start Time (s)": round(start_time, 2),
            "End Time (s)": round(end_time, 2),
            "Duration (s)": round(duration_seconds, 2),
            "Motion": motion_type,
            "Average Speed": round(average_speed, 2),
            "Members Seen": str(sorted(all_member_ids)),
            "First Frame": first_frame,
            "Last Frame": last_frame,
            "Frames Seen": data["frame"].nunique()
        })
    return pd.DataFrame(lifetime_rows)


def save_latex_group_table(group_lifetimes_df, file_base_name):
    """Save the group summary as a LaTeX table."""
    latex_table_df = group_lifetimes_df[[
        "Group ID",
        "Type",
        "Average Size",
        "Max Size",
        "Location",
        "Start Time (s)",
        "End Time (s)",
        "Duration (s)",
        "Motion"
    ]].copy()
    latex_table_df = latex_table_df.sort_values("Group ID")
    latex_table_df["Type"] = latex_table_df["Type"].apply(clean_latex_text)
    latex_table_df["Location"] = latex_table_df["Location"].apply(clean_latex_text)
    latex_table_df["Motion"] = latex_table_df["Motion"].apply(clean_latex_text)
    latex_code = latex_table_df.to_latex(index=False,escape=False,caption=f"Detected Persistent Group Statistics for {file_base_name}",label=f"tab:{file_base_name}_group_statistics")
    latex_code = latex_code.replace("\\begin{tabular}","\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}")
    latex_code = latex_code.replace("\\end{tabular}","\\end{tabular}%\n}")
    table_path = f"{OUTPUT_FOLDER}/{file_base_name}.tex"
    with open(table_path, "w") as file:
        file.write(latex_code)


def process_one_file(input_file):
    """Run the full process for one result file."""
    file_base_name = get_file_base_name(input_file)
    group_df = parse_group_tracking_file(input_file)
    group_df = assign_persistent_group_ids(group_df)
    group_lifetimes_df = build_group_lifetimes(group_df)
    group_lifetimes_df = group_lifetimes_df[group_lifetimes_df["Duration (s)"] >= MIN_DURATION]
    save_latex_group_table(group_lifetimes_df, file_base_name)


def main(input_files):
    """Process all input files."""
    for input_file in input_files:
        process_one_file(input_file)


if __name__ == "__main__":
    input_files = [
        "people-in-park-results.txt",
        "people-walking.txt",
        "wold-results.txt",
        "pier-walking-results.txt",
        "walk-in-park-results.txt"
    ]

    main(input_files)