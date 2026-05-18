"""
@Author : Evan Cillie
@LastEdit : 05-17-26
@Purpose : Read one or more YOLO + DeepSORT group tracking result files
           and create a LaTeX table of persistent group statistics.

Current Run Command :
python3 results.py people_in_park_results.txt people-walking.txt wold_results.txt pier_walking_results.txt walk_in_park_results.txt
"""

import os
import re
import sys
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


def create_output_folder():
    """
    Creates the output folder if it does not already exist.

    @return: None
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def get_file_base_name(file_path):
    """
    Gets the file name without its folder path or file extension.

    Example:
    people_in_park_results.txt -> people_in_park_results

    @param file_path: The path to the input tracking results file.
    @return: The base file name without the extension.
    """
    file_name = os.path.basename(file_path)
    file_base_name = os.path.splitext(file_name)[0]

    return file_base_name


def frame_to_seconds(frame_number):
    """
    Converts a frame number into seconds.

    @param frame_number: The frame number from the original video.
    @return: The timestamp in seconds.
    """
    return frame_number / FPS


def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.

    @param x1: The x-coordinate of the first point.
    @param y1: The y-coordinate of the first point.
    @param x2: The x-coordinate of the second point.
    @param y2: The y-coordinate of the second point.
    @return: The distance between the two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_group_overlap(ids1, ids2):
    """
    Calculates how much two groups overlap based on their member track IDs.

    The overlap is calculated using the smaller group size as the denominator.
    This makes matching more forgiving when a group gains or loses members.

    @param ids1: A set of track IDs from the first group.
    @param ids2: A set of track IDs from the second group.
    @return: A value between 0 and 1 representing group overlap.
    """
    if len(ids1) == 0 or len(ids2) == 0:
        return 0

    intersection = len(ids1.intersection(ids2))
    smaller_group_size = min(len(ids1), len(ids2))

    return intersection / smaller_group_size


def get_screen_region(x, y):
    """
    Converts a group's screen position into a readable screen region.

    Example:
    top-left, middle-center, bottom-right

    @param x: The group's x-coordinate.
    @param y: The group's y-coordinate.
    @return: A string describing the screen region.
    """
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
    """
    Classifies group movement based on average speed.

    @param avg_speed: The average speed of a persistent group.
    @return: The motion label for the group.
    """
    if avg_speed < STATIONARY_SPEED_THRESHOLD:
        return "STATIONARY"
    elif avg_speed < SLOW_MOVING_SPEED_THRESHOLD:
        return "SLOW MOVING"
    else:
        return "FAST MOVING"


def clean_latex_text(text):
    """
    Cleans text so it can be safely written into a LaTeX table.

    @param text: The text value to clean.
    @return: A cleaned text string.
    """
    return str(text).replace("_", "\\_")

def parse_member_ids(member_text):
    """
    Converts the member ID text from the results file into a list of integers.

    This handles both formats:
    [1, 2, 3]
    ['1', '2', '3']

    @param member_text: The text inside the Members brackets.
    @return: A list of integer member IDs.
    """
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
    """
    Reads a group-level tracking results file and converts it into a DataFrame.

    Expected group line format:
    Group 0 | Type: CROWD | Members: [1, 3, 7, 8] | Size: 4 |
    Center: (260.00, 275.00) | Avg Speed: 1.42

    @param file_path: The path to the input tracking results file.
    @return: A DataFrame where each row represents one group in one frame.
    """
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
                continue

            people_match = people_pattern.match(line)
            if people_match:
                current_people_tracked = int(people_match.group(1))
                continue

            groups_detected_match = groups_detected_pattern.match(line)
            if groups_detected_match:
                current_groups_detected = int(groups_detected_match.group(1))
                continue

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
    """
    Links frame-level groups across time to create persistent group IDs.

    A group can be matched by member ID overlap or by nearby screen location.
    This prevents the same real-world group from becoming a new group every frame.

    @param group_df: A DataFrame containing group detections from each frame.
    @return: The same DataFrame with a persistent_group_id column added.
    """
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

            center_distance = calculate_distance(
                current_center[0],
                current_center[1],
                previous_center[0],
                previous_center[1]
            )

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
    """
    Builds the final persistent group summary table.

    Each row represents one persistent group across multiple frames.

    @param group_df: A DataFrame containing frame-level groups with persistent IDs.
    @return: A DataFrame where each row summarizes one persistent group.
    """
    lifetime_rows = []

    for persistent_id, data in group_df.groupby("persistent_group_id"):
        data = data.sort_values("frame")

        first_frame = data["frame"].min()
        last_frame = data["frame"].max()

        start_time = frame_to_seconds(first_frame)
        end_time = frame_to_seconds(last_frame)
        duration_seconds = end_time - start_time

        main_type = Counter(data["type"].tolist()).most_common(1)[0][0]
        main_region = Counter(data["screen_region"].tolist()).most_common(1)[0][0]

        average_size = data["group_size"].mean()
        max_size = data["group_size"].max()
        average_speed = data["avg_speed"].mean()
        motion_type = classify_motion(average_speed)

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
    """
    Saves a LaTeX table with the main persistent group statistics.

    @param group_lifetimes_df: A DataFrame containing persistent group summaries.
    @param file_base_name: The base name of the input file.
    @return: None
    """


    latex_table_df = group_lifetimes_df[[
        "Group ID",
        "Type",
        "Average Size",
        "Max Size",
        "Location",
        "Start Time (s)",
        "End Time (s)",
        "Duration (s)",
        "Motion",
        "Members Seen"
    ]].copy()

    latex_table_df = latex_table_df.sort_values("Group ID")

    latex_table_df["Type"] = latex_table_df["Type"].apply(clean_latex_text)
    latex_table_df["Location"] = latex_table_df["Location"].apply(clean_latex_text)
    latex_table_df["Motion"] = latex_table_df["Motion"].apply(clean_latex_text)
    latex_table_df["Members Seen"] = latex_table_df["Members Seen"].apply(clean_latex_text)

    latex_code = latex_table_df.to_latex(
        index=False,
        escape=False,
        caption=f"Detected Persistent Group Statistics for {file_base_name}",
        label=f"tab:{file_base_name}_group_statistics"
    )

    table_path = f"{OUTPUT_FOLDER}/{file_base_name}_group_statistics_table.tex"

    with open(table_path, "w") as file:
        file.write(latex_code)


def process_one_file(input_file):
    """
    Runs the full results pipeline for one group tracking file.

    This function parses the file, links groups across frames, builds the
    persistent group table, and saves the LaTeX output.

    @param input_file: The path to one tracking results file.
    @return: None
    """
    file_base_name = get_file_base_name(input_file)

    group_df = parse_group_tracking_file(input_file)

    if len(group_df) == 0:
        return

    group_df = assign_persistent_group_ids(group_df)

    group_lifetimes_df = build_group_lifetimes(group_df)
    group_lifetimes_df = group_lifetimes_df[
    group_lifetimes_df["Duration (s)"] >= MIN_DURATION
    ]

    if len(group_lifetimes_df) == 0:
        return

    save_latex_group_table(group_lifetimes_df, file_base_name)


def main(input_files):
    """
    Processes multiple group tracking files.

    @param input_files: A list of tracking result file paths.
    @return: None
    """
    create_output_folder()

    for input_file in input_files:
        if not os.path.exists(input_file):
            continue

        process_one_file(input_file)


if __name__ == "__main__":
    input_files = sys.argv[1:]

    if len(input_files) > 0:
        main(input_files)