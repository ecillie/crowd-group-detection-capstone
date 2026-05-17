"""
@Author : Evan Cillie
@LastEdit : 05-17-26
@Purpose : Read one or more YOLO + DeepSORT tracking results files and create:
           1. a LaTeX table of detected persistent groups
           2. a labeled arrow map showing group movement

Run:
python3 results.py people_in_park_results.txt people-walking.txt wold_results.txt
"""

import os
import re
import sys
import math
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_FOLDER = "processed_results"

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

FPS = 30
PROCESS_EVERY_N_FRAMES = 2

GROUP_DISTANCE_EPS = 70
MIN_GROUP_SIZE = 2

GROUP_MATCH_THRESHOLD = 0.35
MAX_FRAME_GAP = 10
LOCATION_MATCH_DISTANCE = 100

MAX_REASONABLE_DEPTH = 30

STATIONARY_SPEED_THRESHOLD = 1.5
SLOW_MOVING_SPEED_THRESHOLD = 5.0


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


def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two screen points.

    @param x1: The x-coordinate of the first point.
    @param y1: The y-coordinate of the first point.
    @param x2: The x-coordinate of the second point.
    @param y2: The y-coordinate of the second point.
    @return: The distance between the two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_screen_region(x, y):
    """
    Converts a screen coordinate into a readable screen region.

    Example regions:
    top-left, middle-center, bottom-right

    @param x: The x-coordinate of the group center.
    @param y: The y-coordinate of the group center.
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
    Classifies a group based on its average pixel speed.

    @param avg_speed: The average speed of the group in pixels per frame.
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
    Cleans text so it is safer to use in a LaTeX table.

    @param text: The text value that will be written to LaTeX.
    @return: A cleaned version of the text.
    """
    return str(text).replace("_", "\\_")


def calculate_group_overlap(ids1, ids2):
    """
    Calculates how much two groups overlap based on their tracked IDs.

    The overlap is calculated using the smaller group size as the denominator.
    This helps compare groups even if one group gains or loses members.

    @param ids1: A set of track IDs from the first group.
    @param ids2: A set of track IDs from the second group.
    @return: A value between 0 and 1 representing group overlap.
    """
    if len(ids1) == 0 or len(ids2) == 0:
        return 0

    intersection = len(ids1.intersection(ids2))
    smaller_group_size = min(len(ids1), len(ids2))

    return intersection / smaller_group_size


def parse_tracking_file(file_path):
    """
    Reads a YOLO + DeepSORT tracking results file and converts it into a DataFrame.

    Expected person line format:
    ID 1 | Center: (x, y) | Speed: 2.1 px/frame | Depth: 5.0 | Ground Y: 320

    @param file_path: The path to the input tracking results file.
    @return: A pandas DataFrame where each row represents one tracked person in one frame.
    """
    people_rows = []

    current_frame = None
    current_formation = None
    current_people_tracked = None

    frame_pattern = re.compile(r"Frame (\d+)")
    formation_pattern = re.compile(r"Formation: (.+)")
    people_pattern = re.compile(r"People tracked: (\d+)")

    person_pattern = re.compile(
        r"ID (\d+) \| "
        r"Center: \((\d+), (\d+)\) \| "
        r"Speed: ([\d.]+) px/frame \| "
        r"Depth: ([\d.]+) \| "
        r"Ground Y: (\d+)"
    )

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            frame_match = frame_pattern.match(line)
            if frame_match:
                current_frame = int(frame_match.group(1))
                continue

            formation_match = formation_pattern.match(line)
            if formation_match:
                current_formation = formation_match.group(1)
                continue

            people_match = people_pattern.match(line)
            if people_match:
                current_people_tracked = int(people_match.group(1))
                continue

            person_match = person_pattern.match(line)
            if person_match:
                track_id = int(person_match.group(1))
                center_x = int(person_match.group(2))
                center_y = int(person_match.group(3))
                speed = float(person_match.group(4))
                depth = float(person_match.group(5))
                ground_y = int(person_match.group(6))

                clean_depth = min(depth, MAX_REASONABLE_DEPTH)

                people_rows.append({
                    "frame": current_frame,
                    "formation": current_formation,
                    "people_tracked": current_people_tracked,
                    "track_id": track_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "speed": speed,
                    "depth": depth,
                    "clean_depth": clean_depth,
                    "ground_y": ground_y
                })

    return pd.DataFrame(people_rows)


def cluster_single_frame(frame_data):
    """
    Groups people within one frame based on distance.

    If person A is close to person B, and person B is close to person C,
    then all three people are placed into the same group.

    People who are not close enough to anyone else are labeled as -1.

    @param frame_data: A DataFrame containing all tracked people in one frame.
    @return: A list of group labels for the people in that frame.
    """
    points = frame_data[["center_x", "ground_y"]].values.tolist()

    labels = [-1] * len(points)
    used = set()
    next_group_id = 0

    for i in range(len(points)):
        if i in used:
            continue

        current_group = [i]
        used.add(i)

        group_index = 0

        while group_index < len(current_group):
            member_index = current_group[group_index]
            x1, y1 = points[member_index]

            for j in range(len(points)):
                if j in used:
                    continue

                x2, y2 = points[j]
                distance = calculate_distance(x1, y1, x2, y2)

                if distance <= GROUP_DISTANCE_EPS:
                    current_group.append(j)
                    used.add(j)

            group_index += 1

        if len(current_group) >= MIN_GROUP_SIZE:
            for member_index in current_group:
                labels[member_index] = next_group_id

            next_group_id += 1

    return labels


def cluster_groups_by_frame(people_df):
    """
    Applies group clustering to every frame in the tracking data.

    @param people_df: A DataFrame containing all tracked people from the input file.
    @return: A DataFrame with an added spatial_group_id column.
    """
    grouped_people = []

    for frame, frame_data in people_df.groupby("frame"):
        frame_data = frame_data.copy()

        if len(frame_data) == 0:
            continue

        labels = cluster_single_frame(frame_data)

        frame_data["spatial_group_id"] = labels
        grouped_people.append(frame_data)

    if len(grouped_people) == 0:
        return pd.DataFrame()

    return pd.concat(grouped_people, ignore_index=True)


def build_group_summary(grouped_people_df):
    """
    Builds a frame-level group summary.

    Each row in the returned DataFrame represents one detected group in one frame.

    @param grouped_people_df: A DataFrame containing tracked people and their spatial group IDs.
    @return: A DataFrame summarizing each detected group in each frame.
    """
    group_rows = []

    valid_groups = grouped_people_df[grouped_people_df["spatial_group_id"] != -1]

    for (frame, group_id), group_data in valid_groups.groupby(["frame", "spatial_group_id"]):
        member_ids = sorted(group_data["track_id"].unique().tolist())
        group_size = len(member_ids)

        center_x = group_data["center_x"].mean()
        center_y = group_data["center_y"].mean()
        ground_y = group_data["ground_y"].mean()

        avg_speed = group_data["speed"].mean()
        max_speed = group_data["speed"].max()
        avg_depth = group_data["clean_depth"].mean()

        x_spread = group_data["center_x"].max() - group_data["center_x"].min()
        y_spread = group_data["center_y"].max() - group_data["center_y"].min()

        screen_region = get_screen_region(center_x, center_y)
        motion_type = classify_motion(avg_speed)

        if group_size >= 4:
            spatial_type = "CROWD"
        else:
            spatial_type = "SMALL GROUP"

        if group_size >= 3:
            if x_spread < 60 and y_spread > 60:
                spatial_type = "VERTICAL LINE"
            elif y_spread < 60 and x_spread > 60:
                spatial_type = "HORIZONTAL LINE"

        group_rows.append({
            "frame": frame,
            "spatial_group_id": group_id,
            "group_size": group_size,
            "center_x": round(center_x, 2),
            "center_y": round(center_y, 2),
            "ground_y": round(ground_y, 2),
            "screen_region": screen_region,
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "motion_type": motion_type,
            "avg_depth": round(avg_depth, 2),
            "x_spread": round(x_spread, 2),
            "y_spread": round(y_spread, 2),
            "spatial_type": spatial_type,
            "member_ids": member_ids
        })

    return pd.DataFrame(group_rows)


def assign_persistent_group_ids(group_summary_df):
    """
    Links frame-level groups across time using track ID overlap and screen location.

    This prevents the same real-world group from becoming a new group every frame.

    @param group_summary_df: A DataFrame containing one row per group per frame.
    @return: The same DataFrame with an added persistent_group_id column.
    """
    group_summary_df = group_summary_df.copy()
    group_summary_df["persistent_group_id"] = None

    active_groups = {}
    next_persistent_id = 1

    sorted_df = group_summary_df.sort_values("frame")

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

        group_summary_df.at[index, "persistent_group_id"] = best_match_id

        active_groups[best_match_id] = {
            "frame": current_frame,
            "member_ids": current_ids,
            "center": current_center
        }

    group_summary_df["persistent_group_id"] = group_summary_df["persistent_group_id"].astype(int)

    return group_summary_df


def build_group_lifetimes(group_summary_df):
    """
    Builds the final persistent group lifetime table.

    Each row represents one persistent group across multiple frames.

    @param group_summary_df: A DataFrame containing frame-level groups with persistent group IDs.
    @return: A DataFrame where each row summarizes one persistent group.
    """
    lifetime_rows = []

    for persistent_id, data in group_summary_df.groupby("persistent_group_id"):
        data = data.sort_values("frame")

        first_frame = data["frame"].min()
        last_frame = data["frame"].max()

        duration_frames = last_frame - first_frame + PROCESS_EVERY_N_FRAMES
        duration_seconds = duration_frames / FPS

        main_region = Counter(data["screen_region"].tolist()).most_common(1)[0][0]
        main_motion_type = Counter(data["motion_type"].tolist()).most_common(1)[0][0]
        main_spatial_type = Counter(data["spatial_type"].tolist()).most_common(1)[0][0]

        start_row = data.iloc[0]
        end_row = data.iloc[-1]

        all_member_ids = set()
        for member_list in data["member_ids"]:
            all_member_ids.update(member_list)

        lifetime_rows.append({
            "Group ID": f"G{persistent_id}",
            "persistent_group_id": persistent_id,
            "Size": round(data["group_size"].mean(), 2),
            "Max Size": data["group_size"].max(),
            "Location": main_region,
            "Duration (s)": round(duration_seconds, 2),
            "Type": main_spatial_type,
            "Motion": main_motion_type,
            "Average X": round(data["center_x"].mean(), 2),
            "Average Y": round(data["center_y"].mean(), 2),
            "Start X": round(start_row["center_x"], 2),
            "Start Y": round(start_row["center_y"], 2),
            "End X": round(end_row["center_x"], 2),
            "End Y": round(end_row["center_y"], 2),
            "First Frame": first_frame,
            "Last Frame": last_frame,
            "Frames Seen": data["frame"].nunique(),
            "Average Speed": round(data["avg_speed"].mean(), 2),
            "Member IDs Seen": str(sorted(all_member_ids))
        })

    return pd.DataFrame(lifetime_rows)


def save_latex_group_table(group_lifetimes_df, file_base_name):
    """
    Saves a LaTeX table with the most important persistent group statistics.

    @param group_lifetimes_df: A DataFrame containing persistent group lifetime summaries.
    @param file_base_name: The base name of the input file.
    @return: None
    """
    latex_table_df = group_lifetimes_df[[
        "Group ID",
        "Size",
        "Location",
        "Duration (s)",
        "Type"
    ]].copy()

    latex_table_df = latex_table_df.sort_values("Group ID")

    latex_table_df["Location"] = latex_table_df["Location"].apply(clean_latex_text)
    latex_table_df["Type"] = latex_table_df["Type"].apply(clean_latex_text)

    latex_code = latex_table_df.to_latex(
        index=False,
        escape=False,
        caption=f"Detected Persistent Group Statistics for {file_base_name}",
        label=f"tab:{file_base_name}_group_statistics"
    )

    table_path = f"{OUTPUT_FOLDER}/{file_base_name}_group_statistics_table.tex"

    with open(table_path, "w") as file:
        file.write(latex_code)


def save_group_arrow_plot(group_lifetimes_df, file_base_name):
    """
    Saves a labeled movement map for persistent groups.

    Black point = group start position.
    Colored arrow/circle = group end position.
    Color = motion category.

    @param group_lifetimes_df: A DataFrame containing persistent group lifetime summaries.
    @param file_base_name: The base name of the input file.
    @return: None
    """
    if len(group_lifetimes_df) == 0:
        return

    plt.figure(figsize=(10, 6))

    motion_colors = {
        "STATIONARY": "blue",
        "SLOW MOVING": "orange",
        "FAST MOVING": "red"
    }

    for _, row in group_lifetimes_df.iterrows():
        group_id = row["Group ID"]
        size = row["Size"]
        motion_type = row["Motion"]

        start_x = row["Start X"]
        start_y = row["Start Y"]
        end_x = row["End X"]
        end_y = row["End Y"]

        color = motion_colors.get(motion_type, "gray")

        dx = end_x - start_x
        dy = end_y - start_y

        if abs(dx) < 2 and abs(dy) < 2:
            end_x = min(start_x + 8, FRAME_WIDTH - 5)
            end_y = start_y

        arrow_width = max(3, size * 1.5)
        end_circle_size = max(120, size * 140)
        start_circle_size = 45

        plt.scatter(
            start_x,
            start_y,
            c="black",
            s=start_circle_size,
            marker="o",
            alpha=0.9,
            zorder=3
        )

        plt.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=arrow_width,
                mutation_scale=26,
                alpha=0.85
            ),
            zorder=2
        )

        plt.scatter(
            end_x,
            end_y,
            s=end_circle_size,
            c=color,
            alpha=0.80,
            edgecolors="black",
            zorder=4
        )

        start_label_x = min(max(start_x + 5, 5), FRAME_WIDTH - 25)
        start_label_y = min(max(start_y - 5, 12), FRAME_HEIGHT - 10)

        end_label_x = min(max(end_x + 5, 5), FRAME_WIDTH - 25)
        end_label_y = min(max(end_y + 5, 12), FRAME_HEIGHT - 10)

        plt.text(
            start_label_x,
            start_label_y,
            group_id,
            fontsize=8,
            color="black",
            weight="bold",
            zorder=5
        )

        plt.text(
            end_label_x,
            end_label_y,
            group_id,
            fontsize=9,
            color=color,
            weight="bold",
            zorder=5
        )

    plt.xlim(0, FRAME_WIDTH)
    plt.ylim(FRAME_HEIGHT, 0)

    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.title(f"Persistent Group Movement Map: {file_base_name}")

    plt.grid(True)

    plt.scatter([], [], c="black", s=45, marker="o", label="Start Point")
    plt.scatter([], [], c="blue", s=120, label="Stationary")
    plt.scatter([], [], c="orange", s=120, label="Slow Moving")
    plt.scatter([], [], c="red", s=120, label="Fast Moving")

    plt.legend()

    plot_path = f"{OUTPUT_FOLDER}/{file_base_name}_persistent_group_map.png"

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_one_file(input_file):
    """
    Runs the full analysis pipeline for one tracking results file.

    This function parses the file, clusters people into groups, links groups
    across frames, and saves the LaTeX table and movement plot.

    @param input_file: The path to one tracking results file.
    @return: None
    """
    file_base_name = get_file_base_name(input_file)

    people_df = parse_tracking_file(input_file)

    if len(people_df) == 0:
        return

    grouped_people_df = cluster_groups_by_frame(people_df)

    if len(grouped_people_df) == 0:
        return

    group_summary_df = build_group_summary(grouped_people_df)

    if len(group_summary_df) == 0:
        return

    group_summary_df = assign_persistent_group_ids(group_summary_df)

    group_lifetimes_df = build_group_lifetimes(group_summary_df)

    save_latex_group_table(group_lifetimes_df, file_base_name)
    save_group_arrow_plot(group_lifetimes_df, file_base_name)


def main(input_files):
    """
    Runs the group analysis pipeline for multiple input files.

    @param input_files: A list of tracking results file paths.
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