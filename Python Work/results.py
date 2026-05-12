"""
@Author : Evan Cillie
@LastEdit : 05-12-26
@Purpose : Read tracking_results.txt and create:
           1. a LaTeX table of detected persistent groups
           2. a labeled arrow map showing where groups moved
"""

import os
import re
import math
import ast
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Files / constants
# -----------------------------

INPUT_FILE = "tracking_results.txt"
OUTPUT_FOLDER = "processed_results"

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 30
PROCESS_EVERY_N_FRAMES = 2

# Spatial grouping settings
GROUP_DISTANCE_EPS = 85
MIN_GROUP_SIZE = 2

# Persistent group matching settings
GROUP_MATCH_THRESHOLD = 0.40
MAX_FRAME_GAP = 6
LOCATION_MATCH_DISTANCE = 90

# Depth cleanup
MAX_REASONABLE_DEPTH = 30

# Motion classification thresholds
STATIONARY_SPEED_THRESHOLD = 1.0
SLOW_MOVING_SPEED_THRESHOLD = 3.0


# -----------------------------
# Helper functions
# -----------------------------

def create_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_screen_region(x, y):
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
    if avg_speed < STATIONARY_SPEED_THRESHOLD:
        return "STATIONARY"
    elif avg_speed < SLOW_MOVING_SPEED_THRESHOLD:
        return "SLOW MOVING"
    else:
        return "FAST MOVING"


def clean_latex_text(text):
    return str(text).replace("_", "\\_")


# -----------------------------
# Step 1: Parse tracking_results.txt
# -----------------------------

def parse_tracking_file(file_path):
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


# -----------------------------
# Step 2: Spatial grouping
# -----------------------------

def distance_between_people(person1, person2):
    """
    Uses center_x and ground_y.

    center_x = left/right screen position.
    ground_y = approximate standing position on screen.
    """
    return calculate_distance(
        person1["center_x"],
        person1["ground_y"],
        person2["center_x"],
        person2["ground_y"]
    )


def cluster_single_frame(frame_data):
    """
    Custom grouping algorithm without sklearn.

    People are grouped if they are close to any member of the group.
    Speed is not used here. This is only spatial grouping.
    """
    people = frame_data.to_dict("records")
    used = set()
    groups = []

    for i in range(len(people)):
        if i in used:
            continue

        current_group = [i]
        used.add(i)

        changed = True

        while changed:
            changed = False

            for j in range(len(people)):
                if j in used:
                    continue

                for member_index in current_group:
                    distance = distance_between_people(
                        people[member_index],
                        people[j]
                    )

                    if distance <= GROUP_DISTANCE_EPS:
                        current_group.append(j)
                        used.add(j)
                        changed = True
                        break

        groups.append(current_group)

    labels = [-1] * len(people)
    group_id = 0

    for group in groups:
        if len(group) >= MIN_GROUP_SIZE:
            for person_index in group:
                labels[person_index] = group_id
            group_id += 1

    return labels


def cluster_groups_by_frame(people_df):
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


# -----------------------------
# Step 3: Build group summaries per frame
# -----------------------------

def build_group_summary(grouped_people_df):
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
            "member_ids": str(member_ids)
        })

    return pd.DataFrame(group_rows)


# -----------------------------
# Step 4: Match groups across frames
# -----------------------------

def parse_member_ids(member_ids_string):
    try:
        return set(ast.literal_eval(member_ids_string))
    except Exception:
        return set()


def calculate_group_overlap(ids1, ids2):
    if len(ids1) == 0 or len(ids2) == 0:
        return 0

    intersection = len(ids1.intersection(ids2))
    smaller_group_size = min(len(ids1), len(ids2))

    return intersection / smaller_group_size


def assign_persistent_group_ids(group_summary_df):
    group_summary_df = group_summary_df.copy()
    group_summary_df["persistent_group_id"] = None

    active_groups = {}
    next_persistent_id = 1

    for index, row in group_summary_df.sort_values("frame").iterrows():
        current_frame = row["frame"]
        current_ids = parse_member_ids(row["member_ids"])
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


# -----------------------------
# Step 5: Build final persistent group table
# -----------------------------

def build_group_lifetimes(group_summary_df):
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
        for ids_string in data["member_ids"]:
            all_member_ids.update(parse_member_ids(ids_string))

        lifetime_rows.append({
            "Group ID": f"G{persistent_id}",
            "persistent_group_id": persistent_id,

            # Main table columns
            "Size": round(data["group_size"].mean(), 2),
            "Max Size": data["group_size"].max(),
            "Location": main_region,
            "Duration (s)": round(duration_seconds, 2),
            "Type": main_spatial_type,
            "Motion": main_motion_type,

            # Average location
            "Average X": round(data["center_x"].mean(), 2),
            "Average Y": round(data["center_y"].mean(), 2),

            # REQUIRED for arrows
            "Start X": round(start_row["center_x"], 2),
            "Start Y": round(start_row["center_y"], 2),
            "End X": round(end_row["center_x"], 2),
            "End Y": round(end_row["center_y"], 2),

            # Extra info
            "First Frame": first_frame,
            "Last Frame": last_frame,
            "Frames Seen": data["frame"].nunique(),
            "Average Speed": round(data["avg_speed"].mean(), 2),
            "Member IDs Seen": str(sorted(all_member_ids))
        })

    return pd.DataFrame(lifetime_rows)


# -----------------------------
# Step 6: Save LaTeX table
# -----------------------------

def save_latex_group_table(group_lifetimes_df):
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
        caption="Detected Persistent Group Statistics",
        label="tab:group_statistics"
    )

    table_path = f"{OUTPUT_FOLDER}/group_statistics_table.tex"

    with open(table_path, "w") as file:
        file.write(latex_code)

    print(f"Saved LaTeX table: {table_path}")


# -----------------------------
# Step 7: Save arrow plot
# -----------------------------

def save_group_arrow_plot(group_lifetimes_df):
    if len(group_lifetimes_df) == 0:
        print("No groups to plot.")
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

        # If start and end are the same, make a tiny arrow so it is still visible.
        dx = end_x - start_x
        dy = end_y - start_y

        if abs(dx) < 2 and abs(dy) < 2:
            dx = 8
            dy = 0
            end_x = start_x + dx
            end_y = start_y + dy

        # Start point
        plt.scatter(
            start_x,
            start_y,
            c="black",
            s=35,
            marker="o",
            alpha=0.9,
            zorder=3
        )

        # Arrow from start to end
        plt.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=max(2, size),
                mutation_scale=24,
                alpha=0.85
            ),
            zorder=2
        )

        # End point
        plt.scatter(
            end_x,
            end_y,
            s=size * 100,
            c=color,
            alpha=0.80,
            edgecolors="black",
            zorder=4
        )

        # Group label
        plt.text(
            end_x + 5,
            end_y + 5,
            group_id,
            fontsize=9,
            weight="bold",
            zorder=5
        )

    plt.xlim(0, FRAME_WIDTH)
    plt.ylim(FRAME_HEIGHT, 0)

    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.title("Persistent Group Movement Map")

    plt.grid(True)

    # Legend
    plt.scatter([], [], c="black", s=35, marker="o", label="Start Point")
    plt.scatter([], [], c="blue", s=80, label="Stationary")
    plt.scatter([], [], c="orange", s=80, label="Slow Moving")
    plt.scatter([], [], c="red", s=80, label="Fast Moving")

    plt.legend()

    # Save as persistent_group_map.png so you can keep opening the same file name.
    plot_path = f"{OUTPUT_FOLDER}/persistent_group_map.png"

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved arrow map: {plot_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    create_output_folder()

    print("Parsing tracking results...")
    people_df = parse_tracking_file(INPUT_FILE)

    if len(people_df) == 0:
        print("No people data found. Check your tracking_results.txt format.")
        return

    print("Creating spatial groups by frame...")
    grouped_people_df = cluster_groups_by_frame(people_df)

    print("Building group summaries...")
    group_summary_df = build_group_summary(grouped_people_df)

    if len(group_summary_df) == 0:
        print("No spatial groups found.")
        return

    print("Assigning persistent group IDs...")
    group_summary_df = assign_persistent_group_ids(group_summary_df)

    print("Building final group table...")
    group_lifetimes_df = build_group_lifetimes(group_summary_df)

    print("Saving outputs...")
    save_latex_group_table(group_lifetimes_df)
    save_group_arrow_plot(group_lifetimes_df)

    print("\nDone.")
    print("Only saved:")
    print(f"- {OUTPUT_FOLDER}/group_statistics_table.tex")
    print(f"- {OUTPUT_FOLDER}/persistent_group_map.png")


if __name__ == "__main__":
    main()