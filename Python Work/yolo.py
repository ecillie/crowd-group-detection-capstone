"""
@Author : Evan Cillie
@LastEdit : 05-17-26
@Purpose : CSC 488 Capstone YOLO + DeepSORT Multi-Group Crowd Formation Detection

This script detects and tracks people in video files using YOLO and DeepSORT.
Each frame can contain multiple separate groups. The program clusters people
inside each frame and classifies each group separately.

"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict, deque, Counter


PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

GROUP_DISTANCE_THRESHOLD = 120
CROWD_SIZE_THRESHOLD = 4
LINE_ALIGNMENT_THRESHOLD = 60
PASSERBY_SPEED_THRESHOLD = 12

PROCESS_EVERY_FRAMES = 2
TRACK_HISTORY_LENGTH = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 360


def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.

    @param p1: The first point as an (x, y) tuple.
    @param p2: The second point as an (x, y) tuple.
    @return: The distance between the two points.
    """
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_speed(history):
    """
    Calculates the average movement speed of one tracked person.

    Speed is measured in pixels per processed frame. This is not physical
    speed in feet or meters per second.

    @param history: A deque of previous center points for one tracked person.
    @return: The average pixel movement per stored frame.
    """
    if len(history) < 2:
        return 0

    x1, y1 = history[0]
    x2, y2 = history[-1]

    distance = calculate_distance((x1, y1), (x2, y2))

    return distance / len(history)


def get_person_position(person):
    """
    Gets the 2D position used for group clustering.

    The x-coordinate is the center of the bounding box.
    The y-coordinate is the bottom of the bounding box, also called ground_y.
    This better represents where the person is standing.

    @param person: A dictionary containing tracked person data.
    @return: A tuple containing the person's grouping position.
    """
    return person["center_x"], person["ground_y"]


def cluster_people_in_frame(people):
    """
    Splits people in one frame into separate groups.

    This uses connected grouping:
    - If person A is close to person B, they are in the same group.
    - If person B is close to person C, then A, B, and C are grouped together.

    A person who is not close to anyone else becomes a single-person group.

    @param people: A list of dictionaries containing tracked person data.
    @return: A list of groups, where each group is a list of person dictionaries.
    """
    groups = []
    used_indices = set()

    for i in range(len(people)):
        if i in used_indices:
            continue

        current_group_indices = [i]
        used_indices.add(i)

        group_index = 0

        while group_index < len(current_group_indices):
            current_person_index = current_group_indices[group_index]
            current_person = people[current_person_index]
            current_position = get_person_position(current_person)

            for j in range(len(people)):
                if j in used_indices:
                    continue

                other_person = people[j]
                other_position = get_person_position(other_person)

                distance = calculate_distance(current_position, other_position)

                if distance <= GROUP_DISTANCE_THRESHOLD:
                    current_group_indices.append(j)
                    used_indices.add(j)

            group_index += 1

        group = []

        for index in current_group_indices:
            group.append(people[index])

        groups.append(group)

    return groups


def detect_group_line(group):
    """
    Detects whether a group forms a rough horizontal or vertical line.

    A horizontal line has a wide x-spread and small y-spread.
    A vertical line has a small x-spread and wide y-spread.

    @param group: A list of person dictionaries belonging to one group.
    @return: True if the group forms a line, otherwise False.
    """
    if len(group) < 3:
        return False

    x_values = []
    y_values = []

    for person in group:
        x_values.append(person["center_x"])
        y_values.append(person["ground_y"])

    x_spread = max(x_values) - min(x_values)
    y_spread = max(y_values) - min(y_values)

    vertical_line = x_spread < LINE_ALIGNMENT_THRESHOLD and y_spread > LINE_ALIGNMENT_THRESHOLD
    horizontal_line = y_spread < LINE_ALIGNMENT_THRESHOLD and x_spread > LINE_ALIGNMENT_THRESHOLD

    return vertical_line or horizontal_line


def calculate_group_center(group):
    """
    Calculates the average center point of a group.

    @param group: A list of person dictionaries belonging to one group.
    @return: A tuple containing the average x-coordinate and average ground y-coordinate.
    """
    total_x = 0
    total_y = 0

    for person in group:
        total_x += person["center_x"]
        total_y += person["ground_y"]

    center_x = total_x / len(group)
    center_y = total_y / len(group)

    return center_x, center_y


def calculate_group_average_speed(group):
    """
    Calculates the average speed of all people in a group.

    @param group: A list of person dictionaries belonging to one group.
    @return: The average group speed.
    """
    if len(group) == 0:
        return 0

    total_speed = 0

    for person in group:
        total_speed += person["speed"]

    return total_speed / len(group)


def get_group_member_ids(group):
    """
    Gets the DeepSORT track IDs for all people in a group.

    @param group: A list of person dictionaries belonging to one group.
    @return: A sorted list of track IDs.
    """
    member_ids = []

    for person in group:
        member_ids.append(person["track_id"])

    return sorted(member_ids)


def classify_group(group):
    """
    Classifies one group inside a frame.

    Possible group labels:
    SINGLE PERSON
    PASSERBY
    SMALL GROUP
    CROWD
    LINE

    @param group: A list of person dictionaries belonging to one group.
    @return: The group type label.
    """
    group_size = len(group)
    average_speed = calculate_group_average_speed(group)
    is_line = detect_group_line(group)

    if group_size == 1:
        if average_speed >= PASSERBY_SPEED_THRESHOLD:
            return "PASSERBY"

        return "SINGLE PERSON"

    if is_line:
        return "LINE"

    if group_size >= CROWD_SIZE_THRESHOLD:
        return "CROWD"

    return "SMALL GROUP"


def build_group_summaries(groups):
    """
    Creates summary dictionaries for all groups in one frame.

    @param groups: A list of groups, where each group is a list of person dictionaries.
    @return: A list of group summary dictionaries.
    """
    group_summaries = []

    for group_id, group in enumerate(groups):
        group_type = classify_group(group)
        member_ids = get_group_member_ids(group)
        center_x, center_y = calculate_group_center(group)
        average_speed = calculate_group_average_speed(group)

        group_summary = {
            "group_id": group_id,
            "type": group_type,
            "members": member_ids,
            "size": len(group),
            "center_x": center_x,
            "center_y": center_y,
            "avg_speed": average_speed
        }

        group_summaries.append(group_summary)

    return group_summaries


def get_detections(results):
    """
    Extracts person detections from YOLO results.

    DeepSORT expects detections in this format:
    ([x, y, width, height], confidence, class_name)

    @param results: The YOLO result object for one frame.
    @return: A list of person detections formatted for DeepSORT.
    """
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            detections.append(([x1, y1, width, height], confidence, "person"))

    return detections


def get_people_from_tracks(tracks, track_history, unique_track_ids):
    """
    Converts confirmed DeepSORT tracks into person dictionaries.

    Each person dictionary stores:
    track_id
    center_x
    center_y
    ground_y
    speed

    @param tracks: The list of DeepSORT tracks for the current frame.
    @param track_history: A dictionary mapping track IDs to recent center points.
    @param unique_track_ids: A set storing every unique track ID seen.
    @return: A list of tracked person dictionaries.
    """
    people = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        unique_track_ids.add(track_id)

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        ground_y = y2

        track_history[track_id].append((center_x, center_y))
        speed = calculate_speed(track_history[track_id])

        person_data = {
            "track_id": track_id,
            "center_x": center_x,
            "center_y": center_y,
            "ground_y": ground_y,
            "speed": speed
        }

        people.append(person_data)

    return people


def write_frame_results(output_file, frame_number, people, group_summaries):
    """
    Writes tracking and group results for one processed frame.

    @param output_file: The open text file being written to.
    @param frame_number: The current frame number from the video.
    @param people: A list of tracked person dictionaries.
    @param group_summaries: A list of group summary dictionaries.
    @return: None
    """
    output_file.write(f"Frame {frame_number}\n")
    output_file.write(f"People tracked: {len(people)}\n")
    output_file.write(f"Groups detected: {len(group_summaries)}\n")

    for group in group_summaries:
        output_file.write(
            f"Group {group['group_id']} | "
            f"Type: {group['type']} | "
            f"Members: {group['members']} | "
            f"Size: {group['size']} | "
            f"Center: ({group['center_x']:.2f}, {group['center_y']:.2f}) | "
            f"Avg Speed: {group['avg_speed']:.2f}\n"
        )

    output_file.write("\n")


def write_final_summary(output_file, processed_frames, unique_track_ids, group_type_counts):
    """
    Writes the final tracking summary at the end of the output file.

    @param output_file: The open text file being written to.
    @param processed_frames: The number of frames processed by the model.
    @param unique_track_ids: A set of all unique DeepSORT track IDs.
    @param group_type_counts: A Counter storing group type frequencies.
    @return: None
    """
    output_file.write("\nFinal Summary\n")
    output_file.write("-------------\n")
    output_file.write(f"Frames Tracked: {processed_frames}\n")
    output_file.write(f"Total tracked IDs: {len(unique_track_ids)}\n\n")

    output_file.write("Group type counts:\n")

    for group_type, count in group_type_counts.items():
        output_file.write(f"{group_type}: {count} groups\n")


def process_video(video_file_name, output_file_name):
    """
    Runs YOLO and DeepSORT on one video file.

    @param video_file_name: The input video file path.
    @param output_file_name: The output text file path.
    @return: None
    """
    model = YOLO("yolov8n.pt")

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4
    )

    cap = cv2.VideoCapture(video_file_name)

    if not cap.isOpened():
        raise Exception("Could not open video")

    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    group_type_counts = Counter()
    unique_track_ids = set()

    frame_number = 0
    processed_frames = 0

    with open(output_file_name, "w") as output_file:
        output_file.write("CSC 488 Capstone Tracking Results\n")
        output_file.write("YOLO + DeepSORT Multi-Group Formation Detection\n")
        output_file.write("-----------------------------------------------\n\n")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            if frame_number % PROCESS_EVERY_FRAMES != 0:
                continue

            processed_frames += 1

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            results = model(frame, verbose=False)[0]
            detections = get_detections(results)

            tracks = tracker.update_tracks(detections, frame=frame)
            people = get_people_from_tracks(tracks, track_history, unique_track_ids)

            groups = cluster_people_in_frame(people)
            group_summaries = build_group_summaries(groups)

            for group in group_summaries:
                group_type_counts[group["type"]] += 1

            write_frame_results(output_file, frame_number, people, group_summaries)

        write_final_summary(
            output_file,
            processed_frames,
            unique_track_ids,
            group_type_counts
        )

    cap.release()


def main(video_output_pairs):
    """
    Processes multiple videos.

    @param video_output_pairs: A list of tuples containing video file names and output file names.
    @return: None
    """
    for video_file_name, output_file_name in video_output_pairs:
        process_video(video_file_name, output_file_name)


if __name__ == "__main__":
    video_output_pairs = [
        ("people-in-park.mp4", "people-in-park-results.txt"),
        ("people-walking.mp4", "people-walking.txt"),
        ("wold.mp4", "wold-results.txt"),
        ("pier-walking.mp4","pier-walking-results.txt"),
        ("walk-in-park.mp4","walk-in-park-results.txt")
    ]

    main(video_output_pairs)