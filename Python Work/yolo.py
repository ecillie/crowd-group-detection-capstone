"""
@Author : Evan Cillie
@LastEdit : 05-17-26
@Purpose : CSC 488 Capstone YOLO + DeepSORT Formation Detection

This script detects and tracks people in video files using YOLO and DeepSORT.
It writes frame-by-frame tracking results to text files.

Important:
This script does not estimate true physical depth. Since the input is a
single 2D video, the program uses relative bounding-box scale instead.
A larger relative scale usually means the person appears closer to the camera.
A smaller relative scale usually means the person appears farther away.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict, deque, Counter


PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

GROUP_DISTANCE_THRESHOLD = 150
CROWD_THRESHOLD = 4
LINE_ALIGNMENT_THRESHOLD = 60
PASSERBY_SPEED_THRESHOLD = 12

TRACK_HISTORY_LENGTH = 10
PROCESS_EVERY_N_FRAMES = 2

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
    Calculates average movement speed using a person's tracking history.

    Speed is measured in pixels per frame. This is not physical speed.

    @param history: A deque of previous center points for one tracked person.
    @return: The average pixel movement per stored frame.
    """
    if len(history) < 2:
        return 0

    x1, y1 = history[0]
    x2, y2 = history[-1]

    distance = calculate_distance((x1, y1), (x2, y2))

    return distance / len(history)


def calculate_relative_scale(box_height):
    """
    Calculates a relative screen-size value for a detected person.

    This replaces the old depth calculation. It does not estimate true depth.
    It only measures how tall the bounding box is relative to the frame height.

    A larger value usually means the person appears closer to the camera.
    A smaller value usually means the person appears farther away.

    @param box_height: The height of the person's bounding box.
    @return: The relative scale of the person in the frame.
    """
    if box_height <= 0:
        return 0

    return box_height / FRAME_HEIGHT


def detect_close_pairs(people):
    """
    Counts how many pairs of people are close together in screen space.

    @param people: A list of dictionaries containing tracked person data.
    @return: The number of close pairs detected.
    """
    close_pairs = 0

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1 = (people[i]["center_x"], people[i]["ground_y"])
            p2 = (people[j]["center_x"], people[j]["ground_y"])

            if calculate_distance(p1, p2) < GROUP_DISTANCE_THRESHOLD:
                close_pairs += 1

    return close_pairs


def detect_line(people):
    """
    Detects whether the tracked people form a rough horizontal or vertical line.

    @param people: A list of dictionaries containing tracked person data.
    @return: True if a line is detected, otherwise False.
    """
    if len(people) < 3:
        return False

    x_values = []
    y_values = []

    for person in people:
        x_values.append(person["center_x"])
        y_values.append(person["ground_y"])

    x_spread = max(x_values) - min(x_values)
    y_spread = max(y_values) - min(y_values)

    vertical_line = x_spread < LINE_ALIGNMENT_THRESHOLD and y_spread > LINE_ALIGNMENT_THRESHOLD
    horizontal_line = y_spread < LINE_ALIGNMENT_THRESHOLD and x_spread > LINE_ALIGNMENT_THRESHOLD

    return vertical_line or horizontal_line


def detect_passerby(people, track_history):
    """
    Detects tracked people who are moving quickly across the frame.

    @param people: A list of dictionaries containing tracked person data.
    @param track_history: A dictionary mapping track IDs to recent center points.
    @return: A list of track IDs classified as passersby.
    """
    passerby_ids = []

    for person in people:
        track_id = person["track_id"]
        speed = calculate_speed(track_history[track_id])

        if speed > PASSERBY_SPEED_THRESHOLD:
            passerby_ids.append(track_id)

    return passerby_ids


def classify_formation(people, track_history):
    """
    Classifies the current frame's human formation.

    Possible labels include:
    NO PEOPLE, SINGLE PERSON, LINE DETECTED, CROWD DETECTED,
    GROUP DETECTED, PASSERBY DETECTED, and MULTIPLE PEOPLE.

    @param people: A list of dictionaries containing tracked person data.
    @param track_history: A dictionary mapping track IDs to recent center points.
    @return: A formation label for the current frame.
    """
    active_people = len(people)

    if active_people == 0:
        return "NO PEOPLE"

    if active_people == 1:
        if len(detect_passerby(people, track_history)) > 0:
            return "PASSERBY DETECTED"

        return "SINGLE PERSON"

    close_pairs = detect_close_pairs(people)
    is_line = detect_line(people)
    passerby_ids = detect_passerby(people, track_history)

    if is_line:
        return "LINE DETECTED"

    if active_people >= CROWD_THRESHOLD and close_pairs >= CROWD_THRESHOLD:
        return "CROWD DETECTED"

    if close_pairs > 0:
        return "GROUP DETECTED"

    if len(passerby_ids) > 0:
        return "PASSERBY DETECTED"

    return "MULTIPLE PEOPLE"


def get_detections(results):
    """
    Extracts person detections from YOLO results.

    DeepSORT expects detections in the format:
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
    Converts confirmed DeepSORT tracks into person data dictionaries.

    Each dictionary contains the person's track ID, center position, speed,
    relative scale, and ground-level y-coordinate.

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

        box_height = y2 - y1
        relative_scale = calculate_relative_scale(box_height)

        track_history[track_id].append((center_x, center_y))
        speed = calculate_speed(track_history[track_id])

        person_data = {
            "track_id": track_id,
            "center_x": center_x,
            "center_y": center_y,
            "speed": speed,
            "relative_scale": relative_scale,
            "ground_y": ground_y
        }

        people.append(person_data)

    return people


def write_frame_results(output_file, frame_number, status, people):
    """
    Writes the tracking results for one processed frame.

    @param output_file: The open text file being written to.
    @param frame_number: The current frame number from the video.
    @param status: The formation classification for the frame.
    @param people: A list of tracked person dictionaries.
    @return: None
    """
    output_file.write(f"Frame {frame_number}\n")
    output_file.write(f"Formation: {status}\n")
    output_file.write(f"People tracked: {len(people)}\n")

    for person in people:
        output_file.write(
            f"  ID {person['track_id']} | "
            f"Center: ({person['center_x']}, {person['center_y']}) | "
            f"Speed: {person['speed']:.2f} px/frame | "
            f"Relative Scale: {person['relative_scale']:.2f} | "
            f"Ground Y: {person['ground_y']}\n"
        )

    output_file.write("\n")


def write_final_summary(output_file, processed_frames, unique_track_ids, formation_counts):
    """
    Writes the final tracking summary at the end of the output file.

    @param output_file: The open text file being written to.
    @param processed_frames: The number of frames processed by the model.
    @param unique_track_ids: A set of all unique DeepSORT track IDs.
    @param formation_counts: A Counter storing formation label frequencies.
    @return: None
    """
    output_file.write("\nFinal Summary\n")
    output_file.write("-------------\n")
    output_file.write(f"Frames Tracked: {processed_frames}\n")
    output_file.write(f"Total tracked IDs: {len(unique_track_ids)}\n\n")

    output_file.write("Formation counts:\n")

    for formation, count in formation_counts.items():
        output_file.write(f"{formation}: {count} frames\n")


def process_video(video_file_name, output_file_name):
    """
    Runs YOLO and DeepSORT on one video file and writes tracking results.

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
    formation_counts = Counter()
    unique_track_ids = set()

    frame_number = 0
    processed_frames = 0

    with open(output_file_name, "w") as output_file:
        output_file.write("CSC 488 Capstone Tracking Results\n")
        output_file.write("YOLO + DeepSORT Formation Detection\n")
        output_file.write("-----------------------------------\n\n")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            if frame_number % PROCESS_EVERY_N_FRAMES != 0:
                continue

            processed_frames += 1

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            results = model(frame, verbose=False)[0]
            detections = get_detections(results)

            tracks = tracker.update_tracks(detections, frame=frame)
            people = get_people_from_tracks(tracks, track_history, unique_track_ids)

            status = classify_formation(people, track_history)
            formation_counts[status] += 1

            write_frame_results(output_file, frame_number, status, people)

        write_final_summary(
            output_file,
            processed_frames,
            unique_track_ids,
            formation_counts
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
        ("people-in-park.mp4", "people_in_park_results.txt"),
        ("people-walking.mp4", "people-walking.txt"),
        ("wold.mp4", "wold_results.txt")
    ]

    main(video_output_pairs)