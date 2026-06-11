"""
@Author : Evan Cillie
@Purpose : View YOLO + DeepSORT tracking video from a specific frame

Controls:
q = quit
s = save current frame as an image
p = pause / unpause
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict, deque


# ----------------------------
# Change these settings
# ----------------------------
VIDEO_PATH = "people-in-park.mp4"
START_FRAME = 250

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

GROUP_DISTANCE_THRESHOLD = 120
CROWD_SIZE_THRESHOLD = 4
LINE_ALIGNMENT_THRESHOLD = 60
PASSERBY_SPEED_THRESHOLD = 12

TRACK_HISTORY_LENGTH = 10


def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_speed(history):
    if len(history) < 2:
        return 0

    x1, y1 = history[0]
    x2, y2 = history[-1]

    distance = calculate_distance((x1, y1), (x2, y2))

    return distance / len(history)


def get_person_position(person):
    return person["center_x"], person["ground_y"]


def cluster_people_in_frame(people):
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
    total_x = 0
    total_y = 0

    for person in group:
        total_x += person["center_x"]
        total_y += person["ground_y"]

    center_x = total_x / len(group)
    center_y = total_y / len(group)

    return int(center_x), int(center_y)


def calculate_group_average_speed(group):
    if len(group) == 0:
        return 0

    total_speed = 0

    for person in group:
        total_speed += person["speed"]

    return total_speed / len(group)


def classify_group(group):
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


def get_detections(results):
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


def get_people_from_tracks(tracks, track_history):
    people = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        ground_y = y2

        track_history[track_id].append((center_x, center_y))
        speed = calculate_speed(track_history[track_id])

        person_data = {
            "track_id": track_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "ground_y": ground_y,
            "speed": speed
        }

        people.append(person_data)

    return people


def draw_people(frame, people):
    for person in people:
        x1 = person["x1"]
        y1 = person["y1"]
        x2 = person["x2"]
        y2 = person["y2"]
        track_id = person["track_id"]
        speed = person["speed"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Track ID and speed
        label = f"ID {track_id} | Speed {speed:.1f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2
        )

        # Center point
        cv2.circle(frame, (person["center_x"], person["center_y"]), 4, (0, 255, 0), -1)


def draw_groups(frame, groups):
    for group_id, group in enumerate(groups):
        group_type = classify_group(group)
        center_x, center_y = calculate_group_center(group)
        average_speed = calculate_group_average_speed(group)

        label = f"Group {group_id}: {group_type} | Size {len(group)} | Avg Speed {average_speed:.1f}"

        # Group center
        cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)

        # Group label
        cv2.putText(
            frame,
            label,
            (center_x - 80, center_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            2
        )

        # Draw lines from group center to members
        for person in group:
            cv2.line(
                frame,
                (center_x, center_y),
                (person["center_x"], person["ground_y"]),
                (255, 0, 0),
                1
            )


def main():
    model = YOLO("yolov8n.pt")

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))

    frame_number = START_FRAME
    paused = False
    last_frame = None

    print("Controls:")
    print("q = quit")
    print("s = save current clean frame")
    print("p = pause / unpause")

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("End of video or could not read frame.")
                break

            frame_number += 1

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # YOLO and DeepSORT still run in the background
            results = model(frame, verbose=False)[0]
            detections = get_detections(results)

            tracks = tracker.update_tracks(detections, frame=frame)
            people = get_people_from_tracks(tracks, track_history)

            groups = cluster_people_in_frame(people)

            # No boxes, labels, group lines, or frame number are drawn
            cv2.imshow("Clean Video Feed", frame)

            last_frame = frame.copy()

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            if last_frame is not None:
                filename = f"clean_frame_{frame_number}.png"
                cv2.imwrite(filename, last_frame)
                print(f"Saved {filename}")

        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()