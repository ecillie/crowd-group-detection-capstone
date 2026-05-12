"""
@Author : Evan Cillie
@LastEdit : 05-06-26
@Purpose : CSC 488 Capstone YOLO + DeepSORT Formation Detection
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

model = YOLO("yolov8n.pt")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4
)
cap = cv2.VideoCapture("birds-eye-view.mp4")
if not cap.isOpened():
    raise Exception("Could not open video")
track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))

formation_counts = Counter()
unique_track_ids = set()

frame_number = 0
processed_frames = 0


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


def detect_close_pairs(people):
    close_pairs = 0

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1 = (people[i]["center_x"], people[i]["center_y"])
            p2 = (people[j]["center_x"], people[j]["center_y"])

            if calculate_distance(p1, p2) < GROUP_DISTANCE_THRESHOLD:
                close_pairs += 1

    return close_pairs


def detect_line(people):
    if len(people) < 3:
        return False

    x_values = [person["center_x"] for person in people]
    y_values = [person["center_y"] for person in people]

    x_spread = max(x_values) - min(x_values)
    y_spread = max(y_values) - min(y_values)

    vertical_line = x_spread < LINE_ALIGNMENT_THRESHOLD and y_spread > LINE_ALIGNMENT_THRESHOLD
    horizontal_line = y_spread < LINE_ALIGNMENT_THRESHOLD and x_spread > LINE_ALIGNMENT_THRESHOLD

    return vertical_line or horizontal_line


def detect_passerby(people):
    passerby_ids = []

    for person in people:
        track_id = person["track_id"]
        speed = calculate_speed(track_history[track_id])

        if speed > PASSERBY_SPEED_THRESHOLD:
            passerby_ids.append(track_id)

    return passerby_ids


def classify_formation(people):
    active_people = len(people)

    if active_people == 0:
        return "NO PEOPLE"

    if active_people == 1:
        if len(detect_passerby(people)) > 0:
            return "PASSERBY DETECTED"
        return "SINGLE PERSON"

    close_pairs = detect_close_pairs(people)
    is_line = detect_line(people)
    passerby_ids = detect_passerby(people)

    if is_line:
        return "LINE DETECTED"

    if active_people >= CROWD_THRESHOLD and close_pairs >= CROWD_THRESHOLD:
        return "CROWD DETECTED"

    if close_pairs > 0:
        return "GROUP DETECTED"

    if len(passerby_ids) > 0:
        return "PASSERBY DETECTED"

    return "MULTIPLE PEOPLE"


with open("tracking_results.txt", "w") as output_file:
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
        print(f"in Frame {frame_number}")

        frame = cv2.resize(frame, (640, 360))

        results = model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1

                detections.append(([x1, y1, width, height], confidence, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        people = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            unique_track_ids.add(track_id)

            x1, y1, x2, y2 = map(int, track.to_ltrb())

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            track_history[track_id].append((center_x, center_y))

            person_data = {
                "track_id": track_id,
                "center_x": center_x,
                "center_y": center_y,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "speed": calculate_speed(track_history[track_id])
            }

            people.append(person_data)

        status = classify_formation(people)
        formation_counts[status] += 1

        output_file.write(f"Frame {frame_number}\n")
        output_file.write(f"Formation: {status}\n")
        output_file.write(f"People tracked: {len(people)}\n")

        for person in people:
            output_file.write(
                f"  ID {person['track_id']} | "
                f"Center: ({person['center_x']}, {person['center_y']}) | "
                f"Speed: {person['speed']:.2f} px/frame\n"
            )

        output_file.write("\n")

    output_file.write("\nFinal Summary\n")
    output_file.write("-------------\n")
    output_file.write(f"Frames Tracked: {processed_frames}\n")
    output_file.write(f"Total tracked IDs: {len(unique_track_ids)}\n\n")

    output_file.write("Formation counts:\n")
    for formation, count in formation_counts.items():
        output_file.write(f"{formation}: {count} frames\n")


cap.release()

print("Tracking complete.")
print("Results saved to tracking_results.txt")