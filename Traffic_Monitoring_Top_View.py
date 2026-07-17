import cv2
from collections import defaultdict
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.1

VIDEO_SOURCE = "VIDEO/Traffic_Intersection_2.mp4"

TRACK_BUFFER = 30

TARGET_CLASSES = ["car", "bus", "truck", "motorcycle"]

# Original Zone
ZONE_TOP_LEFT = (500, 200)
ZONE_BOTTOM_RIGHT = (900, 400)

# Red zone for intersection, used for direction classification
TURN_ZONE_TOP_LEFT = (350, 150)
TURN_ZONE_BOTTOM_RIGHT = (700, 450)

CLASS_COLORS = {
    "car": (60, 180, 255),
    "bus": (80, 220, 100),
    "truck": (255, 200, 100),
    "motorcycle": (139, 0, 0)
}
DEFAULT_COLOR = (200, 200, 200)
ZONE_COLOR = (255, 255, 255)
TURN_ZONE_COLOR = (0, 0, 255)

# Clockwise order of sides, used to work out turn direction
SIDE_ORDER = ["top", "right", "bottom", "left"]


def build_tracker_config(track_buffer):
    config_path = "bytetrack_custom.yaml"
    config_text = (
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.5\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.6\n"
        f"track_buffer: {track_buffer}\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )
    with open(config_path, "w") as f:
        f.write(config_text)
    return config_path


def point_in_zone(point, top_left, bottom_right):
    x, y = point
    zx1, zy1 = top_left
    zx2, zy2 = bottom_right
    return zx1 <= x <= zx2 and zy1 <= y <= zy2

def get_side(point, top_left, bottom_right):
    """Figures out which edge of a box a point is closest to.
        Used right as a vehicle crosses into or out of the turn zone,
        so 'nearest edge' effectively tells us which edge it crossed."""
    x, y = point
    zx1, zy1 = top_left
    zx2, zy2 = bottom_right

    d_left = abs(x - zx1)
    d_right = abs(x - zx2)
    d_top = abs(y - zy1)
    d_bottom = abs(y - zy2)

    closest = min(d_left, d_right, d_top, d_bottom)
    if closest == d_top:
        return "top"
    if closest == d_bottom:
        return "bottom"
    if closest == d_left:
        return "left"
    return "right"

def classify_direction(entry_side, exit_side):
    """Compares entry edge to exit edge to label the movement as straight, left or right. Uses clockwise edge order (top --> right --> bottom --> left --> top) to work out the relationship."""
    if entry_side == exit_side:
        return None          # entered and left from the same edge, likely noise
    
    e_idx = SIDE_ORDER.index(entry_side)
    straight_exit = SIDE_ORDER[(e_idx + 2) % 4]
    right_exit = SIDE_ORDER[(e_idx + 3) % 4]
    left_exit = SIDE_ORDER[(e_idx + 1) % 4]

    if exit_side == straight_exit:
        return "straight"
    elif exit_side == right_exit:
        return "right"
    elif exit_side == left_exit:
        return "left"
    return None

def main():
    model = YOLO(MODEL_NAME)
    class_names = model.names
    tracker_config = build_tracker_config(TRACK_BUFFER)

    counted_ids = set()
    class_counts = defaultdict(int)

    # TRACKING STATE FOR TURN CLASSIFICATION
    # per track_id: whether currently inside turn zone and what side
    # it entered from, so we can compare against the exit side later

    turn_state = {}
    direction_counts = defaultdict(int)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, frame = cap.read()

    # Draw gridlines every 100px to get sketch for bounding box
    for x in range(0, frame.shape[1], 100):
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 0), 1)
        cv2.putText(frame, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, frame.shape[0], 100):
        cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 1)
        cv2.putText(frame, str(y), (2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite("grid_frame.png", frame)
    print(frame.shape)


    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_SOURCE}")
        return

    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = class_names[class_id]

                if TARGET_CLASSES and class_name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
                
                # Straight path zone counting
                inside_zone = point_in_zone(center, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT)
                if inside_zone and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    class_counts[class_name] += 1

                # Turn zone tracking
                inside_turn_zone = point_in_zone(center, TURN_ZONE_TOP_LEFT, TURN_ZONE_BOTTOM_RIGHT)

                if track_id not in turn_state:
                    turn_state[track_id] = {"was_inside": False, "entry_side": None, "classified": False}

                state = turn_state[track_id]

                if inside_turn_zone and not state["was_inside"]:
                    #just entered the turn zone
                    state["entry_side"] = get_side(center, TURN_ZONE_TOP_LEFT, TURN_ZONE_BOTTOM_RIGHT)
                    state["was_inside"] = True
                elif not inside_turn_zone and state["was_inside"] and not state["classified"]:
                    # just left the turn zone, classify the movement
                    exit_side = get_side(center, TURN_ZONE_TOP_LEFT, TURN_ZONE_BOTTOM_RIGHT)
                    direction = classify_direction(state["entry_side"], exit_side)
                    if direction:
                        direction_counts[direction] += 1
                        state["classified"] = True
                    state["was_inside"] = False

                dot_radius = 6 if track_id in counted_ids else 3
                cv2.circle(frame, center, dot_radius, color, -1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} #{track_id}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, ZONE_COLOR, 2)
        cv2.rectangle(frame, TURN_ZONE_TOP_LEFT, TURN_ZONE_BOTTOM_RIGHT, TURN_ZONE_COLOR, 2)

        y_offset = 40
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_offset += 30

        y_offset += 10
        for direction, count in direction_counts.items():
            cv2.putText(frame, f"{direction}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_offset += 30

        cv2.imshow("Mini Project - Traffic Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final straight zone count:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    print(f"\nFinal direction zone count:")
    for direction, count in direction_counts.items():
        print(f"{direction}: {count}")

if __name__ == "__main__":
    main()