import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime

def is_point_in_polygon(point, polygon):
    """
    Checks if a point is inside a polygon.
    Point should be (x, y), and polygon should be a list of (x, y) tuples.
    """
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def detect_helmet_on_person(person_box, frame):
    """
    Simple helmet detection logic - checks for bright/reflective objects
    on the upper part of the person's bounding box.
    This is a basic implementation. For better accuracy, use a custom-trained
    YOLO model that includes 'helmet' class.
    """
    x1, y1, x2, y2 = person_box
    person_width = x2 - x1
    person_height = y2 - y1

    # Define head region (upper 30% of person bounding box)
    head_y1 = y1
    head_y2 = y1 + int(person_height * 0.3)
    head_x1 = x1 + int(person_width * 0.2)  # Slightly inset from sides
    head_x2 = x2 - int(person_width * 0.2)

    # Ensure coordinates are within frame bounds
    head_y1 = max(0, head_y1)
    head_y2 = min(frame.shape[0], head_y2)
    head_x1 = max(0, head_x1)
    head_x2 = min(frame.shape[1], head_x2)

    if head_x2 <= head_x1 or head_y2 <= head_y1:
        return False

    # Extract head region
    head_region = frame[head_y1:head_y2, head_x1:head_x2]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

    # Define range for bright/reflective colors (yellow, white, orange helmets)
    # Adjust these ranges based on your helmet colors
    lower_bright = np.array([0, 0, 150])   # Low saturation, high brightness
    upper_bright = np.array([180, 50, 255])

    # Create mask for bright colors
    bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)

    # Also check for specific helmet colors (yellow, orange, white)
    # Yellow helmets
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Orange helmets
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # White helmets
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine all helmet color masks
    helmet_mask = cv2.bitwise_or(bright_mask, yellow_mask)
    helmet_mask = cv2.bitwise_or(helmet_mask, orange_mask)
    helmet_mask = cv2.bitwise_or(helmet_mask, white_mask)

    # Calculate percentage of head region that might be helmet
    helmet_pixels = cv2.countNonZero(helmet_mask)
    total_pixels = head_region.shape[0] * head_region.shape[1]

    if total_pixels == 0:
        return False

    helmet_percentage = (helmet_pixels / total_pixels) * 100

    # If more than 15% of head region shows helmet-like colors, consider it has helmet
    # This threshold can be adjusted based on testing
    return helmet_percentage > 15

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can use a custom model trained for helmets

    # Get frame dimensions for full screen zone
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the construction site detection zone as full screen
    detection_zone = [(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)]

    # Create directories for captured images
    helmet_dir = "helmet_compliant"
    no_helmet_dir = "no_helmet_violations"
    os.makedirs(helmet_dir, exist_ok=True)
    os.makedirs(no_helmet_dir, exist_ok=True)

    # Tracking sets
    processed_person_ids = set()
    helmet_violations = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Counters for current frame
        current_people_in_zone = 0
        current_helmet_compliant = 0
        current_no_helmet = 0

        # Perform object detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Process results
        for r in results:
            boxes = r.boxes
            if boxes.id is not None:
                for box in boxes:
                    # Get bounding box coordinates and tracking ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0])

                    # Calculate centroid
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    object_centroid = (centroid_x, centroid_y)

                    # Get class name
                    class_id = int(box.cls)
                    class_name = model.names[class_id]

                    # Only process persons
                    if class_name == "person":
                        # Check if person is in construction zone
                        if is_point_in_polygon(object_centroid, detection_zone):
                            current_people_in_zone += 1

                            # Check for helmet
                            has_helmet = detect_helmet_on_person((x1, y1, x2, y2), frame)

                            if has_helmet:
                                color = (0, 255, 0)  # Green for helmet compliant
                                status_text = "HELMET OK"
                                current_helmet_compliant += 1

                                # Capture compliant person image (once per person)
                                if track_id not in processed_person_ids:
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    image_filename = os.path.join(helmet_dir, f"compliant_id_{track_id}_{timestamp}.jpg")
                                    cv2.imwrite(image_filename, frame)
                                    print(f"Captured compliant person: {image_filename}")

                            else:
                                color = (0, 0, 255)  # Red for no helmet violation
                                status_text = "NO HELMET!"
                                current_no_helmet += 1
                                helmet_violations.add(track_id)

                                # Capture violation image (once per person)
                                if track_id not in processed_person_ids:
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    image_filename = os.path.join(no_helmet_dir, f"violation_id_{track_id}_{timestamp}.jpg")
                                    cv2.imwrite(image_filename, frame)
                                    print(f"Captured violation: {image_filename}")

                            # Mark this person as processed
                            processed_person_ids.add(track_id)

                            # Draw bounding box and status
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f"Person {track_id}", (x1, y1 - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                            cv2.putText(frame, status_text, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

                            # Add zone indicator
                            cv2.putText(frame, "IN CONSTRUCTION ZONE", (centroid_x - 80, centroid_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        # Display statistics
        cv2.putText(frame, f"People in Zone: {current_people_in_zone}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Helmet Compliant: {current_helmet_compliant}", (200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"No Helmet: {current_no_helmet}", (400, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Total Violations: {len(helmet_violations)}", (550, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw the construction zone
        cv2.polylines(frame, [np.array(detection_zone)], True, (255, 255, 0), 3)
        cv2.putText(frame, "CONSTRUCTION ZONE", (detection_zone[0][0] + 20, detection_zone[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Resize frame to make it bigger (2x scale)
        display_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Construction Site Helmet Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final summary
    print("=== HELMET DETECTION SUMMARY ===")
    print(f"Total people processed: {len(processed_person_ids)}")
    print(f"Total helmet violations detected: {len(helmet_violations)}")
    print(f"Compliance rate: {((len(processed_person_ids) - len(helmet_violations)) / max(len(processed_person_ids), 1)) * 100:.1f}%")

if __name__ == '__main__':
    # Replace with your construction site video path
    video_file = r"D:\Yolo\video\People working at construction site.mp4"
    process_video(video_file)