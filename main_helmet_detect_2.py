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

def check_helmet_proximity(person_box, helmet_boxes):
    """
    Check if there's a helmet near a person's head region.
    This is used when the model detects separate helmet and person objects.
    """
    px1, py1, px2, py2 = person_box
    person_center_x = (px1 + px2) / 2
    person_head_y = py1 + (py2 - py1) * 0.2  # Top 20% of person (head area)

    # Check if any helmet is close to the person's head
    for hx1, hy1, hx2, hy2 in helmet_boxes:
        helmet_center_x = (hx1 + hx2) / 2
        helmet_center_y = (hy1 + hy2) / 2

        # Check if helmet is within reasonable proximity of person's head
        distance_x = abs(person_center_x - helmet_center_x)
        distance_y = abs(person_head_y - helmet_center_y)

        # If helmet is close to person's head region, consider it worn
        if distance_x < (px2 - px1) * 0.8 and distance_y < (py2 - py1) * 0.4:
            return True

    return False

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Try to load a specialized safety helmet detection model
    # Available models: 'yolov8-ppe.pt', 'hardhat-worker-detection.pt', 'safety-helmet-detection.pt'
    try:
        model = YOLO('yolov8-ppe.pt')  # Specialized model for helmet detection
        print("Loaded specialized safety helmet detection model")
        use_specialized_model = True
    except:
        print("Specialized model not found, using general YOLOv8n model with custom helmet detection")
        model = YOLO('yolov8n.pt')  # Fallback to general model
        use_specialized_model = False

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

        # Lists to store detected objects
        persons = []  # List of (track_id, box, class_name)
        helmets = []  # List of (box) for helmets

        # Perform object detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Process results and categorize detections
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

                    # Check if person is in construction zone first
                    if is_point_in_polygon(object_centroid, detection_zone):
                        # Categorize detections
                        if class_name.lower() in ['person', 'worker', 'worker_no_helmet', 'person_no_helmet']:
                            persons.append((track_id, (x1, y1, x2, y2), class_name, object_centroid))
                        elif class_name.lower() in ['helmet', 'hardhat', 'worker_with_helmet', 'person_with_helmet']:
                            helmets.append((x1, y1, x2, y2))

                            # If this is a person_with_helmet detection, treat as person
                            if 'with_helmet' in class_name.lower():
                                persons.append((track_id, (x1, y1, x2, y2), class_name, object_centroid))

        # Process each person and check for helmet compliance
        for track_id, person_box, class_name, centroid in persons:
            current_people_in_zone += 1
            px1, py1, px2, py2 = person_box

            # Determine if person has helmet
            has_helmet = False

            # Check class name first (for combined classes like person_with_helmet)
            if 'with_helmet' in class_name.lower() or 'helmet' in class_name.lower():
                has_helmet = True
            elif 'no_helmet' in class_name.lower():
                has_helmet = False
            else:
                # Check proximity to helmets for separate detections
                has_helmet = check_helmet_proximity(person_box, [h_box for h_box in helmets])

            if has_helmet:
                color = (0, 255, 0)  # Green for helmet compliant
                status_text = "" #"HELMET OK"
                current_helmet_compliant += 1

                # Capture compliant person image (once per person)
                if track_id not in processed_person_ids:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = os.path.join(helmet_dir, f"compliant_id_{track_id}_{timestamp}.jpg")
                    cv2.imwrite(image_filename, frame)
                    print(f"Captured compliant person: {image_filename}")

            else:
                color = (0, 0, 255)  # Red for no helmet violation
                status_text =  "" #"NO HELMET!"
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
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
            cv2.putText(frame, f"Person {track_id}", (px1, py1 - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            cv2.putText(frame, status_text, (px1, py1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

            # Add zone indicator
            #cv2.putText(frame, "IN CONSTRUCTION ZONE", (centroid[0] - 80, centroid[1]),
                      #cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

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
    print("=== HELMET DETECTION SUMMARY (Using Specialized Model) ===")
    print(f"Total people processed: {len(processed_person_ids)}")
    print(f"Total helmet violations detected: {len(helmet_violations)}")
    print(f"Compliance rate: {((len(processed_person_ids) - len(helmet_violations)) / max(len(processed_person_ids), 1)) * 100:.1f}%")

if __name__ == '__main__':
    # Replace with your construction site video path
    video_file = r"D:\Yolo\video\People working at construction site.mp4"
    process_video(video_file)