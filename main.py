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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can choose a different model, e.g., 'yolov8m.pt'

    # Define the detection zone (example: a rectangle)
    # You can change these coordinates to define your desired zone
    detection_zone = [(0, 200), (800, 200), (800, 800), (0, 800)]

    # Create a directory for captured images if it doesn't exist
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)

    # Set to keep track of objects for which an image has been captured
    captured_object_ids = set()

    # Set to keep track of unique vehicles that have passed through the zone
    unique_vehicles_in_zone_history = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize count for objects in the zone for this frame
        current_objects_in_zone_count = 0

        # Perform object detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Process results
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            if boxes.id is not None: # Ensure tracking IDs are present
                for box in boxes:
                    class_name = "unknown" # Initialize class_name with a default value
                    # Get bounding box coordinates and tracking ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) # Tracking ID

                    # Calculate centroid of the bounding box
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    object_centroid = (centroid_x, centroid_y)

                    # Get class name
                    class_id = int(box.cls)
                    class_name = model.names[class_id] # This will re-assign if successful

                    # Check if the object's centroid is within the detection zone
                    if is_point_in_polygon(object_centroid, detection_zone):
                        color = (0, 0, 255)  # Red for objects in zone
                        in_zone_text = f"ID {track_id} IN ZONE"
                        cv2.putText(frame, in_zone_text, (centroid_x - 50, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        current_objects_in_zone_count += 1

                        # If the object is a 'car' or 'truck' and in the zone, add its ID to the history
                        if class_name in ["car", "truck"]:
                            unique_vehicles_in_zone_history.add(track_id)

                        # Capture and save image if this object is new to the zone
                        if track_id not in captured_object_ids:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_filename = os.path.join(output_dir, f"object_id_{track_id}_{timestamp}.jpg")
                            cv2.imwrite(image_filename, frame)
                            print(f"Captured image: {image_filename}")
                            captured_object_ids.add(track_id)
                    else:
                        color = (255, 0, 0)  # Blue for objects outside zone

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the count of objects in the zone
        cv2.putText(frame, f"Objects in Zone: {current_objects_in_zone_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the total count of unique vehicles that have passed through the zone
        cv2.putText(frame, f"Total Vehicles Through Zone: {len(unique_vehicles_in_zone_history)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red color

        # Draw the detection zone on the frame
        cv2.polylines(frame, [np.array(detection_zone)], True, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_file = r"D:/Yolo/video/Road traffic video for object recognition.mp4"  # Replace with your video file path
    process_video(video_file)
