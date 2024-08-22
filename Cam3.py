import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

def calculate_angle(center_point, object_center):
    delta_y = center_point[1] - object_center[1]
    delta_x = center_point[0] - object_center[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    adjusted_angle = 90 - angle_deg
    if adjusted_angle < 0:
        adjusted_angle += 360
    return adjusted_angle

model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture("C:\\MCWClusterWorks\\CAS\\Cam3.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w)
out = cv2.VideoWriter("new4.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Reference point is now the bottom-right corner of the frame
center_point = (w, h)
angle_history = {}  # To store angle history for each tracked object
distance_history = {}  # To store distance history for each tracked object
crossed_middle = {}  # To track if the object has crossed the middle of the frame
crossed_left_margin = {}  # To track if the object has crossed the left margin of the frame
collision_alerts = {}  # To store the collision alert state for each object

class_width_mapping = {
    'person': 3.7,
    'bicycle': 3.3,
    'car': 9.2,
    'motorcycle': 3.3,
    'bus': 15.2,
    'truck': 13.3,
    'cat': 0.8,
    'dog': 1.5,
    'cow': 2
}

def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    return (real_object_width * Focal_Length) / object_width_in_frame

# Position of the vertical line near the left margin
left_margin = 300
middle_frame = w // 2

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            object_width_in_frame = box[2] - box[0]
            known_width = class_width_mapping.get(names[int(cls)], 1.5)

            Focal_length_dynamic = FocalLength(known_width, known_width, 182)
            Distance = Distance_finder(Focal_length_dynamic, known_width, object_width_in_frame)
        
            annotator.box_label(box, label=str(track_id), color=(0, 255, 0))  # Green color for bounding box

            # Calculate the center of the bounding box
            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            box_center = (int(box_center[0]), int(box_center[1]))  # Convert to integers

            cv2.putText(im0, f"Distance: {Distance:.2f}m", (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw the vertical line intersecting the reference point
            cv2.line(im0, center_point, (center_point[0], 0), (0, 255, 0), 2)

            # Draw the vertical line near the left margin
            cv2.line(im0, (left_margin, 0), (left_margin, h), (0, 0, 255), 2)
            # Draw the vertical line in the middle of the frame
            cv2.line(im0, (middle_frame, 0), (middle_frame, h), (255, 0, 0), 2)

            # Draw the line from the reference point to the center of the bounding box
            cv2.line(im0, center_point, box_center, (255, 0, 0), 2)

            # Calculate the angle
            angle = calculate_angle(center_point, box_center)

            # Update angle and distance history for the current object
            if track_id not in angle_history:
                angle_history[track_id] = []
            angle_history[track_id].append(angle)
            if len(angle_history[track_id]) > 10:  # Limit history to last 10 angles
                angle_history[track_id].pop(0)

            if track_id not in distance_history:
                distance_history[track_id] = []
            distance_history[track_id].append(Distance)
            if len(distance_history[track_id]) > 10:  # Limit history to last 10 distances
                distance_history[track_id].pop(0)

            # Track if the object has crossed the middle of the frame
            if track_id not in crossed_middle:
                crossed_middle[track_id] = False
            if box_center[0] < middle_frame:
                crossed_middle[track_id] = True

            # Track if the object has crossed the left margin
            if track_id not in crossed_left_margin:
                crossed_left_margin[track_id] = False
            if box_center[0] < left_margin:
                crossed_left_margin[track_id] = True

            # Check if the object has crossed the middle of the frame and moved from right to left
            if crossed_middle[track_id] and len(angle_history[track_id]) > 1 and all(x < y for x, y in zip(angle_history[track_id], angle_history[track_id][1:])):
                moved_from_right_to_left_middle = True
            else:
                moved_from_right_to_left_middle = False

            # Check if the object has crossed the left margin and moved from right to left
            if crossed_left_margin[track_id] and len(angle_history[track_id]) > 1 and all(x < y for x, y in zip(angle_history[track_id], angle_history[track_id][1:])):
                moved_from_right_to_left_margin = True
            else:
                moved_from_right_to_left_margin = False

            # Trigger alert if the object crossed the middle of the frame from right to left
            if (moved_from_right_to_left_middle or moved_from_right_to_left_margin) and Distance < 10:
                collision_alerts[track_id] = True  # Set collision alert for the object
                cv2.putText(im0, "Collision", (int(box_center[0]), int(box_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red collision label
                # Change bounding box color to red
                annotator.box_label(box, label=str(track_id), color=(0, 0, 255))
                angle_text = f"Angle: {angle:.2f} deg"
                cv2.putText(im0, angle_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif track_id in collision_alerts and collision_alerts[track_id]:
                # If a collision alert was previously set for this object, keep showing the alert
                cv2.putText(im0, "Collision", (int(box_center[0]), int(box_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Change bounding box color to red
                annotator.box_label(box, label=str(track_id), color=(0, 0, 255))
                angle_text = f"Angle: {angle:.2f} deg"
                cv2.putText(im0, angle_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Annotate the angle on the frame
                angle_text = f"Angle: {angle:.2f} deg"
                cv2.putText(im0, angle_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out.write(im0)
    cv2.imshow("visioneye-pinpoint", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
