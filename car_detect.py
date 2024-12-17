import cv2
import numpy as np
import os
from tkinter import filedialog, Tk
from time import time
from collections import defaultdict


class CarTracker:
    def __init__(self):
        self.car_positions = defaultdict(list)
        self.car_times = defaultdict(list)
        self.next_car_id = 0
        self.tracked_cars = {}

    def calculate_speed(self, positions, times, pixels_per_meter=10):
        if len(positions) < 2 or len(times) < 2:
            return 0
        dist = np.sqrt(
            (positions[-1][0] - positions[0][0]) ** 2 +
            (positions[-1][1] - positions[0][1]) ** 2
        )
        dist_meters = dist / pixels_per_meter
        time_diff = (times[-1] - times[0]) / 3600
        if time_diff == 0:
            return 0
        speed = (dist_meters / 1000) / time_diff
        return speed


def detect_cars():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select MP4 Video File",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )

    if not video_path:
        print("No file selected. Exiting...")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    target_height = 1080
    target_width = 1920

    output_path = os.path.splitext(video_path)[0] + '_detected.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (target_width, target_height))

    # More sensitive background subtractor
    car_detector = cv2.createBackgroundSubtractorMOG2(
        history=10,  # Reduced history for faster adaptation
        varThreshold=16,  # Lower threshold for more sensitive detection
        detectShadows=False
    )

    tracker = CarTracker()

    # Create a frame buffer for motion analysis
    frame_buffer = []
    buffer_size = 2

    print("\nProcessing video... Press 'q' to quit")
    print(f"Output will be saved to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 1080p
        frame = cv2.resize(frame, (target_width, target_height))

        # Process at reduced resolution for speed
        process_frame = cv2.resize(frame, (target_width // 2, target_height // 2))

        # Add frame to buffer
        frame_buffer.append(process_frame)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        # Apply background subtraction
        mask = car_detector.apply(process_frame)

        # Enhanced mask processing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply additional motion detection if we have enough frames
        if len(frame_buffer) >= 2:
            diff = cv2.absdiff(frame_buffer[0], frame_buffer[-1])
            motion_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(motion_mask, 15, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, motion_mask)

        # Find contours with lower threshold
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_cars = {}
        current_time = time()

        for contour in contours:
            area = cv2.contourArea(contour)
            # Reduced minimum area for earlier detection
            min_area = (target_width * target_height) // 1600

            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                x, y = x * 2, y * 2
                w, h = w * 2, h * 2

                # More permissive aspect ratio check
                aspect_ratio = float(w) / h
                if 0.2 <= aspect_ratio <= 4.0:  # Wider range
                    center = (x + w // 2, y + h // 2)

                    # More permissive matching distance
                    matched = False
                    for car_id, pos in tracker.tracked_cars.items():
                        if abs(pos[0] - center[0]) < 150 and abs(pos[1] - center[1]) < 150:
                            current_cars[car_id] = center
                            tracker.car_positions[car_id].append(center)
                            tracker.car_times[car_id].append(current_time)
                            matched = True
                            break

                    if not matched:
                        car_id = tracker.next_car_id
                        tracker.next_car_id += 1
                        current_cars[car_id] = center
                        tracker.car_positions[car_id].append(center)
                        tracker.car_times[car_id].append(current_time)

        tracker.tracked_cars = current_cars

        # Draw boxes and speeds
        for car_id, center in current_cars.items():
            speed = tracker.calculate_speed(
                tracker.car_positions[car_id],
                tracker.car_times[car_id]
            )

            x = center[0] - 50
            y = center[1] - 50
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), 2)

            if speed > 0:
                speed_text = f"{speed:.0f} km/h"
                cv2.putText(frame, speed_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Car Detection', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    print("\nProcessing complete!")


if __name__ == "__main__":
    detect_cars()