
import cv2
import time
import csv
import os
from datetime import datetime
from collections import Counter
from ultralytics import YOLO


def main():
    # Load the pre-trained YOLO model
    model = YOLO("yolo11n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        print("Check camera permissions or try another camera.")
        return

    # Prepare output folder and performance log
    os.makedirs("outputs", exist_ok=True)
    log_file = os.path.join("outputs", "performance_log.csv")

    # Write CSV header if the file is new or empty
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Timestamp",
                "Detected_Objects",
                "Total_Objects",
                "Extracted_Shapes",
                "Processing_Time_ms",
                "FPS"
            ])

    print("Computer Vision Pipeline Started")
    print("Performance results will be saved to:", log_file)
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read a frame from webcam.")
                break

            start_time = time.perf_counter()

            # 1. Gaussian filter
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)

            # 2. Grayscale conversion
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

            # 3. Adaptive thresholding
            threshold = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # 4. Contour extraction
            contours, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 5. Filter shapes using pixel-area bounds
            contour_frame = frame.copy()
            min_area = 500
            max_area = int(frame.shape[0] * frame.shape[1] * 0.80)
            shape_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)

                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    cv2.drawContours(
                        contour_frame, [contour], -1, (0, 255, 0), 2
                    )
                    cv2.rectangle(
                        contour_frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 0, 0),
                        2
                    )

                    label = f"Area:{int(area)} W:{w} H:{h}"
                    cv2.putText(
                        contour_frame,
                        label,
                        (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1
                    )
                    shape_count += 1

            cv2.putText(
                contour_frame,
                f"Extracted shapes: {shape_count}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # 6. YOLO object detection
            results = model.predict(
                source=frame,
                conf=0.25,
                verbose=False
            )

            # Count detected objects by class
            counts = Counter()

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    counts[class_name] += 1

            # Draw YOLO boxes and labels
            output = results[0].plot()

            # Calculate processing time and approximate processing FPS
            processing_time = time.perf_counter() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0

            # Display performance metrics
            y_position = 30

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            y_position += 30

            cv2.putText(
                output,
                f"Processing: {processing_time * 1000:.1f} ms",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            y_position += 30

            cv2.putText(
                output,
                f"Total objects: {sum(counts.values())}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Display dynamic class counts
            for class_name, count in sorted(counts.items()):
                y_position += 25

                if y_position > output.shape[0] - 10:
                    break

                cv2.putText(
                    output,
                    f"{class_name}: {count}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            # 7. Save performance data for this frame
            detected_objects = "; ".join(
                f"{name}:{count}" for name, count in sorted(counts.items())
            )

            with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    detected_objects,
                    sum(counts.values()),
                    shape_count,
                    round(processing_time * 1000, 2),
                    round(fps, 2)
                ])

            # Display all three pipeline stages
            cv2.imshow("Object Detection Pipeline", output)
            cv2.imshow("Adaptive Threshold", threshold)
            cv2.imshow("Contour Extraction", contour_frame)

            # Press Q on a video window to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
        print("Performance log saved to:", log_file)


if __name__ == "__main__":
    main()
