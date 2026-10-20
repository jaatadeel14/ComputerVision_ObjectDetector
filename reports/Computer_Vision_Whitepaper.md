# Real-Time Computer Vision Pipeline for Object Detection and Shape Analysis

**Project Title:** Real-Time Computer Vision Pipeline for Object Detection and Shape Analysis  
**Technologies:** Python, OpenCV, YOLO11n, NumPy, CSV  
**Model:** YOLO11n Pre-trained Object Detection Model  
**Author:** Adeel Shabbir  
**Date:** October 2026

---

## Abstract

This project presents a real-time computer vision pipeline that combines traditional image processing techniques with deep learning-based object detection. The system uses OpenCV to capture video frames from a webcam and applies Gaussian filtering, grayscale conversion, adaptive thresholding, and contour extraction to identify image regions that satisfy predefined pixel-area constraints. In parallel, a pre-trained YOLO11n model detects and classifies objects in the video stream.

The system dynamically counts detected objects by class and displays relevant information, including total object count, processing time, and estimated processing frames per second (FPS). It also records performance metrics in a CSV file for subsequent analysis. The objective is to demonstrate how conventional computer vision methods and modern deep learning can work together in a single real-time application.

**Keywords:** Computer Vision, Object Detection, YOLO11n, OpenCV, Gaussian Filtering, Adaptive Thresholding, Contour Detection, Real-Time Processing.

## 1. Introduction

Computer vision enables computers to extract meaningful information from images and videos. It is widely used in applications such as surveillance, robotics, industrial inspection, traffic monitoring, and automated quality control.

The purpose of this project is to develop a real-time computer vision system that processes webcam footage and performs two complementary tasks. First, traditional image processing techniques are used to identify shapes and regions of interest. Second, a pre-trained YOLO11n deep learning model detects and classifies recognizable objects.

Using both approaches provides an opportunity to compare their roles. Traditional image processing can identify boundaries and shapes based on pixel information, while YOLO can recognize semantic object categories such as people, bottles, chairs, and other supported classes.

The project also incorporates performance logging so that processing speed and detection results can be evaluated using recorded data rather than visual observations alone.

### 1.1 Project Objectives

The primary objectives are:

- Develop a webcam-based real-time video processing pipeline.
- Apply Gaussian filtering to reduce image noise.
- Use adaptive thresholding to create a binary image.
- Extract contours and filter regions according to pixel-area bounds.
- Detect and classify objects using a pre-trained YOLO11n model.
- Generate dynamic object counts for detected classes.
- Display processing time and estimated FPS.
- Save frame-level performance measurements in CSV format.
- Evaluate system performance and identify limitations.

## 2. System Architecture and Methodology

The application is implemented in Python and uses OpenCV for image processing and webcam access, together with Ultralytics YOLO for deep learning-based object detection.

The system follows two parallel processing paths after a video frame is captured.

### 2.1 Processing Pipeline

**Traditional image processing path:**

1. Capture a frame from the webcam.
2. Apply Gaussian blur.
3. Convert the blurred image to grayscale.
4. Apply adaptive thresholding.
5. Extract external contours.
6. Calculate contour areas and bounding rectangles.
7. Filter contours using minimum and maximum area constraints.
8. Display the extracted shapes and their measurements.

**Deep learning path:**

1. Receive the captured video frame.
2. Run YOLO11n object detection with a confidence threshold of 0.25.
3. Retrieve the predicted class of each detected object.
4. Count detections by class.
5. Draw detection boxes and labels.
6. Calculate and display processing time and estimated FPS.
7. Save performance data to a CSV file.

Both paths operate on the same captured frame, allowing shape analysis and object detection to be observed together.

## 3. Image Processing Techniques

### 3.1 Gaussian Filtering

Gaussian filtering is applied using OpenCV's `GaussianBlur()` function with a 5 × 5 kernel.

The filter smooths local pixel variations and reduces some forms of image noise. This can make subsequent thresholding and contour extraction more stable.

The Gaussian filter can be represented as:

\[
G(x,y)=\frac{1}{2\pi\sigma^2}
e^{-\frac{x^2+y^2}{2\sigma^2}}
\]

Here, \(G(x,y)\) represents the Gaussian weighting function and \(\sigma\) controls the spread of the filter.

Although Gaussian blur can reduce noise, it may also soften fine boundaries. The selected kernel provides a practical starting point for webcam-based processing.

### 3.2 Grayscale Conversion

The blurred color frame is converted into grayscale using OpenCV's `cvtColor()` function.

Grayscale conversion reduces the image to a single intensity channel. This simplifies the thresholding stage because it removes the need to process separate red, green, and blue channels.

The grayscale image serves as the input for adaptive thresholding.

### 3.3 Adaptive Thresholding

Adaptive thresholding converts the grayscale image into a binary image. Instead of applying one threshold to the entire frame, the method calculates thresholds using local image neighborhoods.

The implementation uses:

- Adaptive method: Gaussian-weighted local threshold.
- Threshold type: Binary.
- Neighborhood block size: 11.
- Constant subtraction value: 2.

This approach can be useful under uneven lighting because different regions of the image can receive different threshold values.

However, thresholding is sensitive to shadows, texture, reflections, background patterns, and lighting changes. As a result, the binary image may contain noise or fragmented regions.

### 3.4 Contour Extraction

After thresholding, the system uses OpenCV's `findContours()` function to retrieve external contours.

For each contour, the application calculates its area and bounding rectangle. The contour area is used to determine whether the region satisfies the configured size constraints.

The minimum contour area is set to 500 pixels. The maximum is set to 80% of the total frame area.

Contours within these bounds are drawn on a copy of the original frame. Their areas and bounding-box dimensions are displayed for inspection.

These measurements provide basic geometric information about image regions. However, a contour does not necessarily correspond to one real-world object. A single object may generate several contours, and multiple objects may merge into one region.

## 4. YOLO11n Object Detection

### 4.1 Model Overview

The project uses YOLO11n, a lightweight model provided through the Ultralytics framework. The model is loaded using the pretrained weights file `yolo11n.pt`.

YOLO is a deep learning-based object detection approach that predicts object locations and class labels within an image. Unlike basic thresholding, it learns visual features that help distinguish object categories.

The pretrained model supports the classes represented in its training dataset. It does not automatically recognize every possible object, and it has not been specifically trained on this project's webcam environment.

### 4.2 Confidence Threshold

The application uses a confidence threshold of 0.25.

Predictions below the configured threshold are excluded from the returned detections. Lower thresholds may increase the number of detections but can also introduce more false positives. Higher thresholds may reduce false detections while missing some genuine objects.

The appropriate threshold depends on the application and should be selected through testing.

### 4.3 Dynamic Object Counting

The system maintains a dynamic counter for each detected class. For every detection, the predicted class identifier is mapped to its corresponding class name, and the count is incremented.

For example, if the model detects three people and two bottles in a frame, the displayed class counts would be:

- Person: 3
- Bottle: 2
- Total objects: 5

These values are illustrative examples, not measured results from the current project.

The counts are recalculated for each frame. Consequently, they represent detections in the current frame rather than unique objects tracked across time. An object visible in consecutive frames may be counted repeatedly when considering the full video.

## 5. Performance Measurement and Logging

### 5.1 Processing Time

The program uses Python's high-resolution performance counter to measure the time required for the processing operations within each loop iteration.

Processing time is converted from seconds into milliseconds:

\[
T_{\mathrm{ms}}=T_{\mathrm{seconds}}\times1000
\]

A lower processing time generally indicates that the measured pipeline iteration completes more quickly.

The measured interval includes the operations between the timer start and stop, including traditional image processing, YOLO inference, and associated result preparation. It does not necessarily include every component of end-to-end camera latency or display refresh.

### 5.2 Estimated Frames Per Second

The program estimates processing FPS using the reciprocal of the measured processing time:

\[
FPS\approx\frac{1}{T_{\mathrm{seconds}}}
\]

For example, if the measured processing time is 0.10 seconds, the estimated processing rate is approximately 10 FPS.

This is a processing-rate estimate rather than a separately measured camera capture or display frame rate. Actual end-to-end throughput can differ because of camera delays, rendering, hardware, and other system overhead.

### 5.3 CSV Performance Logging

The program stores performance measurements in:

`outputs/performance_log.csv`

The CSV file contains the following fields:

| Field | Description |
|---|---|
| Timestamp | Time when the frame's measurements were recorded |
| Detected_Objects | Class names and their detection counts |
| Total_Objects | Total detections in the current frame |
| Extracted_Shapes | Number of contours satisfying the area constraints |
| Processing_Time_ms | Measured processing time in milliseconds |
| FPS | Estimated processing frames per second |

Recording this information enables quantitative evaluation over multiple frames. The CSV file can be analyzed using Python, pandas, spreadsheet software, or visualization tools.

## 6. Experimental Evaluation

The system can be evaluated by running the webcam application under different conditions, such as varying lighting, object distance, background complexity, and the number of visible objects.

The main evaluation metrics are processing time, estimated FPS, total detections, class counts, and extracted contour counts.

### 6.1 Recommended Experiments

**Experiment A: Basic object detection**

Place one or more supported objects in front of the webcam and observe the detection labels and bounding boxes.

**Experiment B: Multiple objects**

Introduce several objects of different supported classes and observe how the per-class counters change.

**Experiment C: Lighting variation**

Run the application under bright, dim, and uneven lighting conditions. Observe the effects on threshold images and contour extraction.

**Experiment D: Background complexity**

Compare results against a plain background and a textured background. Record whether extra contours or false object detections appear.

**Experiment E: Processing performance**

Run the application for a fixed duration and analyze the CSV log to calculate mean processing time, average estimated FPS, and the number of recorded frames.
### 6.2 Measured Performance Results

The system was evaluated using 294 recorded video frames.

| Metric | Measured Result |
|---|---:|
| Total recorded frames | 294 |
| Average processing time | 78.12 ms |
| Minimum processing time | 44.87 ms |
| Maximum processing time | 1477.22 ms |
| Average estimated FPS | 15.15 |
| Minimum estimated FPS | 0.68 |
| Maximum estimated FPS | 22.29 |
| Average detected objects per frame | 1.02 |
| Average extracted shapes per frame | 1.55 |

### 6.3 Results Interpretation

The system processed 294 recorded frames with an average processing time of 78.12 milliseconds and an average estimated processing rate of 15.15 FPS. The maximum processing time was 1477.22 milliseconds, showing that some processing iterations took considerably longer than average.

The average number of detected objects was 1.02 per frame, while the average number of extracted shapes was 1.55 per frame. These measurements represent different outputs because YOLO detects object classes, whereas contour extraction identifies image regions based on geometric and pixel-area criteria.

The results demonstrate the integration of real-time object detection, image processing, and performance logging. Additional testing under different lighting and background conditions would help evaluate the system more comprehensively.
<!-- ### 6.2 Results Table

Complete the following table after collecting real measurements. Do not enter estimated values as experimental results.

| Metric | Measured result |
|---|---|
| Total recorded frames | To be measured |
| Average processing time (ms) | To be measured |
| Average estimated FPS | To be measured |
| Minimum processing time (ms) | To be measured |
| Maximum processing time (ms) | To be measured |
| Average detected objects per frame | To be measured |
| Average extracted shapes per frame | To be measured | -->

### 6.3 Interpretation of Results

The measured results should be used to assess whether the pipeline meets the intended performance requirements.

Average processing time and FPS describe computational performance, while object counts and visual inspection help characterize detection behavior. Performance may vary depending on CPU or GPU capability, input resolution, the number of detections, and the runtime environment.

The contour count and YOLO detection count should not be treated as interchangeable measurements because they are produced by different methods and represent different things.

## 7. Limitations

Several limitations should be considered when interpreting the output of this system.

**Dependence on pretrained classes:** YOLO11n is limited by the classes and visual patterns learned during training. Unfamiliar objects may not be recognized correctly.

**Detection errors:** Small, distant, partially hidden, or poorly illuminated objects may be missed. Similar-looking objects may also be confused.

**No object tracking:** The current implementation recalculates detections independently for each frame. It does not assign persistent identifiers to objects across frames.

**Contour noise:** Adaptive thresholding may produce unwanted contours because of shadows, textures, and changes in illumination. Area filtering removes some regions but does not eliminate all noise.

**Hardware dependence:** Processing speed varies across systems and may be limited by CPU performance, available GPU acceleration, and other software running simultaneously.

**Measurement scope:** The reported FPS is calculated from the measured processing interval. It should not be interpreted as a complete end-to-end latency measurement.

**Evaluation data:** Reliable claims about accuracy require labeled test images or videos and appropriate metrics. Visual inspection alone is insufficient to establish precision, recall, or mean average precision.

## 8. Future Improvements

The project can be extended in several ways:

- Add object tracking to maintain identities across consecutive frames.
- Evaluate detection quality using labeled images and ground-truth annotations.
- Calculate precision, recall, and mean average precision where appropriate.
- Compare different confidence thresholds under consistent test conditions.
- Add configurable contour-area limits and noise-reduction methods.
- Optimize inference using suitable hardware acceleration.
- Generate performance graphs from the recorded CSV data.
- Add configurable camera resolution and structured experiment settings.
- Separate detection, logging, and visualization into reusable modules.

These improvements would make the system easier to test, maintain, and adapt to more demanding applications.

## 9. Conclusion

This project demonstrates a combined computer vision pipeline that integrates traditional image processing with deep learning-based object detection. Gaussian filtering, grayscale conversion, adaptive thresholding, and contour extraction provide geometric analysis of image regions, while YOLO11n supplies class-based object detections.

Dynamic counters and CSV performance logging make the system suitable for practical experimentation. The recorded processing time and estimated FPS provide a basis for analyzing computational performance, while the contour and object detection outputs illustrate the complementary strengths and limitations of traditional and learned visual processing methods.

The next stage of development is to collect reproducible experimental data, analyze the performance log, and incorporate measured results into the final evaluation. With additional testing, tracking, and accuracy metrics, the pipeline can be developed into a more comprehensive real-time computer vision application.

## References

1. Ultralytics. *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/
2. OpenCV. *OpenCV Documentation*. https://docs.opencv.org/
3. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
4. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd edition. Springer.
