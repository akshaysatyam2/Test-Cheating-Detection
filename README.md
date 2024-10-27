# Test Cheating Detection

Welcome to the AI Vision-Based Person Detection and Head Orientation Classification project! This tool leverages deep learning and computer vision to detect people, identify communication devices in an image, and classify the direction in which a person is looking based on head orientation landmarks. This README provides an overview of the setup, methodology, and usage of this project.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Methodology](#methodology)
- [How to Use](#how-to-use)
- [Sample Results](#sample-results)
- [Future Improvements](#future-improvements)

---

## Introduction

The primary goal of this project is to analyze images, count the number of people, identify if any communication devices are present, and classify head orientation direction (left, right, up, down, or at-system) for individuals in the image. 

This type of analysis can be useful in various scenarios, such as monitoring attentiveness, surveillance, and interactive systems that adapt based on user direction.

---

## Technologies Used

- **Python 3.12** - Programming Language
- **OpenCV** - For image processing and visualization
- **YOLOv8** - Pre-trained model for object detection
- **MediaPipe** - For human pose and facial landmark detection
- **Math and JSON** - For processing coordinates and formatting outputs

---

## Setup and Installation

### Prerequisites

1. **Python 3.12**: Install Python from [Python's official website](https://www.python.org/).
2. **Required Libraries**: You can install the necessary packages using pip.

### Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/akshaysatyam2/Test-Cheating-Detection.git
    ```
2. Install the required libraries:
    ```bash
    pip install opencv-python ultralytics mediapipe
    ```

---

## Methodology

1. **Person Detection**: YOLOv8 is used to detect objects in the image, particularly focusing on identifying people and specific communication devices (like laptops, remotes, cell phones).
  
2. **Head Orientation Estimation**:
    - MediaPipe’s Pose model is applied to detect facial landmarks, such as the nose, eyes, and ears.
    - These landmarks help in calculating distances to estimate head orientation in multiple directions.
    - Based on the relative positions of these points, the code classifies whether the person is looking directly at the system, or towards the left, right, up, or down.

3. **Result Compilation**:
    - The output includes the count of people, presence of a communication device, and head orientation direction for each detected person.

---

## How to Use

1. Place the images to be processed in a designated folder (e.g., `/content`).
2. Run the `classify_image()` function, providing the path to each image.
3. The code processes each image, displays annotated results, and outputs a JSON with the following information:
    - `people_count`: Number of people in the image.
    - `direction_looking`: Direction of gaze (left, right, up, down, or at-system).
    - `communication_device_present`: Boolean indicating if a communication device is present.

---

## Sample Results

The sample output will look something like this:

```json
{
    "people_count": 1,
    "direction-looking": "right",
    "communication_device_present": true
}
```

The displayed image will show annotated points on detected facial landmarks, aiding in visual confirmation of the calculated gaze direction.

---

## Future Improvements

There are various areas where the project can be enhanced:

1. **Extended Head Pose Estimation**: Incorporate more detailed 3D head pose estimation for finer orientation details.
2. **Real-Time Processing**: Adapt the code for real-time video input or camera feed.
3. **Device-Specific Classification**: Extend communication device detection to a broader range of items, potentially with fine-grained classification.
4. **Model Training**: Currently we are using base model of Yolo and Mediapipe's Pose model so we can finetune or use better models for our tasks.

---


Thank you for using this project! If you encounter any issues or have suggestions, please feel free to reach out.
