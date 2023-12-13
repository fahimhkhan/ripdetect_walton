Overview
This Python script utilizes advanced object detection techniques for identifying rip currents in video streams. The script employs YOLO (You Only Look Once) v8, a state-of-the-art, real-time object detection system, to analyze video footage and detect the presence of rip currents. This application is particularly useful for enhancing safety measures in beach areas by providing real-time rip current alerts.

Installation and Setup
Before running the script, ensure you have Python installed on your system. The script depends on two primary libraries: OpenCV and Ultralytics YOLO. You can install these dependencies using pip:

Copy code
pip install opencv-python
pip install ultralytics
Usage
The script is designed to continuously process video streams for rip current detection. By default, it connects to an online video stream but can be modified to analyze local video files or other stream URLs.

Upon detecting a rip current, the script logs the detection with a timestamp and the confidence score into a CSV file named after the current date.

Key Components:
OpenCV (cv2): For handling video stream capture and image processing.
Ultralytics YOLO: For performing the object detection tasks.
Datetime and Time: For timestamping detections.
Execution
To run the script, simply execute it in a Python environment:

Copy code
python rip_current_detection.py
Customization
You can customize the script to suit different use cases. For example, you can change the video source by modifying the cap = cv2.VideoCapture(...) line with a different stream URL or a local video file path.

Output
The script displays a window showing the live processed video with bounding boxes around detected rip currents. Detection details are logged in a CSV file with timestamps and confidence scores.

Disclaimer
This script is for educational and developmental purposes. The accuracy of rip current detection is subject to various factors, including video quality, environmental conditions, and model limitations.

Contributing
Contributions to enhance the script's functionality and performance are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

License
This project is open-source and available under the MIT License.

Acknowledgments
Thanks to Ultralytics for the YOLO v8 model.
Special thanks to OpenCV for their powerful image processing capabilities.
