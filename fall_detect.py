import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import os
import glob


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants for fall detection
STANDING = 'standing'
SITTING = 'sitting'
LYING = 'lying'
THRESHOLD_TIME = 1  # time in seconds for pose change


def plot_image(image):
    """Plot a single image using matplotlib."""
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def classify_pose(landmarks):
    """Classify the pose based on the position of key landmarks."""
    # if landmarks[mp_pose.PoseLandmark.NOSE.value].visibility < 0.5:
    #     return None

    # if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility < 0.5 or \
    #         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility < 0.5:
    #     return None

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Compute the absolute position difference of the shoulders and hips for x and y-axis
    abs_shoulder_x = abs(left_shoulder.x - right_shoulder.x)
    abs_shoulder_y = abs(left_shoulder.y - right_shoulder.y)
    abs_hip_x = abs(left_hip.x - right_hip.x)
    abs_hip_y = abs(left_hip.y - right_hip.y)

    # print("-------Y-------",abs_shoulder_y, abs_hip_y)
    # print("-------X-------",abs_shoulder_x, abs_hip_x)

    # Classify based on the relative position of shoulders and hips
    if abs_shoulder_y < abs_shoulder_x and abs_hip_y < abs_hip_x:
        return STANDING
    elif abs_shoulder_y >= abs_shoulder_x and abs_hip_y >= abs_hip_x:
        return LYING
    else:
        return None


def detect_fall(video_source=0):
    cap = cv2.VideoCapture(video_source)
    prev_pose = None
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            current_pose = classify_pose(landmarks)

            if current_pose and prev_pose and current_pose != prev_pose:
                current_time = time.time()
                time_diff = current_time - prev_time

                if prev_pose in [STANDING] and current_pose == LYING and time_diff < THRESHOLD_TIME:
                    print("Fall detected!")
                    # Trigger alert
                    # Code to send notification or sound an alarm

                prev_time = current_time

            prev_pose = current_pose

        # Display the frame using matplotlib
        plot_image(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example call to the detect_fall function
detect_fall()


def img_detect_fall(img_path):
    try:
        frame = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            op = classify_pose(results.pose_landmarks.landmark)
            print(op)
            # print(results.pose_landmarks.landmark)

            # Draw pose landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            if op == LYING:
                plot_image(frame)
                return "fall detected"
        else:
            return "no fall"
    except AttributeError as e:
        print(e)
        return "not detected"


directories = ['fall-01-cam0-rgb', 'fall-02-cam0-rgb', 'fall-03-cam0-rgb', 'fall-04-cam0-rgb', 'fall-05-cam0-rgb']


# Function to process PNG files in a given directory
def png_fall(folder):
    # Using glob to find all .png files recursively within the directory
    png_files = glob.glob(os.path.join(folder, '**', '*.png'), recursive=True)
    print(len(png_files))

    # Iterate through all found .png files
    for png_file in png_files:
        res = img_detect_fall(png_file)
        if res == "fall detected":
            return


# Iterate over each directory and process PNG files
# for directory in directories:
    # png_fall(directory)
