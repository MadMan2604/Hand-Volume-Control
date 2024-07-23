import cv2
import mediapipe as mp
import subprocess
import time

# Initialize variables
x1 = y1 = x2 = y2 = 0
prev_time = time.time()
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
my_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

while True:
    # Read frame from webcam
    ret, image = webcam.read()
    if not ret:
        break

    # Flip image horizontally
    image = cv2.flip(image, 1)

    # Convert image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hand detection and tracking
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                if id == 8:
                    cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
                    x1 = x
                    y1 = y
                if id == 4:
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                    x2 = x
                    y2 = y

            # Use distance between two points to calculate volume control
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 4
            if dist > 50:
                # Increase volume
                subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 5)"])
            else:
                # Decrease volume
                subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 5)"])

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Hand volume control using python", image)

    # Exit if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
