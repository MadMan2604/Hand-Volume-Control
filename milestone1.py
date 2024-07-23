import cv2
import numpy as np

# Function to check if thumb and index finger are touching
def fingers_touching(contour, thumb_tip, index_tip):
    for point in contour:
        dist_thumb = np.linalg.norm(thumb_tip - point[0])
        dist_index = np.linalg.norm(index_tip - point[0])
        if dist_thumb < 50 and dist_index < 50:  # Adjust this threshold as needed
            return True
    return False

# Function to count fingers using convexity defects
def count_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                if angle <= np.pi / 2:
                    count += 1
            return count
    return 0

# Initialize camera
cap = cv2.VideoCapture(0)

# Capture a static background image
_, bg = cap.read()
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (5, 5), 0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Calculate absolute difference between background and current frame
    diff = cv2.absdiff(bg, gray_blur)
    
    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Adjust these thresholds based on your hand's size and aspect ratio
        if 1000 < area < 10000 and 0.5 < aspect_ratio < 2.0:
            # Count fingers using convexity defects
            finger_count = count_fingers(contour)
            
            # Get the coordinates of thumb and index finger tips
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            thumb_tip = contour[defects[0][0][0]][0]
            index_tip = contour[defects[1][0][0]][0]
            
            # Check if thumb and index finger are touching
            if fingers_touching(contour, thumb_tip, index_tip):
                print("Hand detected")
                
            # Draw bounding rectangle around the detected hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw contour of the detected hand
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Control', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
