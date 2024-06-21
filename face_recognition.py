import cv2
from deepface import DeepFace

# Load the reference image for face verification
reference_img = cv2.imread("image1.jpg")

def check_face(frame):
    try:
        result = DeepFace.verify(frame, reference_img, enforce_detection=False)
        if result['verified']:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Initialize video capture from the default camera (laptop's camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Perform face verification
    match = check_face(frame)

    # Display the frame with match result
    if match:
        cv2.putText(frame, "MATCH!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO MATCH!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Face Verification', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
