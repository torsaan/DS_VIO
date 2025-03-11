#Johanna test script

import cv2

video_path = "./Data/Processed/standardized/Violence/V_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or frame read error.")
        break
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
