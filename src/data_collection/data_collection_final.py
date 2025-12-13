import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Use the working camera index and backend from test_cam_full.py
capture = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not capture.isOpened():
    print("❌ Could not open camera on index 1. Check permissions.")


# Hand detectors
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Current letter directory (A–Z)
c_dir = 'A'

# Base paths (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE_DIR = os.path.join(BASE_DIR, "data/raw/AtoZ_3.2")

# Ensure base save dir exists
os.makedirs(SAVE_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_BASE_DIR, c_dir), exist_ok=True)

# Count existing images for current letter
count = len(os.listdir(os.path.join(SAVE_BASE_DIR, c_dir)))

offset = 15
step = 1
flag = False
suv = 0

# Prepare white background
white_path = os.path.join(BASE_DIR, "white.jpg")
if not os.path.exists(white_path):
    white_img = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(white_path, white_img)

while True:
    try:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread(white_path)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure ROI is within frame bounds
            y1 = max(y - offset, 0)
            y2 = min(y + h + offset, frame.shape[0])
            x1 = max(x - offset, 0)
            x2 = min(x + w + offset, frame.shape[1])

            image = np.array(frame[y1:y2, x1:x2])

            handz, imz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand2 = handz[0]
                pts = hand2['lmList']

                os_shift = ((400 - w) // 2) - 15
                os1_shift = ((400 - h) // 2) - 15

                # Draw finger bones on white
                for t in range(0, 4):
                    cv2.line(
                        white,
                        (pts[t][0] + os_shift, pts[t][1] + os1_shift),
                        (pts[t + 1][0] + os_shift, pts[t + 1][1] + os1_shift),
                        (0, 255, 0),
                        3
                    )
                for t in range(5, 8):
                    cv2.line(
                        white,
                        (pts[t][0] + os_shift, pts[t][1] + os1_shift),
                        (pts[t + 1][0] + os_shift, pts[t + 1][1] + os1_shift),
                        (0, 255, 0),
                        3
                    )
                for t in range(9, 12):
                    cv2.line(
                        white,
                        (pts[t][0] + os_shift, pts[t][1] + os1_shift),
                        (pts[t + 1][0] + os_shift, pts[t + 1][1] + os1_shift),
                        (0, 255, 0),
                        3
                    )
                for t in range(13, 16):
                    cv2.line(
                        white,
                        (pts[t][0] + os_shift, pts[t][1] + os1_shift),
                        (pts[t + 1][0] + os_shift, pts[t + 1][1] + os1_shift),
                        (0, 255, 0),
                        3
                    )
                for t in range(17, 20):
                    cv2.line(
                        white,
                        (pts[t][0] + os_shift, pts[t][1] + os1_shift),
                        (pts[t + 1][0] + os_shift, pts[t + 1][1] + os1_shift),
                        (0, 255, 0),
                        3
                    )

                cv2.line(white, (pts[5][0] + os_shift, pts[5][1] + os1_shift),
                         (pts[9][0] + os_shift, pts[9][1] + os1_shift), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os_shift, pts[9][1] + os1_shift),
                         (pts[13][0] + os_shift, pts[13][1] + os1_shift), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os_shift, pts[13][1] + os1_shift),
                         (pts[17][0] + os_shift, pts[17][1] + os1_shift), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_shift, pts[0][1] + os1_shift),
                         (pts[5][0] + os_shift, pts[5][1] + os1_shift), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_shift, pts[0][1] + os1_shift),
                         (pts[17][0] + os_shift, pts[17][1] + os1_shift), (0, 255, 0), 3)

                skeleton1 = np.array(white)

                for i in range(21):
                    cv2.circle(
                        white,
                        (pts[i][0] + os_shift, pts[i][1] + os1_shift),
                        2,
                        (0, 0, 255),
                        1
                    )

                cv2.imshow("Skeleton", skeleton1)

        frame = cv2.putText(
            frame,
            f"dir={c_dir}  count={count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1) & 0xFF

        if interrupt == 27:  # ESC
            break

        # Next letter
        if interrupt == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            flag = False
            os.makedirs(os.path.join(SAVE_BASE_DIR, c_dir), exist_ok=True)
            count = len(os.listdir(os.path.join(SAVE_BASE_DIR, c_dir)))

        # Toggle auto-save (a)
        if interrupt == ord('a'):
            if flag:
                flag = False
            else:
                suv = 0
                flag = True

        # Auto-saving captured skeleton images
        if flag:
            if suv == 180:
                flag = False
            if step % 3 == 0 and 'skeleton1' in locals():
                save_dir = os.path.join(SAVE_BASE_DIR, c_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv += 1
            step += 1

    except Exception:
        print("Error in loop:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
