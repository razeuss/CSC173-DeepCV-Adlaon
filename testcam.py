import cv2
import time

def test_camera_index(index, backend=None):
    print(f"\n--- Testing camera index {index} backend {backend} ---")
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        print(f"❌ cap.isOpened() is False for index {index}, backend={backend}")
        return

    print("✅ cap.isOpened() is True, warming up...")

    ok_any = False
    for i in range(30):  # try 30 frames before giving up
        ret, frame = cap.read()
        if not ret:
            print(f"  [try {i+1}] ret=False")
            time.sleep(0.05)
            continue
        print(f"  [try {i+1}] ret=True, frame.shape={frame.shape}")
        ok_any = True
        cv2.imshow(f"Camera {index} backend {backend}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if not ok_any:
        print(f"❌ Never got a valid frame on index {index}, backend={backend}")
    else:
        print(f"✅ Got at least one valid frame on index {index}, backend={backend}")

    cap.release()
    cv2.destroyAllWindows()


print("OpenCV version:", cv2.__version__)
print("Testing camera combos...")

# Try default backend and AVFoundation for indices 0–2
for idx in [0, 1, 2]:
    test_camera_index(idx, cv2.CAP_AVFOUNDATION)
    test_camera_index(idx, cv2.CAP_ANY)

print("\nDone testing all indices.")
