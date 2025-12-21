# Importing Libraries
import numpy as np
import math
import cv2
import os
import traceback
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk

# Offsets / constants
offset = 29

# Base dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHITE_PATH = os.path.join(BASE_DIR,"assets", "white.jpg")

# Ensure white.jpg exists
if not os.path.exists(WHITE_PATH):
    os.makedirs(os.path.dirname(WHITE_PATH), exist_ok=True)
    white_img = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(WHITE_PATH, white_img)

# Hand detectors
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Enchant dictionary
try:
    ddd = enchant.Dict("en_US")
except enchant.errors.DictNotFoundError:
    dicts = enchant.list_dicts()
    if dicts:
        ddd = enchant.Dict(dicts[0][0])
    else:
        ddd = None  # no dictionary at all


class Application:
    def __init__(self):
        # Camera (index 1 from your tests)
        self.vs = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.current_image = None

        # Load model
        model_path = os.path.join(BASE_DIR, "models", "cnn8grps_rad1_model.h5")
        self.model = load_model(model_path)

        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" " for _ in range(10)]

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        # -----------------------------
        # UI SETTINGS (4x4 grid idea)
        # Camera feed: row 0-2 col 0-1
        # ASL guide:   row 0-2 col 2-3
        # Bottom UI:   row 3 col 0-3
        # -----------------------------
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x750")
        self.root.minsize(1200, 700)

        # Configure 4x4 grid
        self.root.grid_columnconfigure(0, weight=2)  # camera
        self.root.grid_columnconfigure(1, weight=2)  # camera
        self.root.grid_columnconfigure(2, weight=2)  # ASL guide
        self.root.grid_columnconfigure(3, weight=2)  # ASL guide

        for r in range(4):
            self.root.grid_rowconfigure(r, weight=2, uniform="row")

        # Title (overlay)
        self.T = tk.Label(self.root)
        self.T.place(x=40, y=8)
        self.T.config(text="Sign Language To Text Conversion",
                      font=("Times New Roman", 26, "bold"))

        # ---- Left: Camera frame (row 0-2, col 0-1)
        cam_frame = tk.Frame(self.root, bd=2, relief="groove")
        cam_frame.grid(row=0, column=0, rowspan=3, columnspan=2, sticky="nsew",
                       padx=12, pady=(55, 12))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.panel = tk.Label(cam_frame)
        self.panel.grid(row=0, column=0, sticky="nsew")

        # ---- Right: ASL guide frame (row 0-2, col 2-3)
        guide_frame = tk.Frame(self.root, bd=2, relief="groove", bg="#111111")
        guide_frame.grid(row=0, column=2, rowspan=3, columnspan=2, sticky="nsew",
                         padx=12, pady=(55, 12))
        guide_frame.grid_rowconfigure(0, weight=1)
        guide_frame.grid_columnconfigure(0, weight=1)

        self.signs_label = tk.Label(guide_frame, bg="#111111")
        self.signs_label.grid(row=0, column=0, sticky="nsew")

        self.signs_pil = None
        self.signs_imgtk = None
        signs_path = os.path.join(BASE_DIR,"assets", "asl.jpg")
        if os.path.exists(signs_path):
            self.signs_pil = Image.open(signs_path).convert("RGBA")

        # ---- Bottom: info / predictions / suggestions (row 3, col 0-3)
        bottom = tk.Frame(self.root, bd=2, relief="groove")
        bottom.grid(row=3, column=0, columnspan=4, sticky="nsew",
                    padx=12, pady=(0, 12))

        # bottom layout: 3 columns (info | suggestions | actions)
        bottom.grid_columnconfigure(0, weight=2)
        bottom.grid_columnconfigure(1, weight=3)
        bottom.grid_columnconfigure(2, weight=1)
        bottom.grid_rowconfigure(0, weight=1)

        info = tk.Frame(bottom)
        info.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        info.grid_columnconfigure(1, weight=1)

        self.T1 = tk.Label(info, text="Character :", font=("Times New Roman", 22, "bold"))
        self.T1.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.panel3 = tk.Label(info, text=" ", font=("Times New Roman", 22, "bold"))
        self.panel3.grid(row=0, column=1, sticky="w", pady=(0, 6))

        self.T3 = tk.Label(info, text="Sentence :", font=("Times New Roman", 22, "bold"))
        self.T3.grid(row=1, column=0, sticky="nw")

        self.panel5 = tk.Label(info, text=" ", font=("Times New Roman", 18),
                               wraplength=520, justify="left")
        self.panel5.grid(row=1, column=1, sticky="w")

        sug = tk.Frame(bottom)
        sug.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        for i in range(4):
            sug.grid_columnconfigure(i, weight=1)

        self.T4 = tk.Label(sug, text="Suggestions :", fg="red",
                           font=("Times New Roman", 22, "bold"))
        self.T4.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        self.b1 = tk.Button(sug)
        self.b1.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        self.b2 = tk.Button(sug)
        self.b2.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        self.b3 = tk.Button(sug)
        self.b3.grid(row=1, column=2, sticky="ew", padx=(0, 8))

        self.b4 = tk.Button(sug)
        self.b4.grid(row=1, column=3, sticky="ew")

        actions = tk.Frame(bottom)
        actions.grid(row=0, column=2, sticky="nsew", padx=12, pady=12)

        self.clear = tk.Button(actions)
        self.clear.pack(fill="x", pady=(30, 0))
        self.clear.config(text="Clear", font=("Times New Roman", 18),
                          wraplength=100, command=self.clear_fun)

        # --- NEW: Separate window for WHITE skeleton display ---
        self.white_win = tk.Toplevel(self.root)
        self.white_win.title("Skeleton (White Background)")
        self.white_win.geometry("420x460")
        self.white_win.minsize(380, 420)

        # Close just this window without closing the app
        def _close_white():
            try:
                self.white_win.withdraw()  # hide window
            except Exception:
                pass

        self.white_win.protocol("WM_DELETE_WINDOW", _close_white)

        self.panel2 = tk.Label(self.white_win)
        self.panel2.pack(fill="both", expand=True, padx=10, pady=10)
        self.panel2_imgtk = None
        # --------------------------------------------------------

        # state
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # Update ASL guide once now and also on window resize
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(300, self._refresh_guide)

        self.video_loop()

    # --------- NEW: crop camera to centered square ----------
    def _center_crop_square(self, img_bgr):
        h, w = img_bgr.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(cx - side // 2, 0)
        y1 = max(cy - side // 2, 0)
        return img_bgr[y1:y1 + side, x1:x1 + side]
    # --------------------------------------------------------

    def _on_resize(self, event=None):
        self._refresh_guide()

    def _refresh_guide(self):
        try:
            if self.signs_pil is None:
                self.signs_label.config(
                    text="ASL Guide not found.\nCheck: assets/signs.png",
                    fg="white",
                    bg="#111111",
                    font=("Times New Roman", 18, "bold"),
                    justify="center"
                )
                return

            w = self.signs_label.winfo_width()
            h = self.signs_label.winfo_height()

            if w < 50 or h < 50:
                self.root.after(100, self._refresh_guide)
                return

            img = self.signs_pil.copy()

            # contain inside panel (prevents cut off)
            img.thumbnail((w - 20, h - 20), Image.LANCZOS)

            self.signs_imgtk = ImageTk.PhotoImage(img)
            self.signs_label.config(image=self.signs_imgtk, text="", bg="#111111")

        except Exception:
            pass

    def _draw_skeleton_on_frame(self, frame_bgr, pts_roi, roi_origin_xy):
        """
        Draw skeleton overlay on the CAMERA frame.
        pts_roi: landmarks in ROI coordinates (handz from cropped image)
        roi_origin_xy: (x1, y1) top-left in original frame where ROI starts
        """
        x1, y1 = roi_origin_xy

        def P(i):
            return (pts_roi[i][0] + x1, pts_roi[i][1] + y1)

        # Fingers
        for t in range(0, 4):
            cv2.line(frame_bgr, P(t), P(t + 1), (0, 255, 0), 3)
        for t in range(5, 8):
            cv2.line(frame_bgr, P(t), P(t + 1), (0, 255, 0), 3)
        for t in range(9, 12):
            cv2.line(frame_bgr, P(t), P(t + 1), (0, 255, 0), 3)
        for t in range(13, 16):
            cv2.line(frame_bgr, P(t), P(t + 1), (0, 255, 0), 3)
        for t in range(17, 20):
            cv2.line(frame_bgr, P(t), P(t + 1), (0, 255, 0), 3)

        # Palm connections
        cv2.line(frame_bgr, P(5), P(9), (0, 255, 0), 3)
        cv2.line(frame_bgr, P(9), P(13), (0, 255, 0), 3)
        cv2.line(frame_bgr, P(13), P(17), (0, 255, 0), 3)
        cv2.line(frame_bgr, P(0), P(5), (0, 255, 0), 3)
        cv2.line(frame_bgr, P(0), P(17), (0, 255, 0), 3)

        # Joints
        for i in range(21):
            cv2.circle(frame_bgr, P(i), 3, (0, 0, 255), -1)

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                print("Camera read failed")
                self.root.after(10, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)  # for drawing overlay

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                y1 = max(y - offset, 0)
                y2 = min(y + h + offset, cv2image_copy.shape[0])
                x1 = max(x - offset, 0)
                x2 = min(x + w + offset, cv2image_copy.shape[1])

                image = cv2image_copy[y1:y2, x1:x2]

                white = cv2.imread(WHITE_PATH)
                handz = hd2.findHands(image, draw=False, flipType=True)
                self.ccc += 1

                if handz:
                    hand2 = handz[0]
                    self.pts = hand2['lmList']

                    os_shift = ((400 - w) // 2) - 15
                    os1_shift = ((400 - h) // 2) - 15

                    # Draw bones on WHITE (model input)
                    for t in range(0, 4):
                        cv2.line(
                            white,
                            (self.pts[t][0] + os_shift, self.pts[t][1] + os1_shift),
                            (self.pts[t + 1][0] + os_shift, self.pts[t + 1][1] + os1_shift),
                            (0, 255, 0), 3
                        )
                    for t in range(5, 8):
                        cv2.line(
                            white,
                            (self.pts[t][0] + os_shift, self.pts[t][1] + os1_shift),
                            (self.pts[t + 1][0] + os_shift, self.pts[t + 1][1] + os1_shift),
                            (0, 255, 0), 3
                        )
                    for t in range(9, 12):
                        cv2.line(
                            white,
                            (self.pts[t][0] + os_shift, self.pts[t][1] + os1_shift),
                            (self.pts[t + 1][0] + os_shift, self.pts[t + 1][1] + os1_shift),
                            (0, 255, 0), 3
                        )
                    for t in range(13, 16):
                        cv2.line(
                            white,
                            (self.pts[t][0] + os_shift, self.pts[t][1] + os1_shift),
                            (self.pts[t + 1][0] + os_shift, self.pts[t + 1][1] + os1_shift),
                            (0, 255, 0), 3
                        )
                    for t in range(17, 20):
                        cv2.line(
                            white,
                            (self.pts[t][0] + os_shift, self.pts[t][1] + os1_shift),
                            (self.pts[t + 1][0] + os_shift, self.pts[t + 1][1] + os1_shift),
                            (0, 255, 0), 3
                        )

                    cv2.line(white, (self.pts[5][0] + os_shift, self.pts[5][1] + os1_shift),
                             (self.pts[9][0] + os_shift, self.pts[9][1] + os1_shift), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[9][0] + os_shift, self.pts[9][1] + os1_shift),
                             (self.pts[13][0] + os_shift, self.pts[13][1] + os1_shift), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[13][0] + os_shift, self.pts[13][1] + os1_shift),
                             (self.pts[17][0] + os_shift, self.pts[17][1] + os1_shift), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_shift, self.pts[0][1] + os1_shift),
                             (self.pts[5][0] + os_shift, self.pts[5][1] + os1_shift), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_shift, self.pts[0][1] + os1_shift),
                             (self.pts[17][0] + os_shift, self.pts[17][1] + os1_shift), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(
                            white,
                            (self.pts[i][0] + os_shift, self.pts[i][1] + os1_shift),
                            2, (0, 0, 255), 1
                        )

                    # Predict from WHITE (unchanged behavior)
                    res = white
                    self.predict(res)

                    # --- NEW: Show WHITE skeleton in the separate window ---
                    try:
                        if hasattr(self, "white_win") and self.white_win.winfo_exists():
                            # Convert BGR -> RGB for Tkinter display
                            white_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                            white_pil = Image.fromarray(white_rgb)

                            # Resize to fit panel2
                            w2 = self.panel2.winfo_width()
                            h2 = self.panel2.winfo_height()
                            if w2 > 50 and h2 > 50:
                                white_pil = white_pil.resize((w2, h2), Image.LANCZOS)

                            self.panel2_imgtk = ImageTk.PhotoImage(white_pil)
                            self.panel2.config(image=self.panel2_imgtk)
                    except Exception:
                        pass
                    # ------------------------------------------------------

                    # Overlay skeleton ON CAMERA (sticks to your hand)
                    self._draw_skeleton_on_frame(cv2image_copy, self.pts, (x1, y1))

                    # Update UI labels/buttons
                    self.panel3.config(text=self.current_symbol, font=("Times New Roman", 22, "bold"))

                    self.b1.config(text=self.word1, font=("Times New Roman", 16),
                                   wraplength=200, command=self.action1)
                    self.b2.config(text=self.word2, font=("Times New Roman", 16),
                                   wraplength=200, command=self.action2)
                    self.b3.config(text=self.word3, font=("Times New Roman", 16),
                                   wraplength=200, command=self.action3)
                    self.b4.config(text=self.word4, font=("Times New Roman", 16),
                                   wraplength=200, command=self.action4)

            # --------- DISPLAY: crop the camera into a square ----------
            square_bgr = self._center_crop_square(cv2image_copy)
            cv2image_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image_rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            # ----------------------------------------------------------

            self.panel5.config(text=self.str, font=("Times New Roman", 18), wraplength=520)

        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    # Suggestion button actions
    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word] + self.word1.upper()

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word] + self.word2.upper()

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word] + self.word3.upper()

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word] + self.word4.upper()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        # Prepare input for model (same as original: 400x400x3)
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')

        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        # -------------------------
        # BEGIN: original giant conditional logic (kept as-is from your provided code)
        # -------------------------

        pl = [ch1, ch2]

        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4

        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
                    (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # subgroup mapping
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'
        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count - 2) % 10] != "next":
                if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                    self.str = self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st + 1:ed]
            self.word = word
            if len(word.strip()) != 0 and ddd is not None:
                suggestions = ddd.suggest(word)
                lenn = len(suggestions)
                self.word4 = suggestions[3] if lenn >= 4 else " "
                self.word3 = suggestions[2] if lenn >= 3 else " "
                self.word2 = suggestions[1] if lenn >= 2 else " "
                self.word1 = suggestions[0] if lenn >= 1 else " "
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "
        else:
            self.word1 = " "
            self.word2 = " "
            self.word3 = " "
            self.word4 = " "

    def destructor(self):
        print("Closing Application...")
        print(self.ten_prev_char)

        try:
            if hasattr(self, "white_win") and self.white_win.winfo_exists():
                self.white_win.destroy()
        except Exception:
            pass

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")
(Application()).root.mainloop()
