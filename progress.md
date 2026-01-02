# CSC173 Deep Computer Vision Project Progress Report
**Project:** ASL Gesture-to-Text System via Convolutional Neural Networks and Real-Time Hand Tracking Integration  
**Student:** Joshua Radz T. Adlaon, 2022-2534  
**Date:** Dec 24, 2026  
**Repository:** [https://github.com/razeuss/CSC173-DeepCV-Adlaon](https://github.com/razeuss/CSC173-DeepCV-Adlaon) 

## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | Kaggle ASL Alphabet dataset organized for training (`dataset/AtoZ_3.1`) |
| Label Strategy Defined | ✅ Completed | Group-based label mapping (8-group mode) aligned to deployment model |
| CNN Model Implemented | ✅ Completed | CNN architecture implemented and compiled for classification |
| Training Run Completed | ✅ Completed | Training executed with checkpointing to save best model |
| Model Exported | ✅ Completed | Best model saved as `models/cnn8grps_rad1_model.h5` |
| Deployment Pipeline Implemented | ✅ Completed | Webcam → hand tracking → ROI → standardization → CNN inference |
| UI Integration | ✅ Completed | Tkinter UI: live feed, predicted symbol, sentence builder, suggestions |
| Documentation | ✅ Completed | README finalized (paper-style sections + references) |
| Demo Assets | ✅ Completed | Screenshot included + demo GIF placeholder prepared |
| Final Submission Readiness | ✅ Completed | Notebooks are presentation-ready and reproducible |


## 1. Dataset Progress
* **Dataset:** American Sign Language Alphabet Dataset (Kaggle)
* **Scope:** Static ASL alphabet gestures **A–Z** (finger-spelling)
* **Approx. size:** ~87,000 labeled images
* **Storage/expected path:** `dataset/AtoZ_3.1`
* **Class handling strategy:**
  * **Primary:** 26 letters (A–Z)
  * **Deployment-aligned mode:** **8-group mapping** used to reduce confusion between visually similar hand shapes before refinement
* **Input specification (training-aligned):**
  * **Image size:** 400 × 400
  * **Channels:** RGB
  * **Loading:** TensorFlow pipeline (`tf.data`) for efficient batching/prefetch


## 2. Training Progress (`train.ipynb`)
### 2.1 Training Configuration (Completed)
* **Batch size:** 32  
* **Epochs:** 10  
* **Learning rate:** 1e-3  
* **Label mode:** `groups` (8 output classes)  
* **Core goal:** Train a CNN model that is directly usable by the deployment notebook.

### 2.2 Model Architecture (Completed)
* **CNN backbone:** stacked convolution blocks + pooling
* **Regularization:** dropout and/or pooling strategy to improve generalization
* **Output:** softmax probabilities (8-group mode by default)

### 2.3 Training + Best Model Saving (Completed)
* **Checkpoint strategy:** saves the *best* performing model during training
* **Export file:** `models/cnn8grps_rad1_model.h5`
* **Training quality checks:** learning curves generated and inspected

### 2.4 Deployment Compatibility Verification (Completed)
* Reloads the saved `.h5` model to confirm successful export
* Performs an **application-style inference sanity check** (same resizing/reshaping assumptions as `asl.ipynb`)


## 3. Deployment Progress (`asl.ipynb`)
### 3.1 Real-Time Input Pipeline (Completed)
* **Webcam capture:** OpenCV `VideoCapture`
* **Hand tracking:** `cvzone.HandDetector` (MediaPipe-based)
* **ROI extraction:** crops detected hand area with padding offset
* **Input standardization:** uses a **white canvas (`assets/white.jpg`)** to reduce background influence and stabilize model input formatting

### 3.2 Model Integration (Completed)
* Loads the exported model directly from:
  * `models/cnn8grps_rad1_model.h5`
* Runs continuous inference on standardized frames.

### 3.3 UI and Text Translation (Completed)
* **UI framework:** Tkinter
* **UI outputs:**
  * live camera panel
  * predicted character display
  * sentence builder / text output
  * optional word suggestion buttons (dictionary-based)
* **Goal met:** converts ASL finger-spelling predictions into readable text in real time.


## 4. Experiments and Results (Completed)
### 4.1 Offline Training Artifacts
* Training curves (accuracy/loss) produced from `train.ipynb`
* Best-model checkpointing confirmed via reload step

### 4.2 Deployment Validation
* Model loads successfully in `asl.ipynb`
* Real-time pipeline runs end-to-end:
  * camera → detection → standardized input → prediction → UI text output

### 4.3 Recommended Metrics Included in Documentation
The README outlines key evaluation metrics suitable for this project:
* **Offline:** accuracy, precision/recall/F1, top-k accuracy, confusion matrix
* **Real-time:** inference latency (ms), FPS, stability/jitter observations

## 5. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|------|--------|------------|
| Train-to-deploy mismatch (input formatting) | ✅ Fixed | Training config and sanity-check ensure app-style input assumptions match deployment |
| Background clutter affecting predictions | ✅ Fixed | ROI cropping + white-canvas standardization reduces background dependence |
| Visually confusable letters in finger-spelling | ✅ Improved | Group-based classification + landmark-aware refinement strategy in deployment pipeline |
| Real-time usability requirement | ✅ Fixed | End-to-end pipeline integrated into Tkinter UI with clear feedback elements |
| Documentation completeness | ✅ Fixed | README structured like a paper and aligned to notebooks and project scope |

## 6. Final Deliverables (All Completed)
- `train.ipynb` — dataset pipeline, CNN training, best-model export
- `asl.ipynb` — webcam pipeline, hand tracking, inference, Tkinter UI gesture-to-text
- `README.md` — full paper-style documentation (abstract → references)
- `assets/` — screenshot + demo GIF placeholder (for submission/presentation)
- `progress.md` — this progress report (**100% complete**)

## 7. Optional Future Enhancements (Not required for completion)
*(Project is already complete; these are optional improvements if extended.)*
- Add temporal smoothing (majority voting) to reduce prediction jitter further
- Expand beyond A–Z finger-spelling to dynamic signs (sequence modeling)
- Add full quantitative confusion-matrix reporting directly into the training notebook
