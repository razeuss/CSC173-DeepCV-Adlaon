# Real-Time American Sign Language (ASL) Alphabet Detection  
**Initial README.md – Project Plan, Dataset Choice, Architecture Sketch**

## Project Plan
This project focuses on building a **real-time American Sign Language (ASL) alphabet detection system** designed to improve communication accessibility during virtual meetings, online classes, and video calls. The system recognizes ASL alphabet hand gestures (A–Z) from a live webcam feed using a deep learning classifier integrated with a real-time hand detection pipeline.

The primary goals include:
- Training a deep learning model capable of classifying ASL alphabet gestures accurately.
- Implementing a real-time detection application using OpenCV and cvzone HandDetector.
- Demonstrating how such a system can bridge communication gaps for Deaf and Hard-of-Hearing individuals in digital spaces.
- Providing a prototype that may serve as a foundation for future sign-to-text or sign-to-speech systems.

This aligns directly with the project's motivation in the proposal.md: to create an inclusive, accessible solution for real-time ASL interpretation in virtual communication environments.

---

## Dataset Choice
The project uses the **American Sign Language Alphabet Dataset** from Kaggle, consisting of approximately **87,000 labeled images** of static ASL alphabet gestures.

Dataset characteristics:
- **Classes:** 26 letters (A–Z)
- **Size:** ~87k images across all categories
- **Format:** RGB images with consistent gesture framing
- **Reason for choice:**  
  - Large enough for CNN training  
  - Balanced classes across alphabet letters  
  - Suitable for static-gesture recognition tasks  
  - Matches the project’s focus on letter-by-letter ASL prediction  

Dataset preparation steps:
- Resize images to 224×224  
- Normalize pixel values  
- Apply augmentation (rotation, brightness adjustment, flipping)  
- Split dataset into Train/Validation/Test sets (70/15/15)

This matches the dataset description and preprocessing plan outlined in the proposal.md.

---

## Architecture Sketch
The system is composed of two major components:

### **1. Deep Learning Model (Classifier)**
A **TensorFlow/Keras CNN** trained to classify ASL alphabet gestures.

**Model pipeline:**
1. Input: 224×224 RGB image  
2. Convolution → ReLU → Max Pool layers (stacked blocks)  
3. Flatten + Dense layers  
4. Dropout for regularization  
5. Softmax output layer (26 classes: A–Z)  

Alternate model option:
- **MobileNetV2 transfer learning** for improved accuracy and lower inference time.

### **2. Real-Time Detection System**
Integrated using OpenCV + cvzone HandDetector.

**Real-time pipeline flow:**
1. Capture frame from webcam  
2. Detect hand using HandDetector  
3. Crop the Region of Interest (ROI)  
4. Resize and preprocess ROI  
5. Predict ASL letter using the trained model  
6. Display prediction on-screen in real time  

### **Architecture Diagram (Text Sketch)**

