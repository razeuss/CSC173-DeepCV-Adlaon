# CSC173 Deep Computer Vision Project Proposal  
**Student:** Joshua Radz T. Adlaon, 2022-2534 
**Date:** December 11, 2025

## 1. Project Title  
**Real-Time American Sign Language (ASL) Alphabet Detection for Enhanced Communication**

## 2. Problem Statement  
Millions of Deaf and Hard-of-Hearing individuals rely on American Sign Language (ASL) for primary communication. However, most virtual meeting platforms and digital communication tools lack real-time ASL interpretation support, creating barriers in online learning, workplace collaboration, and telehealth consultations.  

This project aims to address this gap by developing a real-time ASL alphabet detection system using computer vision and deep learning. The model recognizes ASL alphabet gestures through a webcam feed, enabling more inclusive digital communication and serving as a foundation for future sign-to-text or sign-to-speech systems.

## 3. Objectives  
- Develop a deep learning–based real-time ASL alphabet recognition model.  
- Implement a complete training pipeline including data preprocessing, augmentation, training, validation, and evaluation.  
- Achieve high classification accuracy across the ASL alphabet classes.  
- Build a functional, real-time detection interface using OpenCV and HandDetector to simulate usage during virtual calls or live demonstrations.

## 4. Dataset Plan  
- **Source:** American Sign Language Alphabet Dataset (Kaggle, ~87,000 labeled images)  
- **Classes:** 26 alphabet letters (A–Z), excluding dynamic gestures such as J and Z if necessary.  
- **Acquisition:** Public dataset download with supplementary preprocessing (normalization, resizing, ROI extraction).

## 5. Technical Approach  

### Architecture Overview  
A CNN-based model trained on ASL alphabet images is connected to a real-time webcam detection pipeline. The system uses cvzone's HandDetector to identify the hand region, crops the ROI, preprocesses it, and feeds it into the trained classifier for live prediction.

### Model  
- Custom TensorFlow/Keras CNN  
- Optional alternative: Transfer learning with MobileNetV2 for improved accuracy  
- Real-time inference integrated with Python and OpenCV  

### Framework  
- Python  
- TensorFlow / Keras  
- OpenCV  
- cvzone HandTrackingModule  
- NumPy, PIL, enchant for helper utilities  

### Hardware  
- Google Colab GPU for model training  
- Local desktop/laptop CPU for real-time detection  

## 6. Expected Challenges & Mitigations  

### Challenge: Variability in lighting, angle, and background  
**Mitigation:** Data augmentation (brightness adjustment, rotation, blurring), improved ROI cropping, background noise handling.

### Challenge: Imbalanced dataset across certain ASL letters  
**Mitigation:** Oversampling, augmentation, and confusion matrix–guided fine-tuning.

### Challenge: Real-time performance limitations  
**Mitigation:** Use lightweight CNN architectures, optimize frame processing, ensure ROI extraction is efficient.

### Challenge: Similar-looking hand gestures  
**Mitigation:** Add more class-specific samples, refine model decision boundaries through additional training and tuning.

---

## Conclusion  
This project develops a functional real-time ASL alphabet detection system aimed at improving communication accessibility in virtual meetings and digital interactions. It lays the groundwork for future expansions such as word-level recognition, sign-to-text translation, and integration into assistive communication tools.  
