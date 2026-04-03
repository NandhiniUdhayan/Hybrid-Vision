# Hybrid-Vision
Hybrid Vision–IMU Deep Learning Framework with Graph Convolutional Networks and Attention for Personalized Yoga Posture Identification

Overview
This repository contains the implementation of a hybrid deep learning framework for yoga posture recognition using:

- Graph Convolutional Networks (GCN) for spatial skeletal modeling
- LSTM networks for temporal IMU signal processing
- Attention-based fusion for combining multimodal features

The system integrates vision-based pose estimation with synthetically generated IMU data to achieve robust and personalized posture classification.

Methodology
The proposed architecture consists of three main components:

1. **Vision Stream**
   - Extracts skeletal keypoints from images
   - Models joint relationships using GCN

2. **IMU Stream**
   - Uses synthetic IMU signals derived from joint motion
   - Captures temporal dynamics using LSTM

3. **Fusion Module**
   - Attention-based mechanism to dynamically combine both modalities


## 📂 Project Structure
