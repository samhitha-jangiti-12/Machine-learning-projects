# Purchase Prediction of Books - Customer Segmentation and Targeted Marketing

This repository contains code and information regarding the implementation of machine learning algorithms to predict customer purchases for a bookstore (referred to as CBC in this documentation). The goal is to optimize targeted marketing strategies to reach the most profitable customers and prospects.

## Objective
The objective of this project is to leverage customer data to predict which customers are likely to contribute to sales, enabling CBC to design and implement targeted marketing campaigns effectively.

## Implementation Details
### Customer Acquisition
- Direct Mailing: Promotional materials and offers are sent directly to existing club members via postal mail.
- Telemarketing: Sales calls are made to existing club members to promote products or solicit feedback.

### Data Collection
- All customer responses are recorded and maintained in a centralized database.
- Critical missing information is requested directly from the customer to ensure comprehensive data collection.

### Machine Learning Models
- Logistic Regression: Implemented to predict customer purchases.
  - Achieved an accuracy score of 92.583% on the provided dataset.
  - Evaluated using ROC curve and confusion matrix, showing good performance.
  - Features selected based on analysis of histogram plots, box plots, heatmap for individual variability, joint variability, and correlation values.

# Document Mining and Information Retrieval with NLP

This project focuses on the importance and applicability of Document Mining and Information Retrieval using Natural Language Processing (NLP), specifically in the context of news articles.

## Objective
The project aims to demonstrate how NLP techniques can be used to interpret human language queries and extract relevant information from news articles.

## Dataset
The dataset consists of news articles with respective contexts, providing a diverse set of textual data for analysis.

## Methodology
- Utilizes counting methods like TF-IDF and word embedding concepts for data preprocessing.
- Implements various NLP algorithms to extract relevant information from news articles.
- Computes article rankings based on different similarity measures.

## Applications
- Crime investigation
- Feedback mechanisms
- Content recommendation systems
- Personalized search engines, etc.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Gensim
- Word2Vec

# Autonomous Security Patrolling Robot using Raspberry Pi

## Introduction
In many regions of the world, women's safety remains a pressing concern, and both women and men are still fearful, especially in isolated areas. To address this issue, we propose the development of an autonomous security patrolling robot utilizing Raspberry Pi. This robot is designed to patrol broad regions, ensuring the safety and security of facilities.

## Features
- **Robotic Truck**: Equipped with cameras and a microphone, the robotic vehicle follows a predetermined path.
- **Sensors**: Utilizes sound and camera sensors to detect potential threats or disturbances.
- **Path Following**: Employs an infrared-based path-following technology for precise navigation.
- **Alert System**: When sound is detected, the robot halts at specific points and then resumes patrolling.
- **Surveillance**: Utilizes HD cameras to monitor the surroundings for potential problems, especially during nighttime patrols.
- **Immediate Reporting**: Sends captured photos of any suspicious activity to an IoT website for immediate review.
- **Alarm Notification**: Provides alarm noises via the IoT website to alert users of potential security breaches.

## Purpose
The primary objective of this project is to develop a fully autonomous security robot that patrols designated areas continuously, enhancing safety and security. By leveraging advanced technology such as Raspberry Pi, cameras, and sensors, the robot can effectively monitor and respond to security threats in real-time.

## Usage
1. Assemble the robotic truck and install necessary components including cameras, microphone, and sensors.
2. Program the Raspberry Pi to follow predetermined paths, detect sound disturbances, and capture images.
3. Deploy the robot to patrol the designated region autonomously.
4. Monitor the IoT website for captured images and receive alarm notifications in case of security breaches.

## Benefits
- Enhances safety and security, particularly in isolated or high-risk areas.
- Provides continuous surveillance and immediate response to security threats.
- Reduces reliance on human security personnel, especially during nighttime patrols.
- Offers peace of mind to both facility owners and occupants by ensuring round-the-clock protection.

## Future Scope
- Integration of additional sensors for detecting various types of threats.
- Implementation of machine learning algorithms for advanced threat detection and prediction.
- Enhancement of the IoT website interface for better user interaction and analysis of captured data.



