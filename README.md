# SignVision: ASL to Text Translation
AI4ALL Spring '24 Cohort: Group 2 Project 

SignVision aims to bridge communication gaps by translating American Sign Language (ASL) hand signs to text in real time. This project showcases the application of machine learning to foster inclusivity and demonstrates the capabilities developed through the Apply AI program.

## Problem Statement

With the aspiration to make technology more inclusive, SignVision addresses the challenge of non-verbal communication barriers faced by the deaf and hard-of-hearing community. The motivation is to provide an AI tool that facilitates everyday communication, with a conscious effort to counteract biases often found in AI models, particularly those related to skin tone.

## Key Results

- Developed a Convolutional Neural Network (CNN) model capable of translating ASL to text with high accuracy.
- Implemented data augmentation and model adjustment techniques to mitigate bias in data predominantly featuring light skin tones.
- Achieved perfect precision, recall, and F1 scores, indicating a need for reevaluation of data diversity and model generalizability.

## Methodologies

Our approach involved iterative model tuning and validation. We used a combination of convolutional layers for feature extraction and fully connected layers for classification, supplemented by dropout for regularization. Bias mitigation was addressed through data augmentation and layer techniques.

## Data Sources

The project utilized the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle, which is inspired by the classical MNIST dataset but specifically tailored for ASL hand signs. 
For future iterations of the project, we will seek out more diverse datasets that include a broader range of skin tones and hand sizes to ensure the model's effectiveness across different demographics.

## Technologies Used

- Python for overall development
- TensorFlow and Keras for building and training the CNN model
- OpenCV for image processing and augmentation
- NumPy and Pandas for data manipulation
- Matplotlib for data visualization
- GitHub for version control and collaboration

## Authors

This project was completed in collaboration with:
- Amy Vaughn
- Daniel Ung
- Alfredo Medina


