AI Internship Projects - DecodeLabs

A collection of hands on AI proj completed during my
Industrial Training internship at DecodeLabs (Batch 2026). Each project
builds on the last, moving from foundational rule based logic toward
supervised machine learning with an emphasis on writing clean, reproducible
pipelines rather than just theory.

About the Internship

DecodeLabs AI track is structured as a series of progressive milestones,
each one simulating a real task an AI Engineer would tackle on the job. Every
project in this repo was verified for quality and completed as part of that
qualification process.

Projects


Project 1 — Rule Based AI Chatbot
A deterministic chatbot that responds to user inputs using pure Python 
control flow no ML libraries.

Architecture: continuous while loop → input sanitization (.lower().strip()) 
→ dictionary-based intent matching → response generation → clean exit

Knowledge Base: 7 intents (greeting, farewell, how are you, name query, 
time lookup, help, thanks), each with multiple trigger phrases and 
randomized responses to avoid repetition

Matching: O(1) hash map lookup via Python dict.get() instead of if-elif 
ladder (O(n)) — longest-trigger-first substring fallback for natural phrasing

Handles: case variance, extra whitespace, empty input, unknown queries 
(fallback response), and clean exit via "bye" / "exit" / "quit"




Project 2 — Data Classification Using AI

A supervised learning pipeline that classifies iris flowers into 3 species
using their sepal/petal measurements.


Dataset: Iris dataset (150 samples, 3 classes, 4 features), sourced from
the UCI Machine Learning Repository,
loaded via sklearn.datasets.load_iris
Pipeline: train/test split → feature scaling (StandardScaler) → K
chosen via cross-validated elbow method → K-Nearest Neighbors classifier
Evaluation: confusion matrix, precision/recall/F1 score (not just raw
accuracy)
Result: ~93% accuracy, 0.93 macro F1
Skills: data handling, supervised learning basics, model training,
train/test methodology, output validation



Project 3  — Tech Stack Recommender(Recommendation System)
A content based recommendation engine that maps a user's skills to the most relevant job roles.
Given at least three input skills, the script vectorizes user and role data using TF-IDF, scores relevance with cosine similarity, and returns the Top 3 best matching career paths.
Files
tech_stack_recommender.py : core recommendation logic (ingestion, scoring, sorting, filtering)
raw_skills.csv :  dataset of job roles and their associated skills





 Project 4— Image & Text Recognition (OCR)
A basic optical character recognition (OCR) pipeline built with Python, OpenCV, and Tesseract. It takes a raw image, pre-processes it (grayscale → blur → deskew → adaptive thresholding), extracts text using pytesseract, and filters results through an 80% confidence gate before generating an annotated output image with bounding boxes and confidence scores.
Tech stack: Python · OpenCV · pytesseract (Tesseract OCR) · NumPy · Pillow
Highlights:

Full image pre-processing chain for noisy/skewed scans
Confidence-based filtering (drops low-confidence detections)
Visual output with labeled bounding boxes
Sample input + output included for quick testing
