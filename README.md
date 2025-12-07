🎬 IMDB Movie Review Sentiment Analysis

Deep Learning • Transformers • TensorFlow • Flask UI • Docker Deployment

✅ Project Overview

This project is a movie review sentiment analysis system that identifies whether a given movie review expresses a Positive or Negative opinion. It is built using a Transformer-based deep learning model (DistilBERT) and is deployed as a web application for real-time predictions.

The application demonstrates how a powerful NLP model can be trained, evaluated, and used in a real-world web-based environment for automatic opinion analysis.

🎯 Key Objective

The main objective of this project is to automate the understanding of customer opinions from textual movie reviews and accurately classify them into positive and negative sentiments. This helps in understanding public feedback at scale without manual effort.

🧠 Technologies Used

Transformer Model – DistilBERT

TensorFlow & Keras

Python

Flask Web Framework

HTML Frontend

Docker for Deployment

⚙️ Core Features

Predicts sentiment from user-entered movie reviews

Displays prediction confidence along with sentiment result

Web-based interface for easy interaction

Supports container-based deployment

Model and tokenizer stored for reuse

Training accuracy and loss are tracked and saved for analysis

📊 Model Performance

Fine-tuned Transformer model (DistilBERT)

Achieves approximately 92% test accuracy

Training history includes loss and accuracy visualization

Model shows stable convergence and strong generalization

🔄 Application Workflow

User enters a movie review in the web interface

The system processes the text using a trained tokenizer

The fine-tuned DistilBERT model predicts the sentiment

The result is displayed on the web interface with a confidence score

📚 Dataset Description

This project uses the IMDB Movie Reviews Dataset, which is a standard benchmark dataset for sentiment analysis tasks.

Dataset Characteristics

The dataset contains 50,000 movie reviews

It is equally divided into:

25,000 training samples

25,000 testing samples

Each review is labeled as:

Positive (1)

Negative (0)

Features in the Dataset

Review Text – The full movie review written by the user

Sentiment Label – Binary value representing positive or negative sentiment

There are no traditional numerical features in this dataset. The entire learning process is based on textual data.

🧹 Data Cleaning & Preprocessing

The raw text data was cleaned to remove:

Special characters

Unnecessary symbols

Extra white spaces

All text was transformed into tokenized format using the DistilBERT tokenizer

Reviews were converted into:

Input IDs

Attention masks

Handling Null Values

The IMDB dataset does not contain missing (null) values in the review text or labels

However, validation checks were performed to confirm data integrity before training

⚠️ Challenges Faced & Solutions
1. Large Text Processing

Challenge: Processing long textual reviews increased memory usage and training time.
Solution: Maximum token length was fixed and padding & truncation techniques were applied.

2. Model Training Time

Challenge: Fine-tuning Transformer models requires high computational resources.
Solution:

Used DistilBERT instead of full BERT (faster and lighter)

Reduced number of epochs

Optimized batch size for stable training

3. Overfitting Risk

Challenge: The model initially showed signs of overfitting.
Solution:

Validation monitoring was used

Training was stopped at optimal convergence

Learning rate tuning was applied

4. Deployment Issues

Challenge: Converting the trained model into a real-time web application was complex.
Solution:

Flask was used to integrate the trained model with a clean HTML interface

Docker was used to create a portable containerized version of the application

🛠️ Practical Applications

This project can be applied in several real-world domains:

Online movie review platforms

Customer feedback analysis systems

Social media sentiment monitoring

Product review classification systems

Brand reputation analysis

Educational NLP demonstration projects

🗂️ Project Structure Summary

Data processing and training pipeline

Model construction and evaluation module

Logging and visualization utilities

Web application for real-time predictions

Deployment configuration using containers

Complete project documentation

🎓 Skills Demonstrated

Natural Language Processing (NLP)

Transformer-based Deep Learning

Model Training and Evaluation

Full-Stack Machine Learning Deployment

Model Optimization and Hyperparameter Tuning

Performance Tracking and Visualization

Software Project Documentation


🚀 Future Scope

Comparison with traditional deep learning models (LSTM, GRU)

Enhanced visual analytics for predictions

Cloud deployment for public access

Support for batch review predictions through file upload

Multi-language sentiment support

👤 Author

Shravan adapa
AI/ML & Data Science Enthusiast
Open to collaborations and research opportunities