# Sentiment Analysis: Traditional NLP vs. Transformer Models

## Project Purpose
This project is designed to build and compare two distinct sentiment analysis systems using a dataset of user-written movie reviews. The primary goal is to classify each review into one of two categories: **Positive** or **Negative** (e.g., "This movie was absolutely amazing." -> Positive). 

By implementing both a **Traditional NLP Pipeline** and a **Transformer-Based Pipeline** (using BERT), this project provides a comprehensive comparison of classical machine learning techniques against state-of-the-art deep learning approaches in terms of accuracy, computational speed, and the ability to understand nuanced language and context (such as sarcasm).

## Project Objectives

### Part 1: Data Exploration
- Load the dataset and display basic statistics (number of samples, class distribution).
- Print samples of positive and negative reviews.
- Analyze the average review length to inform tokenization strategies.

### Part 2: Traditional NLP Pipeline
- **Text Preprocessing**: Apply lowercasing, punctuation removal, stop-word removal, and basic tokenization.
- **Feature Extraction**: Convert text into numerical features using TF-IDF Vectorization.
- **Model Training**: Train a classical ML model such as Logistic Regression, Naive Bayes, or Support Vector Machine (SVM).
- **Evaluation**: Assess the model using Accuracy, Precision, Recall, F1-score, and a Confusion matrix.

### Part 3: Transformer-Based Sentiment Analysis
- **Initialization**: Load a pretrained transformer model (e.g., `bert-base-uncased`).
- **Tokenization**: Tokenize reviews using the corresponding transformer tokenizer.
- **Classification**: Run sentiment classification using the Hugging Face `pipeline` or fine-tune the model using the `Trainer` API.
- **Comparison**: Compare the transformer's results against the traditional model across metrics like accuracy, speed, and context understanding.

### Bonus Objectives
1. **Misclassification Analysis**: Identify 5 incorrectly classified reviews and explain potential reasons for the failure.
2. **Attention Exploration**: Visualize attention weights (e.g., using BertViz) to see which words the model focused on.
3. **Multi-Class Sentiment**: Extend the problem to classify reviews as Positive, Neutral, or Negative.

## Project Structure

- `data/` : Directory to store the raw and processed datasets.
- `notebooks/` : Jupyter notebooks containing the step-by-step implementation of data exploration, traditional NLP, and Transformer models.
- `src/` : Reusable Python scripts for data processing, model training, and evaluation.
- `models/` : Directory to save trained classical models and fine-tuned transformer weights.
- `reports/` : Directory to store generated figures, confusion matrices, and attention visualizations.
- `requirements.txt` : List of Python dependencies required to run the project.

## Getting Started

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Place your dataset in the `data/` folder run src/download_data.py.
4. Open the Jupyter notebooks in the `notebooks/` directory to follow the pipeline steps.
