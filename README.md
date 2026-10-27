# Spam Email Classifier Using Machine Learning

## 1. Project Overview

The **Spam Email Classifier** is a machine learning project that identifies whether an email message is **Spam** or **Ham (Not Spam)**.

Spam emails may contain unwanted advertisements, suspicious offers, or misleading messages. This project uses Natural Language Processing (NLP) and machine learning techniques to analyze email text and classify it into the appropriate category.

The project includes a terminal-based prediction program and an interactive web application built using Streamlit.

## 2. Project Objectives

- Develop a machine learning model for email spam detection.
- Process and analyze textual email data.
- Convert text into numerical features using TF-IDF vectorization.
- Train and evaluate a classification model.
- Build an interactive interface for testing email messages.
- Understand the practical application of machine learning in text classification.

## 3. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and analysis |
| Scikit-learn | Model training and evaluation |
| TF-IDF | Converts email text into numerical features |
| Support Vector Classifier (SVC) | Classifies emails into Spam or Ham |
| Streamlit | Builds the interactive web application |
| Joblib | Saves and loads the trained model and vectorizer |

## 4. Dataset Description

The project uses a labeled email dataset containing **5,171 records**.

- **Ham emails:** 3,672
- **Spam emails:** 1,499
- **Missing values:** None found during initial exploration
- **Target variable:** Email label (Spam or Ham)

The dataset includes email text and corresponding labels. During preprocessing, the labels are represented numerically, with Ham as `0` and Spam as `1`.

## 5. Project Workflow

1. **Data Collection:** Load the labeled email dataset.
2. **Data Exploration:** Examine the dataset structure, class distribution, missing values, and duplicates.
3. **Text Vectorization:** Apply TF-IDF to represent email text as numerical features.
4. **Train-Test Split:** Divide the dataset into training and testing sets using an 80:20 split.
5. **Model Training:** Train a Support Vector Classifier (SVC).
6. **Model Evaluation:** Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
7. **Model Saving:** Save the trained model and TF-IDF vectorizer for reuse.
8. **Prediction:** Classify new email text using the terminal program or Streamlit application.

## 6. Machine Learning Algorithm

### Support Vector Classifier (SVC)

Support Vector Classifier (SVC) is a supervised machine learning algorithm used for classification. In this project, a linear SVC is trained on TF-IDF features to classify email messages as Spam or Ham.

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical values based on the importance of words within documents and across the dataset.

In this project, the vectorizer uses a maximum of 5,000 features and removes common English stop words.

## 7. Model Performance

The SVC model achieved approximately **98.74% accuracy** on the test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.74% |
| Ham Precision | 99% |
| Ham Recall | 99% |
| Ham F1-score | 99% |
| Spam Precision | 97% |
| Spam Recall | 98% |
| Spam F1-score | 98% |

These results reflect performance on the available test dataset. Actual predictions may vary on new email messages.

## 8. Key Features

- Classifies email text as Spam or Ham.
- Provides terminal-based predictions.
- Offers an interactive Streamlit web interface.
- Displays classification results clearly.
- Reuses saved model and vectorizer files.
- Uses a trained machine learning model rather than manually defined keyword rules.

## 9. Project Structure

```text
Spam Email Classifier/
├── data/
│   └── spam_ham_dataset.csv
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── predict.py
├── train_model.py
├── data_exploration.py
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

### Prerequisites
- Python installed on your system
- Required Python libraries
- Project files downloaded or cloned

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the terminal classifier

```bash
python predict.py
```

Enter an email message when prompted to receive a prediction. Type `exit` to stop the program.

### Step 3: Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the web application.

**Note:** The saved model and vectorizer files must be present in the expected `models/` directory before running the prediction programs.

## 11. Limitations

- The model may misclassify unfamiliar or ambiguous messages.
- Performance depends on the quality and representativeness of the training dataset.
- The classifier analyzes text and does not independently verify links, attachments, or sender identity.
- A high accuracy score does not guarantee perfect spam detection.

## 12. Future Enhancements

- Improve text preprocessing and feature engineering.
- Evaluate additional machine learning algorithms.
- Test the model on a wider variety of email messages.
- Add more detailed prediction and evaluation visualizations.
- Explore deployment options for online access.

## 13. Conclusion

The Spam Email Classifier demonstrates how Natural Language Processing and machine learning can be used to classify email messages automatically. By combining TF-IDF vectorization, Support Vector Classifier (SVC), and Streamlit, the project provides both a trained classification model and a user-friendly