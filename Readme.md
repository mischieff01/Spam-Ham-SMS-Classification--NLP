# SMS Spam-Ham Classification 📩

This repository focuses on classifying SMS messages as either spam or ham (non-spam) using various **machine learning and deep learning models**. The models are trained with different text representation techniques, including **Bag of Words (BoW), TF-IDF, Word2Vec, RNN, and LSTM-RNN**.

## 📜 Table of Contents

- [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Preprocessing](https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/)
- [Models]
  - [Random Forest with BoW](https://github.com/mischieff01/Spam-Ham-SMS-Classification--NLP/blob/main/bow.ipynb)
  - [Random Forest with TF-IDF](https://github.com/mischieff01/Spam-Ham-SMS-Classification--NLP/blob/main/tfidf.ipynb)
  - [Random Forest with Word2Vec](https://github.com/mischieff01/Spam-Ham-SMS-Classification--NLP/blob/main/Word2Vec.ipynb)
  - [Recurrent Neural Network (RNN)](https://github.com/mischieff01/Spam-Ham-SMS-Classification--NLP/blob/main/RNN.ipynb)
  - [Long Short-Term Memory (LSTM-RNN)](https://github.com/mischieff01/Spam-Ham-SMS-Classification--NLP/blob/main/LstmRNN.ipynb)
- [Model Comparison]
- [Conclusion]
- [References]

## 🔍 Introduction

The goal of this project is to build a robust SMS classification model that can accurately distinguish between **spam** and **ham** messages. We experimented with different **machine learning** and **deep learning** models to identify the most effective approach.

## 📊 Dataset

The dataset consists of SMS messages labeled as either **spam** or **ham**. The messages were preprocessed to remove noise and irrelevant information to ensure better model performance.

## 🔄 Preprocessing

The following preprocessing steps were applied:
- **Lowercasing** all text
- **Removing punctuation**
- **Tokenization** (splitting text into words)
- **Stopword removal**
- **Stemming/Lemmatization** (reducing words to their root form)
- **Vectorization** using BoW, TF-IDF, or Word2Vec embeddings

## 🏆 Models and Text Embeddings

### 1️⃣ Random Forest with BoW
- **Text Representation**: Bag of Words (BoW)
- **Algorithm**: Random Forest Classifier
- **Approach**: Converts text into a fixed-size vector where each dimension represents word occurrence.
- **Pros**: Simple and effective for structured text data.
- **Cons**: Ignores word order and meaning.

### 2️⃣ Random Forest with TF-IDF
- **Text Representation**: Term Frequency-Inverse Document Frequency (TF-IDF)
- **Algorithm**: Random Forest Classifier
- **Approach**: Assigns weights to words based on their importance in the corpus.
- **Pros**: Reduces the impact of common words.
- **Cons**: Still lacks context understanding.

### 3️⃣ Random Forest with Word2Vec
- **Text Representation**: Word2Vec embeddings (word vectors)
- **Algorithm**: Random Forest Classifier
- **Approach**: Generates word embeddings that capture semantic relationships between words.
- **Pros**: More meaningful representation of text.
- **Cons**: Requires a large dataset for effective training.

### 4️⃣ Recurrent Neural Network (RNN)
- **Text Representation**: Word embeddings (Word2Vec / GloVe)
- **Algorithm**: Vanilla RNN
- **Approach**: Processes text sequentially, capturing dependencies across words.
- **Pros**: Can model sequential relationships.
- **Cons**: Suffers from **vanishing gradient problem**, making it ineffective for long sequences.

### 5️⃣ Long Short-Term Memory (LSTM-RNN)
- **Text Representation**: Word embeddings (Word2Vec / GloVe)
- **Algorithm**: LSTM-based RNN
- **Approach**: Uses memory cells and gates to retain long-term dependencies.
- **Pros**: Addresses the vanishing gradient problem, making it better for long texts.
- **Cons**: Computationally expensive.

---

## 📈 Model Performance Comparison

| Model                      | Accuracy | Precision | Recall | F1-Score |
|----------------------------|----------|-----------|--------|----------|
| **Random Forest with BoW**     | 98.03%   | 100%      | 86.2%  | 92.6%    |
| **Random Forest with TF-IDF**  | 98.74%   | 99.27%    | 95.10% | 97.14%   |
| **Random Forest with Word2Vec**| 97.36%   | 98.48%    | 90%    | 94.05%   |
| **RNN (Vanilla)**          | 76.30%   | -         | -      | -        |
| **LSTM-RNN**               | 79.10%   | -         | -      | -        |

**Key Observations**:
- **Random Forest with TF-IDF performed best** among traditional machine learning models.
- **RNN and LSTM underperformed**, likely due to insufficient data or lack of hyperparameter tuning.
- **LSTM slightly outperformed Vanilla RNN**, showing its effectiveness in handling long-term dependencies.

---

## 🔥 Conclusion

- **For quick and effective SMS spam detection**, Random Forest with **TF-IDF** is the best choice.
- **For deep learning-based approaches**, LSTM performs better than vanilla RNN, but **requires more training data and tuning**.
- **Future Work**: Improve LSTM performance by using pre-trained embeddings, hyperparameter tuning, and more data.

---

## 📚 References

- [Understanding Word2Vec and its Applications](https://towardsdatascience.com/understanding-word2vec-and-its-applications-6e6e41d9cfa7)
- [RNN vs LSTM](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/)
- [TF-IDF Explanation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

