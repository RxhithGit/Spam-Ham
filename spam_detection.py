import nltk  # type: ignore
import re
import numpy as np
import pandas as pd # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import PorterStemmer # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

dataset = pd.read_csv('spam.csv', encoding='latin-1') 
sent = dataset['v2']
label = dataset['v1']

le = LabelEncoder()
label = le.fit_transform(label)

stem = PorterStemmer()
preprocessed_sentences = []

for sen in sent:
    senti = re.sub('[^A-Za-z]', ' ', sen)
    senti = senti.lower()
    words = word_tokenize(senti)
    words = [stem.stem(word) for word in words if word not in stopwords.words('english')]
    preprocessed_sentences.append(' '.join(words))

tfidf = TfidfVectorizer(max_features=6000)
features = tfidf.fit_transform(preprocessed_sentences).toarray()

feature_train, feature_test, label_train, label_test = train_test_split(
    features, label, test_size=0.2, random_state=7
)

model_svm = SVC(kernel='linear')
model_svm.fit(feature_train, label_train)
label_pred_svm = model_svm.predict(feature_test)

print("SVM Accuracy:", accuracy_score(label_test, label_pred_svm))
print("SVM Classification Report:\n", classification_report(label_test, label_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(label_test, label_pred_svm))

def classify_message(message: str) -> str:
    # Preprocess the message
    msg = re.sub('[^A-Za-z]', ' ', message).lower()
    words = [PorterStemmer().stem(word) for word in word_tokenize(msg) if word not in stopwords.words('english')]
    processed_msg = ' '.join(words)
    feature = tfidf.transform([processed_msg]).toarray()
    prediction = model_svm.predict(feature)[0]  # 0 = ham, 1 = spam
    return "Spam" if prediction == 1 else "Ham"
