import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv('NLPlabeledData.tsv', delimiter='\t', quoting=3)
print(df.head())

def process(review):
    review = BeautifulSoup(review, "html.parser").get_text()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words('english'))
    review = [w for w in review if w not in stop_words]
    return " ".join(review)

train_x_all = []
for r in range(len(df["review"])):
    if (r + 1) % 1000 == 0:
        print("no of reviews:", r + 1)
    train_x_all.append(process(df["review"][r]))

x = train_x_all
y = np.array(df["sentiment"])
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=4)

vectorizer = CountVectorizer(max_features=5000)
train_x = vectorizer.fit_transform(train_x)
train_x = train_x.toarray()
test_x = vectorizer.transform(test_x)
test_x = test_x.toarray()

model = RandomForestClassifier(n_estimators=100)
model.fit(train_x, train_y)

test_predict = model.predict(test_x)
test_predict_prob = model.predict_proba(test_x)[:, 1]
score = roc_auc_score(test_y, test_predict_prob)

print("success percentage:",round(score*100))
