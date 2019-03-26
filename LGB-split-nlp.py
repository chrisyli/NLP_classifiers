import time
start_time = time.time()

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.stem.lancaster import LancasterStemmer

df = pd.read_csv("pre_processed_data.csv")
df = df.dropna(axis=1)

stemmer = LancasterStemmer()

df['summary_strct'] = df['summary_strct'].str.split(" ")
df['summary_strct'] = df['summary_strct'].apply(lambda x: [stemmer.stem(y) for y in x])
df['summary_strct'] = df['summary_strct'].apply(lambda x: ' '.join([y for y in x]))


vectorizer=TfidfVectorizer(min_df = .005,
                           max_df = .9,
                           ngram_range=(1,3),
                           strip_accents="unicode"
                           )

x_train, x_test, y_train, y_test = train_test_split(df, df['outcome_score'], stratify=df['outcome_score'], 
                                   test_size=0.3, random_state=2018)
                                    
train_aux = x_train.iloc[:,np.r_[3:len(x_train.columns)]]
test_aux = x_test.iloc[:,np.r_[3:len(x_test.columns)]]

train_tfidf = vectorizer.fit_transform(x_train['summary_strct']).todense()
test_tfidf = vectorizer.transform(x_test['summary_strct']).todense()

train_tfidf = pd.DataFrame(np.array(train_tfidf))
test_tfidf = pd.DataFrame(np.array(test_tfidf))

train_tfidf = train_tfidf.reset_index(drop=True)
test_tfidf = test_tfidf.reset_index(drop=True)

train_aux=train_aux.reset_index(drop=True)
test_aux=test_aux.reset_index(drop=True)
    
x_train = pd.concat([train_aux, train_tfidf], ignore_index=True, axis=1)
x_test = pd.concat([test_aux, test_tfidf], ignore_index=True, axis=1)
    
del [train_tfidf, train_aux, test_tfidf, test_aux]

nb_classes=len(y_train.value_counts())
# train
lgb_model = LGBMClassifier(boosting_type='gbdt', objective = "multiclass",
						   metric = "multi_logloss", learning_rate = 0.01,
                           n_estimators = 1200, num_classes = nb_classes,
                           subsample_freq = 3, min_child_samples = 5,
                           min_child_weight = 0.1, colsample_bytree = 0.8,
                           subsample = 0.7, min_split_gain = 0.05,
                           max_bin = 25, max_depth = -1,
                           num_leaves = 25, random_state = 2018)
    
lgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="multi_logloss", verbose = 200)
    
# predict
y_pred_prob = lgb_model.predict_proba(x_test)
y_pred_class = y_pred_prob.argmax(axis=-1)

print("LGB test confusion matrix:\n", confusion_matrix(y_test, y_pred_class).transpose())
print("LGB test accuracy:", accuracy_score(y_test, y_pred_class))
print("LGB classification report:\n", classification_report(y_test, y_pred_class))

