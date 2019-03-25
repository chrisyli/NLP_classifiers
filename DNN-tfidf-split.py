import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.stem.lancaster import LancasterStemmer

df = pd.read_csv("~/Documents/preprocessed_data.csv")
df = df.dropna(axis=1)

stemmer = LancasterStemmer()

df['summary_strct'] = df['summary_strct'].str.split(" ")
df['summary_strct'] = df['summary_strct'].apply(lambda x: [stemmer.stem(y) for y in x])
df['summary_strct'] = df['summary_strct'].apply(lambda x: ' '.join([y for y in x]))

vectorizer=TfidfVectorizer(min_df = .005,
                           max_df = .9,
                           ngram_range=(1,3),
                           strip_accents="unicode")

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

np.random.seed(2019)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# define training
model = Sequential()
model.add(Dense(1000, input_dim = len(x_train.columns)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')
print(model.summary())

model.fit(x_train, Y_train, validation_data = (x_test, Y_test), epochs=5, \
          batch_size=64, verbose=1)
    
# predict
test_pred_prob = model.predict(x_test)
test_pred_class = test_pred_prob.argmax(axis=-1)

print("DNN test confusion matrix:\n", confusion_matrix(y_test, test_pred_class).transpose())
print("DNN test accuracy:", accuracy_score(y_test, test_pred_class))
print("DNN classification report:\n", classification_report(y_test, test_pred_class))

