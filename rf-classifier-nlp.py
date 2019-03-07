import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

df = pd.read_csv("pre_processed_data.csv")
df = df.dropna(axis=1)

# ID
id_label_pd = df.iloc[:,0:1]
# Auxiliary data (numeric features)
aux_data_pd = df.iloc[:,3:]

#lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df['summary_strct'] = df['summary_strct'].str.split(' ')
df['summary_strct'] = df['summary_strct'].apply(lambda x: [stemmer.stem(y) for y in x])
#df['summary_strct'] = df['summary_strct'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
df['summary_strct'] = df['summary_strct'].apply(lambda x: ' '.join([y for y in x]))

vectorizer=TfidfVectorizer(min_df=2, ngram_range=(1, 4), stop_words='english', 
                           max_features=5000, strip_accents='unicode', norm='l2')

y_label=[]
y_pred=[]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4124)
for train_index, test_index in skf.split(df, df['outcome_score']):
	# Fit and transform: train
    train_tfidf=vectorizer.fit_transform(df.iloc[train_index,]['summary_strct']).todense()
	# Transform: test
    test_tfidf=vectorizer.transform(df.iloc[test_index,]['summary_strct']).todense()
    train_tfidf=pd.DataFrame(data=np.array(train_tfidf))
    test_tfidf=pd.DataFrame(np.array(test_tfidf))
    
    train_aux=aux_data_pd.iloc[train_index,]
    test_aux=aux_data_pd.iloc[test_index,]
    train_aux=train_aux.reset_index(drop=True)
    test_aux=test_aux.reset_index(drop=True)
    
    x_train=pd.concat([train_aux, train_tfidf], ignore_index=True, axis=1)
    x_test=pd.concat([test_aux, test_tfidf], ignore_index=True, axis=1)
    
    y_train=df.iloc[train_index,]['outcome_score']
    y_test=df.iloc[test_index,]['outcome_score']
    rf_model=RandomForestClassifier(n_estimators=1500, random_state=2018)
    rf_model.fit(x_train, y_train)
    
    y_label+=list(y_test)
    y_pred+=list(rf_model.predict(x_test))
    
print("cv confusion matrix:\n", confusion_matrix(y_label, y_pred).transpose())
print("cv accuracy:", accuracy_score(y_label, y_pred))

RF_pred = {"outcome_score": y_label, "RF_pred": y_pred}
RF_pred = pd.DataFrame(data=RF_pred)

filepath = "C:/Users/Desktop/ysli/RF_pred_cv5.csv"
RF_pred.to_csv(filepath)
