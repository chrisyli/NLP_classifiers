import numpy as np
import pandas as pd
import keras

from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import keras.utils.np_utils as utils

from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

def create_model():
	print('Building model..........')
	filters = 250
	kernel_size = 3
	meta_input = Input(shape=(49,), name='meta_input')
	nlp_input = Input(shape=(256,), name='nlp_input')
	emb = Embedding(output_dim=256, input_dim=5000, input_length=256)(nlp_input)
	nlp_out = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(emb)
	nlp_out = GlobalMaxPooling1D()(nlp_out)
	x = keras.layers.concatenate([nlp_out, meta_input])
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.5)(x)
	main_output = Dense(5, activation='softmax', name='main_output')(x)
	model = Model(inputs=[nlp_input, meta_input], output=main_output)
	model.compile(optimizer='rmsprop', metrics=['accuracy'], loss={'main_output': 'categorical_crossentropy'})
	return model

seed = 3
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model,
#            validation_split=0.2,
            verbose=1, batch_size = 64,
			epochs = 4)

df = pd.read_csv("pre_processed_data.csv")
df = df.dropna(axis=1)

#vectorizer = TfidfVectorizer(min_df=2, 
#                             ngram_range=(1, 3), 
#                             stop_words='english', 
#                             max_features= 2500,
#                             strip_accents='unicode', 
#                             norm='l2')
#tf_idf = vectorizer.fit_transform(df['summary_strct']).todense()
aux_data = df.iloc[:,5:58].values
#aux_tf_idf = np.concatenate((aux_data, tf_idf), axis=1)

#cleaned_text = df['summary_raw_strct'].map(lambda x: clean_text(x))
cleaned_text = df['summary_strct']
token = Tokenizer(num_words=5000)
token.fit_on_texts(cleaned_text)
text_input_data = sequence.pad_sequences(token.texts_to_sequences(cleaned_text), maxlen=256)
#x_test = sequence.pad_sequences(seq_test, maxlen=max_len)
#labels = utils.to_categorical(df['outcome_score'], num_classes=5)

id_labels = pd.DataFrame()
pred_probs = pd.DataFrame()
pred_classes = pd.DataFrame(columns = ["pred_class"])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=5781)
for train_index, test_index in folds.split(aux_data, df['outcome_score']):
    model = create_model()
    text_train, text_test = text_input_data[train_index], text_input_data[test_index]
    aux_train, aux_test = aux_data[train_index], aux_data[test_index]
    label_train, label_test = df['outcome_score'][train_index], df['outcome_score'][test_index]
    label_train = utils.to_categorical(label_train, num_classes=5)
    label_test = utils.to_categorical(label_test, num_classes=5)
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(df['outcome_score'][train_index]),
                                                 df['outcome_score'][train_index])

    history = model.fit(x={'nlp_input': text_train, 'meta_input': aux_train},
                      y={'main_output': label_train},
#                      validation_split=0.2, 
                      validation_data = ([text_test, aux_test], label_test),
                      class_weight=class_weights,
                      epochs=3, 
                      batch_size=64)
    print('Evaluation Result: ', model.evaluate([text_test, aux_test], label_test))
    
    # Prediction probabilities
    pred_prob = model.predict({'nlp_input': text_test, 'meta_input': aux_test})    
    pred_prob_df = pd.DataFrame(data=pred_prob)
    pred_probs = pd.concat((pred_probs, pred_prob_df), axis=0)
    
    # Prediction class
    pred_class = pred_prob.argmax(axis=-1)
    pred_class_df = pd.DataFrame(data=pred_class[0:,])
    pred_classes = pd.concat((pred_classes, pred_class_df), axis=0)
    
    #Id and labels
    id_label = pd.concat((df['ev_id'][test_index], df['outcome_score'][test_index]), axis=1)
    id_labels = pd.concat((id_labels, id_label), axis=0)

pred_classes=pred_classes.reset_index(drop=True)
pred_probs=pred_probs.reset_index(drop=True)
id_labels=id_labels.reset_index(drop=True)

cv_final = pd.concat([id_labels, pred_classes], axis=1, ignore_index=True, sort=False)
filepath = "C:/Users/Desktop/ysli/DL_pred_cv5.csv"
cv_final.to_csv(filepath, index=False)
