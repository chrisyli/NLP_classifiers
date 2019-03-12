import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.utils.np_utils as utils

from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

seed = 2019
np.random.seed(seed)

def create_model():
    print("Building model..........")
    filters = 100
    kernel_size = 3
    
    meta_input = Input(shape=(len(df.columns)-3,), name="meta_input")
#    meta_conv = Conv1D(128,
#                     kernel_size,
#                     activation="relu")(meta_input)
#    meta_conv = Dropout(0.5)(meta_conv)
#    meta_conv = GlobalMaxPooling1D()(meta_conv)
    
    nlp_input = Input(shape=(256,), name="nlp_input")
    emb = Embedding(input_dim=5000, output_dim=256, input_length=256)(nlp_input)
    nlp_out = Conv1D(filters,
                     kernel_size,
                     padding="valid",
                     activation="relu",
                     strides=1)(emb)
    nlp_out = GlobalMaxPooling1D()(nlp_out)
#    nlp_out = Dropout(0.5)(nlp_out)
    
    x = keras.layers.concatenate([nlp_out, meta_input])
    x = Dense(250, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(250, activation="relu")(x)
    x = Dropout(0.5)(x)
#    x = Dense(128, activation="relu")(x)
#    x = Dropout(0.5)(x)
#    x = Dense(256, activation="relu")(x)
#    x = Dropout(0.5)(x)
    
    main_output = Dense(5, activation="softmax", name="main_output")(x)
    model = Model(inputs=[nlp_input, meta_input], output=main_output)
    model.compile(optimizer="adam",
                  metrics=["accuracy"],
                  loss={"main_output": "categorical_crossentropy"})
    
    return model


def single_split_fit(df):    
    print("Performing single split fit.................")
    X_train, X_test, y_train, y_test = train_test_split(df, df["outcome_score"],
                                                        stratify=df["outcome_score"], 
                                                        test_size=0.25)
    
    token = Tokenizer(num_words=5000)
    token.fit_on_texts(df["summary_strct"])
    text_train = sequence.pad_sequences(token.texts_to_sequences(X_train["summary_strct"]), maxlen=256)
    text_test = sequence.pad_sequences(token.texts_to_sequences(X_test["summary_strct"]), maxlen=256)
    aux_train = X_train.iloc[:,3:len(df.columns)].values
    aux_test = X_test.iloc[:,3:len(df.columns)].values
    label_train = utils.to_categorical(y_train, num_classes=5)
    label_test = utils.to_categorical(y_test, num_classes=5)
    class_weights = class_weight.compute_class_weight("balanced",
                                                 np.unique(X_train["outcome_score"]),
                                                 X_train["outcome_score"])
    
    model = create_model()
    history = model.fit({"nlp_input": text_train, "meta_input": aux_train}, # Features
          {"main_output": label_train}, # Target vector
          epochs=3, # Number of epochs
#          callbacks=callbacks, # Early stopping
          verbose=1, # Print description after each epoch
#          batch_size=batch_size, # Number of observations per batch
          class_weight=class_weights,
          validation_data = ([text_test, aux_test], label_test)# Data for evaluation
          )
    print("History: ", history)
    
    # Prediction probabilities
    pred_prob = model.predict({"nlp_input": text_test, "meta_input": aux_test})    
    pred_prob_df = pd.DataFrame(data=pred_prob[0:,:])
    
    # Prediction class
    pred_class = pred_prob.argmax(axis=-1)
    pred_class_df = pd.DataFrame(data=pred_class[0:,], columns=["pred_score"])
    
    #Id and labels
    id_label = pd.concat([X_test["ev_id"].reset_index(drop=True),y_test.reset_index(drop=True)], axis=1)
    result = pd.concat([id_label.reset_index(drop=True), pred_class_df.reset_index(drop=True)], axis=1)
    result = pd.concat([result, pred_prob_df.reset_index(drop=True)], axis=1)
    
    return result


def cv_fit(df):       
    print("Performing cross validation fit.................")
    aux_data = df.iloc[:,3:len(df.columns)].values
    token = Tokenizer(num_words=5000)
    token.fit_on_texts(df["summary_strct"])
    text_input_data = sequence.pad_sequences(token.texts_to_sequences(df["summary_strct"]), maxlen=256)

    id_labels = pd.DataFrame()
    pred_probs = pd.DataFrame()
    pred_classes = pd.DataFrame()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(aux_data, df["outcome_score"]) :
        model = create_model()
        text_train, text_test = text_input_data[train_index], text_input_data[test_index]
        aux_train, aux_test = aux_data[train_index], aux_data[test_index]
        label_train, label_test = df["outcome_score"][train_index], df["outcome_score"][test_index]
        label_train = utils.to_categorical(label_train, num_classes=5)
        label_test = utils.to_categorical(label_test, num_classes=5)
        class_weights = class_weight.compute_class_weight("balanced",
                                                     np.unique(df["outcome_score"][train_index]),
                                                     df["outcome_score"][train_index])
    
        history = model.fit(x={"nlp_input": text_train, "meta_input": aux_train},
                          y={"main_output": label_train},
                          validation_data = ([text_test, aux_test], label_test),
                          class_weight=class_weights,
                          epochs=3, 
                          batch_size=64)
        print("History: ", history)
        print("Evaluation Result: ", model.evaluate([text_test, aux_test], label_test))
        
        # Prediction probabilities
        pred_prob = model.predict({"nlp_input": text_test, "meta_input": aux_test})    
        pred_prob_df = pd.DataFrame(data=pred_prob[0:,:])
        pred_probs = pd.concat((pred_probs, pred_prob_df), axis=0)
        
        # Prediction class
        pred_class = pred_prob.argmax(axis=-1)
        pred_class_df = pd.DataFrame(data=pred_class[0:,], columns=["pred_score"])
        pred_classes = pd.concat((pred_classes, pred_class_df), axis=0)
        
        #Id and labels
        id_label = pd.concat((df["ev_id"][test_index], df["outcome_score"][test_index]), axis=1)
        id_labels = pd.concat((id_labels, id_label), axis=0)

    result = pd.concat([id_labels.reset_index(drop=True), pred_classes.reset_index(drop=True)], axis=1)
    result = pd.concat([result, pred_probs.reset_index(drop=True)], axis=1)
    
    return result

def non_split_fit(df):    
    print("Performing non-split fit.................")
    
    token = Tokenizer(num_words=5000)
    token.fit_on_texts(df["summary_strct"])
    text = sequence.pad_sequences(token.texts_to_sequences(df["summary_strct"]), maxlen=256)
    aux = df.iloc[:,3:len(df.columns)].values
    label = utils.to_categorical(df.outcome_score, num_classes=5)
    class_weights = class_weight.compute_class_weight("balanced",
                                                 np.unique(df["outcome_score"]),
                                                 df["outcome_score"])
    
    #model = KerasClassifier(build_fn=create_model)
    model = create_model()
    model.fit({"nlp_input": text, "meta_input": aux}, # Features
          {"main_output": label}, # Target vector
          epochs=3, # Number of epochs
          verbose=1, # Print description after each epoch
          class_weight=class_weights
          )
    print("***************Model Fitted on full data*****************")
    
    return model

df = pd.read_csv("~/Documents/preprocessed_data.csv")
df = df.dropna(axis=1)
model = non_split_fit(df)
model.save("dl_conv_1D.h5")
