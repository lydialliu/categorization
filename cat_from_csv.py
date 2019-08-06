import os
import random
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
import csv



# writer = tf.summary.FileWriter(["./graphs", [graph]])

def load_cat_dataset(data_path, seed=123):
    """Loads the categories dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.

    # Reference
        https://developers.google.com/machine-learning/guides/text-classification/
    """

    cat_to_label = {}
    label_to_cat = {}
    data = []
    seen = []

    name_col, desc_col, cat_id_col = 1, [2, 3], 5

    with open(data_path) as datafile:
        i = -1 #b/c first row isn't data
        reader = csv.reader(datafile, quoting=csv.QUOTE_ALL)
        for row in reader:  # each row is a list
            data.append(row)
            if (row[cat_id_col] not in seen):
                cat_to_label[row[cat_id_col]] = i
                label_to_cat[i] = row[cat_id_col]
                seen.append(row[cat_id_col])
                i += 1

    data = data[1:]
    random.seed(seed)
    random.shuffle(data)

    train_data = data[10:]
    test_data = data[:10]

    train_texts = []
    train_labels = []
    for row in train_data:
        desc = ""
        for col in desc_col:
            desc += row[col]
        train_texts.append(desc)
        train_labels.append(cat_to_label[row[cat_id_col]])

    test_texts = []
    test_labels = []
    for row in test_data:
        desc = ""
        for col in desc_col:
            desc += row[col]
        test_texts.append(desc)
        test_labels.append(cat_to_label[row[cat_id_col]])

    return (label_to_cat, (train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))


# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """

    op_units = 14
    op_activation = 'softmax'

    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=50,
                      batch_size=6,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):

    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
    """


    """
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    """


    (train_texts, train_labels), (val_texts, val_labels) = data
    num_classes = 14

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(
        train_texts, train_labels, val_texts)

    # Create model instance.
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    log_dir = './log/scalars'

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=[tensorboard_callback],
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size
    )

    history = history.history

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs", sess.graph)
    """
        writer.set_as_default()
        summary = sess.run(merged)
        writer.add_summary(summary)
    """

    predictions = model.predict(x_val)

    # Save model.
    model.save('cat_mlp_model.h5')
    print(history['val_acc'][-1], history['val_loss'][-1])

    return predictions, history

    # results = model.evaluate(val_texts, val_labels)

    # Print results.

def second_largest(lst):
    sorted = lst.argsort()
    return sorted[-2]

def third_largest(lst):
    sorted = lst.argsort()
    return sorted[-3]


if __name__ == '__main__':
    label_to_cat, (train_texts, train_labels), (test_texts, test_labels) = load_cat_dataset('cat/ml_data_part2_copy_view.csv', 125)
    raw_predictions, history = train_ngram_model(((train_texts, train_labels), (test_texts, test_labels)))
    predictions = []
    top_three_predictions = []
    for row in raw_predictions:
        predictions.append(np.argmax(row))
        top_three_predictions.append({1: np.argmax(row), 2: second_largest(row), 3 : third_largest(row)})

    # Accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))



