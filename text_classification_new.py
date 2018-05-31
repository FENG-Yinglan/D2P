
# coding: utf-8


from __future__ import print_function
import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

TEXT_DATA_DIR = 'C:/Users/inpluslab/Documents/4class_whyper_orin/' 
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2




texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    label_id = len(labels_index)
    labels_index[name] = label_id
    f = open(path,'r')
    data = f.read().split('\n')
    for s in data:
        texts.append(s)
        labels.append(label_id)
    f.close()
print('Found %s texts.' % len(texts))




tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
s1 = sequences[:246]
s2 = sequences[247:529]
s3 = sequences[530:765]
s4 = sequences[766:]
l1 = labels[:246]
l2 = labels[247:529]
l3 = labels[530:765]
l4 = labels[766:]
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))




def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical




data = pad_sequences(s1, maxlen=MAX_SEQUENCE_LENGTH)
label = to_categorical(l1, num_classes= 4 )

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
x_train1 = x_train
for i in range(30):
    x_train1 = np.append(x_train1, x_train, axis=0)
    
y_train = label[:-nb_validation_samples]
y_train1 = y_train
for i in range(30):
    y_train1 = np.append(y_train1, y_train, axis=0)
    
x_val = data[-nb_validation_samples:]
x_val1 = x_val
for i in range(30):
    x_val1 = np.append(x_val1, x_val, axis=0)
    
y_val = label[-nb_validation_samples:]
y_val1 = y_val
for i in range(30):
    y_val1 = np.append(y_val1, y_val, axis=0)
print('Shape of data_train tensor:', x_train1.shape)
print('Shape of label_train tensor:', y_train1.shape)
print('Shape of data_val tensor:', x_val1.shape)
print('Shape of label_val tensor:', y_val1.shape)




data = pad_sequences(s2, maxlen=MAX_SEQUENCE_LENGTH)
label = to_categorical(l2, num_classes= 4 )

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
x_train2 = x_train
for i in range(30):
    x_train2 = np.append(x_train2, x_train, axis=0)
    
y_train = label[:-nb_validation_samples]
y_train2 = y_train
for i in range(30):
    y_train2 = np.append(y_train2, y_train, axis=0)
    
x_val = data[-nb_validation_samples:]
x_val2 = x_val
for i in range(30):
    x_val2 = np.append(x_val2, x_val, axis=0)
    
y_val = label[-nb_validation_samples:]
y_val2 = y_val
for i in range(30):
    y_val2 = np.append(y_val2, y_val, axis=0)
print('Shape of data_train tensor:', x_train2.shape)
print('Shape of label_train tensor:', y_train2.shape)
print('Shape of data_val tensor:', x_val2.shape)
print('Shape of label_val tensor:', y_val2.shape)




data = pad_sequences(s3, maxlen=MAX_SEQUENCE_LENGTH)
label = to_categorical(l3, num_classes= 4 )

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
x_train3 = x_train
for i in range(30):
    x_train3 = np.append(x_train3, x_train, axis=0)
    
y_train = label[:-nb_validation_samples]
y_train3 = y_train
for i in range(30):
    y_train3 = np.append(y_train3, y_train, axis=0)
    
x_val = data[-nb_validation_samples:]
x_val3 = x_val
for i in range(30):
    x_val3 = np.append(x_val3, x_val, axis=0)
    
y_val = label[-nb_validation_samples:]
y_val3 = y_val
for i in range(30):
    y_val3 = np.append(y_val3, y_val, axis=0)
    
print('Shape of data_train tensor:', x_train3.shape)
print('Shape of label_train tensor:', y_train3.shape)
print('Shape of data_val tensor:', x_val3.shape)
print('Shape of label_val tensor:', y_val3.shape)




data = pad_sequences(s4, maxlen=MAX_SEQUENCE_LENGTH)
label = np_utils.to_categorical(l4)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train4 = data[:-nb_validation_samples]
y_train4 = label[:-nb_validation_samples]
x_val4 = data[-nb_validation_samples:]
y_val4 = label[-nb_validation_samples:]

print('Shape of data_train tensor:', x_train4.shape)
print('Shape of label_train tensor:', y_train4.shape)
print('Shape of data_val tensor:', x_val4.shape)
print('Shape of label_val tensor:', y_val4.shape)




a = np.append(x_train1, x_train2, axis=0)
b = np.append(x_train3, x_train4, axis=0)
x_train = np.append(a,b,axis=0)

a = np.append(y_train1, y_train2, axis=0)
b = np.append(y_train3, y_train4, axis=0)
y_train = np.append(a,b,axis=0)

a = np.append(x_val1, x_val2, axis=0)
b = np.append(x_val3, x_val4, axis=0)
x_val = np.append(a,b,axis=0)

a = np.append(y_val1, y_val2, axis=0)
b = np.append(y_val3, y_val4, axis=0)
y_val = np.append(a,b,axis=0)

print('Shape of data tensor:', x_train.shape)
print('Shape of data tensor:', y_train.shape)
print('Shape of data tensor:', x_val.shape)
print('Shape of data tensor:', y_val.shape)
# print('Shape of label tensor:', label.shape)



embeddings_index = {}
f = open('C:/Users/inpluslab/Documents/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))



embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector




from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Training model.')




sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['mae','accuracy',])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=20, batch_size=128)






