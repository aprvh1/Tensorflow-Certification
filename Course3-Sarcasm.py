import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 25000  #{can be} over-ridden this variable in later code.
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 22500

headlines=[]
labels=[]

with open('/tmp/sarcasm.json') as f:
    items=json.load(f)  #shuffle list
    random.shuffle(items)
    for item in items:
        headlines.append(item['headline'])
        labels.append(int(item['is_sarcastic']))

tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(headlines)
print(len(tokenizer.word_index))

headlines_seq = tokenizer.texts_to_sequences(headlines)

train_headlines = pad_sequences(headlines_seq[:training_size],maxlen=max_length,truncating=trunc_type,padding=padding_type)
train_labels = np.array(labels[:training_size])

testing_headlines = pad_sequences(headlines_seq[training_size:],maxlen=max_length,truncating=trunc_type,padding=padding_type)
testing_labels = np.array(labels[training_size:])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(train_headlines, train_labels,
                    epochs=num_epochs,
                    validation_data=(testing_headlines, testing_labels))

print(history.history)

plt.plot(range(len(history.history["val_loss"])),history.history["val_loss"])
plt.show()