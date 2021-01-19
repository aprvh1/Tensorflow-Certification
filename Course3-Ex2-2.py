#Sub-words and tokenizer is made.
import tensorflow as tf
import tensorflow_datasets as tfds

imdb,info = tfds.load("imdb_reviews/subwords8k",with_info=True,as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']
tokenizer=info.features['text'].encoder     #Use this tokenizer to tokenize.

sample_string='Tensorflow from basic to mastery'
tok_string=tokenizer.encode(sample_string)
print(tok_string)

#Model Define Here
embedding_dim=64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),       #In place of Flatten()
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

#Compile
#Fit