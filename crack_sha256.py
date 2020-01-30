"""Code to see if we can decode sha256 hash to passwords

Attributes:
    ALLOW_DIGITS (bool): Whether to allow digits in password construction
    ALLOW_LETTERS (bool): Whether to allow letters
    ALLOW_PUNCTUATION (bool): Whether to allow punctuations
    ALLOW_UPPERCASE (bool): Whether to allow uppercase letters
    LEARNING_RATE (float): learning rate for training model
    MAX_PASSWORD_LEN (int): max length of the password
    MIN_PASSWORD_LEN (int): min length of the password
    NUM_EPOCHS (int): number of epochs to train the model
    unique_words (dict): dictionary to keep track of unique words we have
                         generated as passwords
"""
import hashlib
import random
import string
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers

# for reproducability
random.seed(101)
L = tf.keras.layers

MAX_PASSWORD_LEN = 8
MIN_PASSWORD_LEN = 4
ALLOW_LETTERS = True
ALLOW_UPPERCASE = False
ALLOW_DIGITS = False
ALLOW_PUNCTUATION = False


LEARNING_RATE = 0.001
NUM_EPOCHS = 300


unique_words = {}


def get_character_set():
    """Get all possible characters that can occur in our password

    Returns:
        list of allowed characters
    """
    characters = []
    if ALLOW_LETTERS:

        if ALLOW_UPPERCASE:
            characters.extend(list(string.ascii_letters))
        else:
            characters.extend(list(string.ascii_lowercase))

    if ALLOW_DIGITS:
        characters.extend(list(string.digits))

    if ALLOW_PUNCTUATION:
        characters.extend(list(string.punctuation))
    return characters


def get_random_words(num_words=1000, randomize_length=False):
    """Generates random words(passwords)

    Args:
        num_words (int, optional): number of words we want to generate
        randomize_length (bool, optional): whether we should randomize password
                                           length

    Returns:
        list of random strings
    """
    words = []
    characters = get_character_set()

    while(len(words) < num_words):
        if randomize_length:
            random_len = random.randint(MIN_PASSWORD_LEN, MAX_PASSWORD_LEN)
            word = ''.join(random.choices(characters, k=random_len))
        else:
            word = ''.join(random.choices(characters, k=MAX_PASSWORD_LEN))

        if unique_words.get(word, None) is None:
            words.append(word)
            unique_words[word] = 1
    return words


def binary_sha256(s):
    """convert string to binart sha256 hash

    Args:
        s (str): the string to hash

    Returns:
        list of bits of SHA hash
    """
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    binary_string = bin(int(h, 16))[2:].zfill(256)
    return [int(n) for n in binary_string]


def hash_words(words):
    """Hashes words using SHA256

    Args:
        words (list): word list

    Returns:
        list of binary hashes
    """
    hashes = []

    for word in words:
        hashes.append(binary_sha256(word))
    return np.array(hashes)


def string_vectorizer(strings, characters=get_character_set()):
    """One hot encodes characters for model training

    Args:
        strings (list): strings list
        characters (, optional): list of allowed chars in password

    Returns:
        Numpy one hot encoded vectors
    """
    vectors = []
    for _string in strings:
        vector = [[0 if char != letter else 1 for char in characters]
                  for letter in _string]
        vectors.append(vector)
    return np.array(vectors)


def decode_predictions(character_indices_vector):
    """Decodes model predictions to string for comparision with label

    Args:
        character_indices_vector (numpy array): contains predictions of model
                   Shape (batch_size, MAX_len)

    Returns:
        list of decoded strings
    """
    predictions = []
    characters = get_character_set()
    for character_indexes in character_indices_vector:
        s = ''
        for idx in character_indexes:
            s += characters[idx]
        predictions.append(s)
    return predictions


class SHAToChar(Model):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units

    def build(self, input_shape):

        self.net = L.Dense(self.units,
                           input_shape=input_shape)
        super().build(input_shape=input_shape)

    def call(self, x):
        character_logits = self.net(x)
        character_logits = tf.reshape(
            character_logits, (-1, MAX_PASSWORD_LEN, len(get_character_set())))

        return character_logits

    def loss(self, labels, preds):
        loss_t = tf.nn.softmax_cross_entropy_with_logits(labels, preds, axis=-1)
        return tf.math.reduce_mean(loss_t, axis=[0, 1])

    def predict(self, x):
        character_logits = self.net(x)
        character_logits = tf.reshape(
            character_logits, (-1, MAX_PASSWORD_LEN, len(get_character_set())))
        character_probs = tf.math.softmax(character_logits, axis=-1)
        characters_indices = tf.math.argmax(character_probs, axis=-1)

        return decode_predictions(characters_indices.numpy())


if __name__ == "__main__":
    labels_string = get_random_words(1500)
    labels = string_vectorizer(labels_string)

    data = hash_words(labels_string)

    print(data.shape, labels.shape)

    dense_units = MAX_PASSWORD_LEN * len(get_character_set())

    model = SHAToChar(dense_units)
    optimizer = optimizers.Adam(learning_rate=0.001)

    data_t = tf.convert_to_tensor(data, dtype='float32')
    labels_t = tf.convert_to_tensor(labels, dtype='int32')

    X, y = data_t[:1000], labels_t[:1000]
    X_val, y_val = data_t[1000:], labels_t[1000:]

    print(X.shape, y.shape)
    print('\nTraining .... \n')
    for epoch in range(NUM_EPOCHS):

        with tf.GradientTape() as tape:
            logits = model(X)
            loss_t = model.loss(y, logits)
            grads = tape.gradient(loss_t, model.trainable_weights)
            # update to weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        logits_val = model(X_val)
        val_loss_t = model.loss(y_val, logits_val)

        loss = loss_t.numpy()
        val_loss = val_loss_t.numpy()
        if (epoch + 1) % 15 == 0:
            print(
                f'Epoch {epoch + 1 }: train_loss: {loss:.2f}, val_loss: {val_loss:.2f}')
    print(f'\n Train data Predictions after {NUM_EPOCHS} epochs: ')
    print('Labels        : ', labels_string[0:10])
    print('Predications  : ', model.predict(X[:10]))

    print(f'\n Validation data Predictions after {NUM_EPOCHS} epochs: ')
    print('Labels        : ', labels_string[1000:1010])
    print('Predications  : ', model.predict(X_val[:10]))
