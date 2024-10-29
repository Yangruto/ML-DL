import pickle
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, GRU, Embedding

CATEGORY_NUM = 2 # True or False
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 128
EPOCH = 10
TOKENIZER_PATH = './event_tokenizer.pkl'

# arg
parser = argparse.ArgumentParser()
parser.add_argument('type', required=True, help='Sequential or Nonsequential')
args = parser.parse_args()

def app_event_tokenize(data, max_sequence_length):
	'''
	Transform categorical events to numerical
	'''
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(data['event'])
	data = data[['event', 'tag']]
	# word-index dictionary
	word_index = tokenizer.word_index
	# convert text to sequences
	sequences = tokenizer.texts_to_sequences(data['event'])
	# one hot encoding for all targets
	target = to_categorical(data['tag'])
	# zero padding
	data = pad_sequences(sequences, maxlen=max_sequence_length, truncating='post', padding='post')
	return data, word_index, target, tokenizer

def split_data(data, target, size):
	x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=size, random_state=100, stratify=target)
	return x_train, x_test, y_train, y_test

def LSTM_model(max_sequence_length, word_kind, category_num):
	model = Sequential()
	model.add(Embedding(input_dim=word_kind + 1, output_dim=100, input_length=max_sequence_length))
	model.add(LSTM(50, return_sequences=True))
	model.add(LSTM(50))
	model.add(Dropout(0.5))
	model.add(Dense(category_num, activation='sigmoid'))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.summary()
	return model

def GRU_model(max_sequence_length, word_kind, category_num):
	model = Sequential()
	model.add(Embedding(input_dim=word_kind + 1, output_dim=100, input_length=max_sequence_length))
	model.add(GRU(50, return_sequences=True))
	model.add(GRU(50))
	model.add(Dropout(0.5))
	model.add(Dense(category_num, activation='sigmoid'))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.summary()
	return model

def Nonsequential_data_prepare(data):
	data = data.reshape((data.shape[0], data.shape[1], 1))
	x_train, x_test, y_train, y_test = split_data(data, target, 0.2)
	x_train1 = x_train[:, :-1]
	x_train1 = np.asarray(x_train_1).astype(np.float32)
	x_train2 = x_train[:, -1:]
	x_train2 = np.array(x_train_2).astype(np.float32)
	x_test1 = x_test[:, :-1]
	x_test1 = np.array(x_test_1).astype(np.float32)
	x_test2 = x_test[:, -1:]
	x_test2 = np.array(x_test_2).astype(np.float32)
	return x_train1, x_train2, x_test1, x_test2, y_train, y_test

def Nonsequential_GRU_model(max_sequence_length, word_kind, category_num):
	input_1 = Input(shape=(max_sequence_length,))
	input_2 = Input(shape=(1,))
	embed_1 = Embedding(input_dim=word_kind + 1, output_dim=100, input_length=max_sequence_length)(input_1)
	gru_1 = GRU(50, return_sequences=True)(embed_1)
	gru_2 = GRU(10)(gru_1)
	merge = concatenate([gru_2, input_2])
	dense_1 = Dense(category_num, activation='sigmoid')(merge)
	model = Model(inputs=[input_1, input_2], outputs=dense_1)
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.summary()
	return model

def precision_recall(target, predict):
    precision, recall, fscore, support = precision_recall_fscore_support(target, predict)
    result = pd.DataFrame([precision, recall, fscore, support])
    result = result.transpose()
    result.columns = ['precision', 'recall', 'fscore', 'support']
    result.index = result.index.astype('str')
    result.index = ['False', 'True']
    print(f'Accuracy: {accuracy_score(target, predict) * 100}')
    return result


if __name__ == "__main__":
	df = pd.read_csv('data.csv')
	df = shuffle(df)
	data, word_index, target, tokenizer = app_event_tokenize(df, MAX_SEQUENCE_LENGTH)
	WORD_KIND = len(word_index)
	# save tokenizer for future usage
	with open(TOKENIZER_PATH, 'wb') as f:
		pickle.dump(tokenizer, f)
	if arg.task == 'Sequential':
		x_train, x_test, y_train, y_test = split_data(data, target, 0.2)
		model = GRU_model(MAX_SEQUENCE_LENGTH, WORD_KIND, CATEGORY_NUM)
		model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_data=(x_test, y_test))
		model.save(f'lstm_event_{MAX_SEQUENCE_LENGTH}.h5')
	else:
		x_train1, x_train2, x_test1, x_test2, y_train, y_test = Nonsequential_data_prepare(data)
		model = Nonsequential_GRU_model(MAX_SEQUENCE_LENGTH, WORD_KIND, CATEGORY_NUM)
		model.fit([x_train1, x_train2], y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_data=([x_test1, x_test2], y_test))
		model.save(f'lstm_event_nonsequential_{MAX_SEQUENCE_LENGTH}.h5')

	# # Predict
	# model = load_model(f'lstm_event_{MAX_SEQUENCE_LENGTH}.h5')
	# tokenizer = joblib.load(TOKENIZER_PATH)
	# predict_input = tokenizer.texts_to_sequences(df['event'])
	# predict_input = pad_sequences(predict_input, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
	# predict_output = np.argmax(model.predict(predict_input), axis=-1)
	# accuracy = precision_recall(df['tag'], predict_output)
	# df['predict'] = predict_output
	# print(accuracy)

	# model = load_model(f'lstm_event_nonsequential_{MAX_SEQUENCE_LENGTH}.h5')
	# tokenizer = joblib.load(TOKENIZER_PATH)
	# predict_input = tokenizer.texts_to_sequences(df['event'])
	# predict_input = pad_sequences(predict_input, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
	# predict_input1 = predict_input[:, :-1]
	# predict_input2 = predict_input[:, -1:]
	# predict_output = np.argmax(model.predict([predict_input1, predict_input2]), axis=-1)
	# accuracy = precision_recall(df['tag'], predict_output)
	# df['predict'] = predict_output
	# print(accuracy)

	# # Retrain
	# df = pd.read_csv('data.csv')
	# df = shuffle(df)
	# model = load_model(f'lstm_event_{MAX_SEQUENCE_LENGTH}.h5')
	# tokenizer = joblib.load(TOKENIZER_PATH)
	# WORD_KIND = len(tokenizer.word_index)
	# data = tokenizer.texts_to_sequences(df['event'])
	# target = to_categorical(df['tag'])
	# data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
	# x_train, x_test, y_train, y_test = split_data(data, target, 0.2)
	# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_data=(x_test, y_test))
	# model.save(f'./lstm_event_{max_sequence_length}_retrain.h5')

	# df = pd.read_csv('data.csv')
	# df = shuffle(df)
	# model = load_model(f'lstm_event_nonsequential_{MAX_SEQUENCE_LENGTH}.h5')
	# tokenizer = joblib.load(TOKENIZER_PATH)
	# WORD_KIND = len(tokenizer.word_index)
	# data = tokenizer.texts_to_sequences(df['event'])
	# target = to_categorical(df['tag'])
	# data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH ,truncating='post', padding='post')
	# x_train1, x_train2, x_test1, x_test2, y_train, y_test = Nonsequential_data_prepare(data)
	# model.fit([x_train1, x_train2], y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_data=([x_test1, x_test2], y_test))
	# model.save(f'lstm_event_nonsequential_{MAX_SEQUENCE_LENGTH}_retrain.h5')