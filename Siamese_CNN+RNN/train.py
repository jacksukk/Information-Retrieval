import pandas as pd
import sys
import json 
import numpy as np
import math as m
import gensim
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input, BatchNormalization, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
   # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
###########################################   read json file   ################################################
train_dict = []
train_data = open(sys.argv[1]).readlines()
for sentence in train_data : train_dict.append(json.loads(sentence))

word2vec_model = gensim.models.Word2Vec.load("word2vec_train+test.model")

###########################################   data processing   ################################################
train_tokens = []
train_span = []
train_nodes_dict = []
#train_nodes = []
train_edges = []

for dic in train_dict : 
	train_tokens.append(dic["tokens"])
	nodes_dict = {}
	node = []
	span_list = []
	edges_list = []
	last_end = 0
	for nodes in dic["nodes"] : 
		start , end = nodes[0][0] , nodes[0][1]
		for i in range(start - last_end) : span_list.append((dic["tokens"][last_end + i] , "fuck"))
		s = dic["tokens"][start]
		for i in range(1 , end - start) : s += (" " + dic["tokens"][start + i])
		nodes_dict[tuple(nodes[0])] = (s , list(nodes[1].keys())[0])		
		span_list.append((s , list(nodes[1].keys())[0]))
		last_end = end
	train_nodes_dict.append(nodes_dict)
	train_span.append(span_list)
	for edges in dic["edges"] : edges_list.append((nodes_dict[tuple(edges[0])][0] , nodes_dict[tuple(edges[1])][0] , list(edges[2].keys())[0]))
	train_edges.append(edges_list)

temp = pd.Series(["fuck", "value" , "agent" , "condition" , "theme", "theme_mod" , "quant_mod" , 
	"co_quant", "null", "location" , "whole" , "source" , "reference_time" , "quant" , "manner" , "time" , "cause" , "+" , "-" ])
node_label = pd.get_dummies(temp)



sentence_vector = []
for line in train_span:
	temp = []
	for word in line:
		temp.append(word2vec_model[word[0]])
	sentence_vector.append(temp)
sentence_vector = np.array(sentence_vector)
edges_vector = []
sentence_len = len(sentence_vector)
for i in range(sentence_len):
	for pairs in train_edges[i]:
		for k in range(len(train_span[i])):
			if train_span[i][k][0] == pairs[0] : index1 = k
			if train_span[i][k][0] == pairs[1] : index2 = k
		
		vec = []  
		for index in range(len(train_span[i])):
			temp = []
			temp.append(index - index1)
			temp.append(index - index2)
			if train_span[i][index][0] == pairs[0]:
				temp.append(1)
			else : temp.append(0)
			if train_span[i][index][0] == pairs[1]:
				temp.append(1)
			else : temp.append(0)

			vec.append(temp)
		vec = np.array(vec)
		edges_vector.append(np.hstack((sentence_vector[i] , vec)))

train_x = pad_sequences(edges_vector, maxlen = 50, dtype = 'float64', padding = 'post', truncating = 'post', value = np.zeros(len(edges_vector[0][0])))
print(train_x[0].shape)

train_y = []
for line in train_edges:
	for p in line:
		train_y.append(p[2])
train_y = pd.Series(train_y)
train_y = np.array(pd.get_dummies(train_y))

###########################################   building model   ################################################
q_input = Input(shape=(50, 260))

inner = Conv1D(16, 3, padding='same', name='conv11', kernel_initializer='he_normal')(q_input)  # (None, 128, 64, 64)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max11')(inner)  # (None,64, 32, 64)

inner = Conv1D(32, 3, padding='same', name='conv22', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max22')(inner)  # (None, 32, 16, 128)

inner = Conv1D(64, 3, padding='same', name='conv33', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
inner = BatchNormalization()(inner)
inner1 = Activation('relu')(inner)
inner = LSTM(256 , return_sequences = True, input_length = 50, input_dim = 260, dropout = 0.5, recurrent_dropout = 0.5, kernel_initializer='he_normal')(inner)
inner = LSTM(256 , return_sequences = False, input_length = 50, input_dim = 260, dropout = 0.5, recurrent_dropout = 0.5, kernel_initializer='he_normal')(inner)
inner = Dense(512, activation='relu')(inner)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Dense(512, activation='relu')(inner)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Dense(output_dim = 3)(inner)
y_pred = Activation('softmax')(inner)
model = Model(q_input, y_pred)
model.summary()

checkpoint = ModelCheckpoint('model.h5', monitor='val_f1', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(optimizer = adam,
            loss = 'categorical_crossentropy',
			metrics = ['accuracy', f1])
model.fit(train_x , train_y , batch_size = 64 , epochs = 500 , validation_split = .2, verbose = 1, callbacks = callbacks_list, shuffle = True)



