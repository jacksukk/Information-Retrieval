import pandas as pd
import sys
import json 
import numpy as np
import math as m
import gensim
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D , Bidirectional
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model ,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input, BatchNormalization, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import LeakyReLU


###########################################   read json file   ################################################
train_dict = []
train_data = open(sys.argv[1]).readlines()
#test_data = open(sys.argv[2]).readlines()
for sentence in train_data : train_dict.append(json.loads(sentence))
#for sentence in test_data : train_dict.append(json.loads(sentence))
word2vec_model = gensim.models.Word2Vec.load("word2vec_train+test.model")

###########################################   data processing   ################################################
train_tokens = []
train_span = []
train_nodes_dict = []
#train_nodes = []
train_edges = []
#word2vec_train = []
for dic in train_dict : 
	train_tokens.append(dic["tokens"])
	nodes_dict = {}
	node = []
	span_list = []
	edges_list = []
	#word2vec = []
	last_end = 0
	for nodes in dic["nodes"] : 
		start , end = nodes[0][0] , nodes[0][1]
		for i in range(start - last_end) : 
			span_list.append((dic["tokens"][last_end + i] , "fuck"))
			#word2vec.append(dic["tokens"][last_end + i])
		s = dic["tokens"][start]
		for i in range(1 , end - start) : s += (" " + dic["tokens"][start + i])
		nodes_dict[tuple(nodes[0])] = (s , list(nodes[1].keys())[0])		
		span_list.append((s , list(nodes[1].keys())[0]))
		#word2vec.append(s)
		last_end = end
	train_nodes_dict.append(nodes_dict)
	train_span.append(span_list)
	#word2vec_train.append(word2vec)
	for edges in dic["edges"] : edges_list.append((nodes_dict[tuple(edges[0])][0] , nodes_dict[tuple(edges[1])][0] , list(edges[2].keys())[0]))
	train_edges.append(edges_list)

temp = pd.Series(["fuck", "value" , "agent" , "condition" , "theme", "theme_mod" , "quant_mod" , 
	"co_quant", "null", "location" , "whole" , "source" , "reference_time" , "quant" , "manner" , "time" , "cause" , "+" , "-" ])
node_label = pd.get_dummies(temp)

#train_span   , train_nodes , train_edges    

'''word2vec_model = gensim.models.word2vec.Word2Vec(word2vec_train , min_count = 1, size = 256 , iter = 10)
word2vec_model.save("word2vec_train+test.model")'''

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
			temp += list(node_label[train_span[i][index][1]])
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
q_input = Input(shape=(50, 279))
inner = Bidirectional(LSTM(256 , return_sequences = True, input_length = 50, input_dim = 279, dropout = 0.5, recurrent_dropout = 0.5, kernel_initializer='he_normal'))(q_input)
inner = Bidirectional(LSTM(256 , return_sequences = False, input_length = 50, input_dim = 279, dropout = 0.5, recurrent_dropout = 0.5, kernel_initializer='he_normal'))(inner)
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

checkpoint = ModelCheckpoint('model_bir.h5', monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(optimizer = adam,
            loss = 'categorical_crossentropy',
			metrics = ['accuracy'])
model.fit(train_x , train_y , batch_size = 64 , epochs = 100 , validation_split = .2, verbose = 1, callbacks = callbacks_list, shuffle = True)



