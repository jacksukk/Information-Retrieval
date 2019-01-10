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
from keras.models import Sequential, Model , load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input, BatchNormalization, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import LeakyReLU

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

word2vec_model = gensim.models.Word2Vec.load("word2vec_train+test.model")
model = load_model(sys.argv[2])
###########################################   read json file   ################################################
test_dict = []
test_data = open(sys.argv[1]).readlines()
for sentence in test_data : test_dict.append(json.loads(sentence))


###########################################   data processing   ################################################
test_tokens = []
test_span = []
test_nodes_dict = []
#test_nodes = []
test_edges = []

for dic in test_dict : 
	test_tokens.append(dic["tokens"])
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
	test_nodes_dict.append(nodes_dict)
	test_span.append(span_list)
	for edges in dic["edges"] : edges_list.append((nodes_dict[tuple(edges[0])][0] , nodes_dict[tuple(edges[1])][0] , list(edges[2].keys())[0]))
	test_edges.append(edges_list)

temp = pd.Series(["fuck", "value" , "agent" , "condition" , "theme", "theme_mod" , "quant_mod" , 
	"co_quant", "null", "location" , "whole" , "source" , "reference_time" , "quant" , "manner" , "time" , "cause" , "+" , "-" ])
node_label = pd.get_dummies(temp)


sentence_vector = []
for line in test_span:
	temp = []
	for word in line:
		temp.append(word2vec_model[word[0]])
	sentence_vector.append(temp)
sentence_vector = np.array(sentence_vector)
edges_vector = []
sentence_len = len(sentence_vector)
for i in range(sentence_len):
	for pairs in test_edges[i]:
		for k in range(len(test_span[i])):
			if test_span[i][k][0] == pairs[0] : index1 = k
			if test_span[i][k][0] == pairs[1] : index2 = k
		
		vec = []  
		for index in range(len(test_span[i])):
			temp = []
			temp.append(index - index1)
			temp.append(index - index2)
			if test_span[i][index][0] == pairs[0]:
				temp.append(1)
			else : temp.append(0)
			if test_span[i][index][0] == pairs[1]:
				temp.append(1)
			else : temp.append(0)
			temp += list(node_label[test_span[i][index][1]])
			vec.append(temp)
		vec = np.array(vec)
		edges_vector.append(np.hstack((sentence_vector[i] , vec)))

test_x = pad_sequences(edges_vector, maxlen = 50, dtype = 'float64', padding = 'post', truncating = 'post', value = np.zeros(len(edges_vector[0][0])))


y_predict = model.predict(test_x)
y = y_predict.argmax(axis = -1).reshape(test_x.shape[0],1)

temp = []
acc = 0
total = 0
ans = []
for i in test_edges:
	for j in i:
		ans.append(j[2])
for i in range(len(y)):
	if y[i][0] == 0:
		temp.append('anaolgy')
	elif y[i][0] == 1:
		temp.append('eq')
	else:
		temp.append('fact')
	if temp[i][0] == ans[i][0]:
		acc += 1

tpf = 0
fpf = 0
tnf = 0
fnf = 0
tpa = 0
fpa = 0
tna = 0
fna = 0
tpe = 0
fpe = 0
tne = 0
fne = 0
for i in range(len(y)):
	if temp[i][0] == 'f':
		if ans[i][0] == 'f':
			tpf += 1
		else:
			fpf += 1
	else:
		if ans[i][0] != 'f':
			tnf += 1
		else:
			fnf += 1

	if temp[i][0] == 'e':
		if ans[i][0] == 'e':
			tpe += 1
		else:
			fpe += 1
	else:
		if ans[i][0] != 'e':
			tne += 1
		else:
			fne += 1

	if temp[i][0] == 'a':
		if ans[i][0] == 'a':
			tpa += 1
		else:
			fpa += 1
	else:
		if ans[i][0] != 'a':
			tna += 1
		else:
			fna += 1
#print(tpa, tpe, tpf)
af1 = 2*tpa/(2*tpa + fna + fpa)
ff1 = 2*tpf/(2*tpf + fnf + fpf)
ef1 = 2*tpe/(2*tpe + fne + fpe)
print("F1 Score : ",(af1 + ff1 + ef1)/3)
#print(acc/len(y))

