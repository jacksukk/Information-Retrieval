import numpy as np
import json
import argparse
from gensim.models import Word2Vec
from keras.layers import Input, LSTM, Dense, Conv1D, Concatenate, Flatten, Bidirectional, LSTM, MaxPooling1D, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import adam, Adam
from keras.preprocessing.sequence import pad_sequences

node_dic = {'value': 0,'agent': 1,'condition': 2,'theme': 3,'theme_mod': 4,'quant_mod':5 ,'co_quant': 6,'null': 7,'location': 8,'whole': 9,'source': 10
,'reference_time': 11,'quant': 12,'manner': 13,'time': 14,'cause': 15,'+': 16,'-': 17}
edge_dic = {'equivalence': 0, 'fact': 1, 'analogy': 2}
def transform_w2v(s, model):
	vec = []
	for w in s:
		# if word in vocabuary the get the vector, else use OOV to represent
		if w in model.wv.vocab:
			vec.append(model.wv[w])
		else:
			vec.append(model.wv['OOV'])
	return np.array(vec)
def transform_node(node, model):
	vec = []
	for n in node:
		s = 'node_%s' % n
		vec.append(model.wv[s])
	return np.array(vec)
def read_data(path):
	token = []
	node = []
	edge = []
	with open(path, 'r') as fp:
		for line in fp:
			data = json.loads(line)
			token.append(data['tokens'])
			node.append(data['nodes'])
			edge.append(data['edges'])
	return token, node, edge

def build_model(length,dim=100):
	# define model architecture
	Conv1 = Conv1D(16,5,strides=1)
	Conv2 = Conv1D(32,5,strides=1)
	Conv3 = Conv1D(64,5,strides=1)
	nor = BatchNormalization()
	dense1 = Dense(64)
	dense2 = Dense(32)
	# question CNN model
	q_input = Input(shape=(length,dim))
	q_1 = Conv1(q_input)
	q_2 = Conv2(q_1)
	q_3 = Conv3(q_2)
	q_4 = nor(q_2)
	# comment CNN model
	a_input = Input(shape=(length,dim))
	a_1 = Conv1(a_input)
	a_2 = Conv2(a_1)
	a_3 = Conv3(a_2)
	a_4 = nor(a_2)
	# concat and DNN model
	concat = Concatenate()([q_4,a_4])
	o = Flatten()(concat)
	o = dense1(o)
	o = dense2(o)
	# output edge
	out = Dense(3, activation='softmax')(o)
	model = Model([q_input,a_input],out)
	model.summary()
	return model

def main():
	# parse argument
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='None', type=str)
	parser.add_argument('--model_name', default='None', type=str)
	parser.add_argument('--w2v', type=str)
	args = parser.parse_args()
	# choose which word2vec to use
	w2v = Word2Vec.load(args.w2v)
	token, node, edge = read_data(args.data_path)
	start, end = [], []
	start_inf, end_inf = [], []
	edge_inf = []
	for i in range(len(token)):
		for j in edge[i]:
			word_start = []
			word_end = []
			start_i, end_i = [], []
			e = np.zeros(3)
			for k in range(j[0][0],j[0][1]):
				word_start.append(token[i][k])
			for k in range(j[1][0],j[1][1]):
				word_end.append(token[i][k])
			for k in node[i]:
				if j[0] in k:
					for inf in k[1:]:
						start_i.append(node_dic[next(iter(inf))])
				if j[1] in k:
					for inf in k[1:]:
						end_i.append(node_dic[next(iter(inf))])
			e[edge_dic[next(iter(j[2]))]] = 1
			edge_inf.append(e)
			start.append(np.concatenate((transform_w2v(word_start, w2v), transform_node(start_i, w2v)), axis=0))
			end.append(np.concatenate((transform_w2v(word_end, w2v), transform_node(end_i, w2v)), axis=0))
	start = np.asarray(start)
	end = np.asarray(end)
	edge_inf = np.asarray(edge_inf)
	print(start.shape)
	print(end.shape)
	print(edge_inf[0])
	pad = 0
	for i in range(len(start)):
		if len(start[i]) > pad:
			pad = len(start[i])
		if len(end[i]) > pad:
			pad = len(end[i])
	start_pad = pad_sequences(start, maxlen=pad, dtype='float32', padding='post', value=w2v.wv['PAD'])
	end_pad = pad_sequences(end, maxlen=pad, dtype='float32', padding='post', value=w2v.wv['PAD'])
	model = build_model(pad)
	adam = Adam(lr=1e-5, decay=1e-7)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit([start_pad, end_pad], edge_inf, batch_size=128, epochs=300, validation_split=0.1)
	model.save(args.model_name)
if __name__ == '__main__':
	main()