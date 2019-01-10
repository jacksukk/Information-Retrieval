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
def evaluate(result, Y):
	acc = 0
	fp_dividend, fp_divisor = 0, 0
	fr_dividend, fr_divisor = 0, 0
	ap_dividend, ap_divisor = 0, 0
	ar_dividend, ar_divisor = 0, 0
	ep_dividend, ep_divisor = 0, 0
	er_dividend, er_divisor = 0, 0

	for idx, r in enumerate(result):
		pred = np.argmax(r)
		ans = np.argmax(Y[idx])
		if pred == ans:
			acc += 1
		
		# fact precision
		if pred == 0:
			fp_divisor += 1
			if ans == 0:
				fp_dividend += 1
		# fact recall
		if ans == 0:
			fr_divisor += 1
			if pred == 0:
				fr_dividend += 1
		
		# analogy precision
		if pred == 1:
			ap_divisor += 1
			if ans == 1:
				ap_dividend += 1

		# analogy recall
		if ans == 1:
			ar_divisor += 1
			if pred == 1:
				ar_dividend += 1

		# equivalence precision
		if pred == 2:
			ep_divisor += 1
			if ans == 2:
				ep_dividend += 1

		# equivalence recall
		if ans == 2:
			er_divisor += 1
			if pred == 2:
				er_dividend += 1

	fp = fp_dividend / fp_divisor
	fr = fr_dividend / fr_divisor
	ff1 = 2 * fp * fr / (fp + fr)
	ap = ap_dividend / ap_divisor
	ar = ar_dividend / ar_divisor
	af1 = 2 * ap * ar / (ap + ar)
	ep = ep_dividend / ep_divisor
	er = er_dividend / er_divisor
	ef1 = 2 * ep * er / (ep + er)
	print("accuracy is ", acc / len(result))
	print("fact p, r, f1", fp, fr, ff1)
	print("analogy p, r, f1", ap, ar, af1)
	print("equivalence p, r, f1", ep, er, ef1)
	print("average p, r, f1", (fp + ap + ep) / 3, (fr + ar + er) / 3, (ff1 + af1 + ef1) / 3)

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

	start_pad = pad_sequences(start, maxlen=23, dtype='float32', padding='post', value=w2v.wv['PAD'])
	end_pad = pad_sequences(end, maxlen=23, dtype='float32', padding='post', value=w2v.wv['PAD'])
	model = load_model(args.model_name)
	ans = model.predict([start_pad,end_pad])
	edge_pre = []
	wrong = 0
	evaluate(ans, edge_inf)
if __name__ == '__main__':
	main()