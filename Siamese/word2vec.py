import numpy as np
from gensim.models import Word2Vec
import argparse
import json

def read_data(path):
	token = []
	with open(path, 'r', encoding='utf8') as fp:
		for line in fp:
			data = json.loads(line)
			token.append(data['tokens'])
	token.append(['OOV','PAD'])
	token.append(['node_%s' % i for i in range(18)])
	return token
def main():
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='None', type=str)
	parser.add_argument('--model_name', type=str)
	args = parser.parse_args()
	data_path = args.data_path
	data = read_data(data_path)
	# training
	model = Word2Vec(data, min_count=1, size=100, iter=10)
	# save
	model.save(args.model_name)

if __name__ == '__main__':
	main()