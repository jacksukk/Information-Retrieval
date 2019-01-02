import json
import sys

###########################################   read json file   ################################################
train_dict = []
train_data = open(sys.argv[1]).readlines()
for sentence in train_data : train_dict.append(json.loads(sentence))



###########################################   data processing   ################################################
train_tokens = []
train_span = []
train_nodes_dict = []
train_nodes = []
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
		for i in range(start - last_end) : span_list.append(dic["tokens"][last_end + i])
		s = dic["tokens"][start]
		for i in range(1 , end - start) : s += (" " + dic["tokens"][start + i])
		nodes_dict[tuple(nodes[0])] = (s , list(nodes[1].keys())[0])
		node.append((s , list(nodes[1].keys())[0]))
		span_list.append(s)
		last_end = end
	train_nodes_dict.append(nodes_dict)
	train_span.append(span_list)
	train_nodes.append(node)
	for edges in dic["edges"] : edges_list.append((nodes_dict[tuple(edges[0])][0] , nodes_dict[tuple(edges[1])][0] , list(edges[2].keys())[0]))
	train_edges.append(edges_list)

#train_span   , train_nodes , train_edges    








