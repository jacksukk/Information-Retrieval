import json
from sys import argv
import statistics

def transform(ifname, ofname):
    inlist = []
    data = open(ifname).readlines()
    for sample in data:
        inlist.append(json.loads(sample))
    
    outlist = []
    relationset = set()
    entityset = set()
    lenlist = []
    for sample in inlist:
        sentence = ""
        lenlist.append(len(sample["tokens"]))
        for idx, t in enumerate(sample["tokens"]):
            if idx != len(sample["tokens"]) - 1:
                sentence += t
                sentence += " "
            else:
                sentence += t
        # 0: head span, 1: tail span, 2: edge type
        for e in sample["edges"]:
            # print(e)
            d = dict()
            head_span = e[0]
            tail_span = e[1]
            """
            for idx in range(e[0][0], e[0][1]):
                if idx != e[0][1] - 1:
                    head_word += sample["tokens"][idx]
                    head_word += " "
                else:
                    head_word += sample["tokens"][idx]
            for idx in range(e[1][0], e[1][1]):
                if idx != e[1][1] - 1:
                    tail_word += sample["tokens"][idx]
                    tail_word += " "
                else:
                    tail_word += sample["tokens"][idx]
            """ 
            relation = [k for k in e[2].keys()][0]
            
            for n in sample["nodes"]:
                if n[0] == e[0]:
                    head_type = [k for k in n[1].keys()][0]
                    entityset.add(head_type)
                if n[0] == e[1]:
                    tail_type = [k for k in n[1].keys()][0]
                    entityset.add(tail_type)
            d["sentence"] = sentence
            d["head"] = {"span": head_span, "node_type": head_type}
            d["tail"] = {"span": tail_span, "node_type": tail_type}
            d["relation"] = relation
            relationset.add(relation)
            
            # print(d)
            outlist.append(d)

    # print("\n\n\n")
    print("len of outlist", len(outlist))
    print(relationset)
    print(entityset)
    print("median of sentence length", statistics.median(lenlist))
    print("mean of sentence length", statistics.mean(lenlist))
    print("std of sentenc length", statistics.stdev(lenlist))
    with open(ofname, "w") as outfile:
        json.dump({"data": outlist}, outfile)

def main():
    transform(argv[1], argv[2])

if __name__ == "__main__":
    main()
