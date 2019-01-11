from train import load_data, position2encoding, text2token
from sys import argv
import tensorflow as tf
from keras.models import load_model
from keras.backend import set_session
import os
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

def evaluate(result, Y):
    """
    Computes accuracy and precision, recall, f1 for each class
    """
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
    testdata = argv[1]
    modelname = argv[2]
    word2idx_path = os.path.join('models', modelname, 'word2idx.pkl')
    model_path = os.path.join('models', modelname, 'model.h5') 
     
    X, Position, _, Y, _ = load_data(testdata)
    print("load word2idx from " + word2idx_path)
    word2idx = pickle.load(open(word2idx_path, "rb"))
    X = text2token(X, word2idx)
    Position = position2encoding(Position)
    print("load model from " + model_path) 
    model = load_model(model_path)

    result = model.predict([X, Position], batch_size = 256, verbose=1)
    print(result.shape, Y.shape)
    evaluate(result, Y)

if __name__ == "__main__":
    main()
