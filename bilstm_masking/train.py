import json
from sys import argv
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Input, GRU, LSTM, Dense, Dropout, Bidirectional, concatenate, Lambda, average, multiply, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.backend import set_session
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

ENTITY_DICT = {'quant': 0,
               'condition': 1,
               'manner': 2,
               'co_quant': 3,
               'theme_mod': 4,
               'whole': 5,
               'reference_time': 6,
               'value': 7,
               'time': 8,
               'quant_mod': 9,
               'theme': 10,
               'agent': 11,
               'cause': 12,
               'source': 13,
               'location': 14}

RELATION_DICT = {'fact': 0,
                 'analogy': 1,
                 'equivalence': 2}

WORDDIM = 200
SEQLEN = 60
DROPOUT = 0.5
HIDDEN = 64

def load_data(fname):
    """
    Arguments
        fname: string, path to your train_trans.json
    Returns
        X: A list of sentences
        Position: A list of tuples, each tuple specifies the spans of head node and tail node
        Entity: A list of tuples, each tuple specifies the types of head node and tail node
        Y: A list of one-hot vectors, each specifies a relation type
        sample_weight: The sample weight for avoiding class imbalance problem
    """
    print("start loading data")
    X = []
    Position = []
    Entity = []
    Y = []
    with open(fname) as f:
        data = json.load(f)
    
    for d in data["data"]:
        X.append(d["sentence"].lower())
        Position.append((d["head"]["span"], d["tail"]["span"]))
        Entity.append((ENTITY_DICT[d["head"]["node_type"]], ENTITY_DICT[d["tail"]["node_type"]]))
        Y.append(RELATION_DICT[d["relation"]])
    print("finish loading data")

    sample_weight = compute_sample_weight("balanced", Y)
    Y = to_categorical(Y)
    print("label_sum", np.sum(Y, axis=0))
    print("sample_weight", sample_weight)

    return X, Position, Entity, Y, sample_weight

def load_embedding(embedfile):
    """
    Arguments
        embedfile: string, path to "glove.6B.200d.txt"
    Returns
        embeddict: dictionary, keys are words, values are the corresponding embeddings
        word2idx: dictionary, maps each word to an index
        embedding_matrix: numpy array, the embedding matrix to fill in the embedding layer
    """
    print("start loading embedding")
    embeddict = dict()
    word2idx = dict()
    with open(embedfile, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip().split(" ")
            embeddict[line[0]] = np.array(line[1:])
            word2idx[line[0]] = idx + 2 
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1
    print("len of embedding", len(embeddict["the"]))
    

    # make embedding matrix
    bound = np.sqrt(6.0) / np.sqrt(len(word2idx))
    embedding_matrix = np.zeros((len(word2idx), WORDDIM))
    embedding_matrix[0] = np.random.uniform(-bound, bound, WORDDIM)
    embedding_matrix[1] = np.random.uniform(-bound, bound, WORDDIM)
    for word in word2idx:
        if word != "UNK" and word != "PAD":
            embedding_matrix[word2idx[word]] = embeddict[word]

    print("finish loading embedding")
    return embeddict, word2idx, embedding_matrix

def text2token(X, word2idx, seqlen=SEQLEN):
    """
    Arguments
        X: list, input sentences
        word2idx: dicitonary, maps each word to an index
        seqlen: int, the length to pad to
    Returns
        X: list, output tokens
    """
    print("start text2token")
    for idx, x in enumerate(X):
        tmp = []
        for i, word in enumerate(x.split(" ")):
            if i == seqlen:
                break
            if word not in word2idx:
                tmp.append(1)
            else:
                tmp.append(word2idx[word])
        if len(tmp) < seqlen:
            tmp += ([0] * (seqlen - len(tmp)))
        assert len(tmp) == seqlen
        X[idx] = tmp
    print("finish text2token")
    return X

def position2encoding(Position):
    """
    Arguments
        Position: A list of tuples, each tuple specifies the spans of head node and tail node
    Returns
        Position: A list of 1d numpy array, each array is a position sequence specifying the spans of nodes
    """
    print("start position2encoding")
    for idx, ([head_start, head_end], [tail_start, tail_end]) in enumerate(Position):
        hp = np.zeros(SEQLEN)
        if head_end <= SEQLEN and head_start <= SEQLEN:
            for i in range(head_start, head_end):
                hp[i] = 1.0
        elif head_end > SEQLEN and head_start <= SEQLEN:
            for i in range(head_start, SEQLEN):
                hp[i] = 1.0
        tp = np.zeros(SEQLEN)
        if tail_end <= SEQLEN and tail_start <= SEQLEN:
            for i in range(tail_start, tail_end):
                tp[i] = 1.0
        elif tail_end > SEQLEN and tail_start <= SEQLEN:
            for i in range(tail_start, SEQLEN):
                tp[i] = 1.0

        Position[idx] = hp + tp 
    return Position


def lstm(embedding_matrix):
    """
    Arguments
        embedding_matrix: numpy array, the embedding matrix to fill in the embedding layer
    Returns
        model: a keras model for relation classification
    """
    inputs = Input(shape=(SEQLEN,))
    position = Input(shape=(SEQLEN,))

    text_embedding = Embedding(len(embedding_matrix), WORDDIM, weights=[embedding_matrix], trainable=True)(inputs)
    bilstm_cell0 = Bidirectional(LSTM(HIDDEN, return_sequences=True, dropout=DROPOUT))
    context_layer = bilstm_cell0(text_embedding)
   
    position_embedding = Embedding(2, HIDDEN * 2, weights=[np.array([np.zeros(HIDDEN * 2), np.ones(HIDDEN * 2)])], trainable=False)(position)

    filtered_layer = multiply([context_layer, position_embedding])
    
    bilstm_cell1 = Bidirectional(LSTM(HIDDEN // 2, return_sequences=False, dropout=DROPOUT))
    outputs = bilstm_cell1(filtered_layer)
    outputs = BatchNormalization()(outputs)
    outputs = Dense(HIDDEN // 2, activation='relu')(outputs)
    outputs = Dropout(DROPOUT)(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dense(HIDDEN // 4, activation='relu')(outputs)
    outputs = Dropout(DROPOUT)(outputs)
    outputs = Dense(3, activation='softmax')(outputs)
    model = Model(inputs=[inputs, position], outputs=outputs)
    model.summary()
    adam = Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',])
    return model
    
if  __name__ == "__main__":
    traindata = argv[1]
    embedfile = argv[2]
    modelname = argv[3]
    embeddict, word2idx, embedding_matrix = load_embedding(embedfile)
    X, Position, Entity, Y, sample_weight = load_data(traindata)
    X = text2token(X, word2idx)
    Position = position2encoding(Position)
    model = lstm(embedding_matrix)

    earlystopping = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='max')

    if not os.path.isdir(os.path.join('models', modelname)):
        os.makedirs(os.path.join('models', modelname))
    pickle.dump(word2idx, open(os.path.join('models', modelname, 'word2idx.pkl'), 'wb'))

    model_path = os.path.join('models', modelname, 'model.h5')
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 save_weights_only=False,
                                 mode='max')
    model.fit([X, Position], Y, epochs=30, batch_size=256, validation_split=0.1, shuffle=True, callbacks=[earlystopping, checkpoint], class_weight=sample_weight)

    
