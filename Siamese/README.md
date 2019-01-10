# Siamese model with CNN encoding
You may refer to Report.pdf to see the model structure.

## Usage
All code is written in Python3.5. 
Run the following command to train a word2vec model that will be used in training part
```
python3 word2vec.py --data_path [path to training data] --model_name [path to save w2v model]
```

## Training
Run the following command to train a Siamese model
```
python3 train.py --data_path [path to training data] --model_name [path to save NN model] --w2v [path to word2vec model]
```


## Testing
Run the following command to test and evaluate a model
```
python3 predict.py --data_path [path to testing data] --model_name [path to NN model] --w2v [path to word2vec model]
