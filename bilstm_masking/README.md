# Bilstm with Masking Contextual Embeddings
You may refer to Report.pdf to see the model structure.

## Requirements
All code is written in Python3.6. You may install the required packages by running the following command.
```
pip3 install -r requirements.txt
```

## Download Pre-trained Word Embeddings
```
wget -c https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.200d.txt
```

## Data Transformation
Run transform.py to transform original dataset to an input file of train.py
```
python3 transform.py train.json train_trans.json
python3 transform.py test.json test_trans.json 
```

## Training
Run the following command to train a model
```
python3 train.py train_trans.json glove.6B.200d.txt [modelname]
```


## Testing
Run the following command to test and evaluate a model
```
python3 test.py test_trans.json [modelname]
```
 


