# Differentialy Private - Relation Extraction 💪🏼

**Future Updates**: ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) -> containerization for easy distribution of code. 

### This repo contains the code for our project in CMPUT 622 Fall 2022.
Relation Extraction using Differential Privacy
### Datasets:
1. Semeval 2010 Task 8
2. Macdonald and Barbosa 2020 - https://doi.org/10.7939/DVN/SHL1SL


### Model used:
BERT + 1 LSTM unit + Fully Connected


### Package installation.

First create a new conda environment using the following (make sure you have [Anaconda](https://www.anaconda.com/) installed.)
```
conda create --name DP_RE python=3.9
```
```
conda activate DP_RE
```
Now inside the 'DP_RE' conda environment, install the following packages. Follow pip installation guidelines. If using Anaconda package manager, use conda to install packages, but generally pip should work.

Python data science stack.
```
TODO
```
Extra packages. 
1. Tensorflow >=2.5.0
```
pip install tensorflow
```
2. sentencepiece
```
pip install sentencepiece
`````````
3. TensorflowHub
```
pip install "tensorflow>=2.0.0"
pip install --upgrade tensorflow-hub
```

NOTE: If you want to create a demo for the trained model, you need streamlit. No need to install if demo is not required. This is **optional**.
```
pip install streamlit
```


### Instructions to run the code.

#### Training from scratch:
TODO:
1. For the semeval dataset:
2. For the table dataset:

Run python cmput_622.py -h to see the list of arguments.
```
python cmput_622.py -h
usage: cmput_622.py [-h] [-m MODEL] [-d DEMO] [-p PRETRAINED]

Train the proposed model. An example on how to run the script is as follows:
python cmput_622.py --model=comemnet-bilstm

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Can be either 'comemnet-lstm', 'comemnet-bilstm' or
                        'erin'. Example: model=comemnet-bilstm
                        (default=comemnet-bilstm)
  -d DEMO, --demo DEMO  Boolean. Whether to create a gradio demo
                        (default=False).
  -p PRETRAINED, --pretrained PRETRAINED
                        Boolean. Whether to use pretrained model.
                        (default=False). If pretrained=True, please download
                        the google drive files mentioned in the README.
```
Run the code using the following line (will use default arguments).
```
python main.py
```

### Hyperparameter comparison

| Hyperparameter              |Data: Table / Semeval         |
|:---------------------------:|:--------:|:-----------------:|
| LSTM/BiLSTM units           | 1                            |
| Batch Size                  | 16 / 8                       |
| Optimizer                   | Adam                         |
| Max Token Length            | 50                           |
| Learning Rate               | 0.001                        |
| Epochs                      | 5 / 100                      |
 
 
### Results
TODO: show private results for table data
TODO: show private results for semeval data.

