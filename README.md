# Differentialy Private - Relation Extraction 💪🏼

### This repo contains the code for our project in CMPUT 622 Fall 2022.
Relation Extraction using Differential Privacy
### Datasets:
1. Semeval 2010 Task 8 - https://aclanthology.org/S10-1006/
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

1. Tensorflow >=2.5.0 
NOTE: Install the GPU version if you want to run using the GPU.
Follow this tutorial for details: https://www.tensorflow.org/install/pip

```
pip install tensorflow
```
2. sentencepiece
```
pip install sentencepiece
`````````
3. TensorflowHub
```
pip install --upgrade tensorflow-hub
```
4. PyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
5. Scikit-Learn
```
pip install -U scikit-learn
```
6. Transformers
```
pip install transformers
```
7. MLFlow (for model tracking)
```
pip install mlflow
```
8. Opacus (for differentially private optimizers)
```
pip install opacus
```
9. Torchinfo (to summarize pytorch models)
```
pip install torchinfo
```

10. Other general packages.
```
pip install pandas
pip install matplotlib
pip install tqdm
pip install argparse
```


NOTE: If you want to create a demo for the trained model, you need streamlit. No need to install if demo is not required. This is **optional**. IF you choose to run the demo, then follow the instruction provided in the Streamlit demo section.

### Instructions to run the code.

#### Model Training
Run python project_622.py -h to see the list of arguments.
```
python project_622.py -h
usage: project_622.py [-h] [-p PRIVATE] [-eps EPSILON] [-e EPOCHS] [-s SEED]

Train the private or non-private version of the model. By default, the non-private model is trained.

optional arguments:
  -h, --help            show this help message and exit
  -p PRIVATE, --private PRIVATE
                        Boolean, True or False. Default=False
  -eps EPSILON, --epsilon EPSILON
                        Float in the range [0.5, Infinity]. Default=1.0. If --private is False, this is ignored
  -e EPOCHS, --epochs EPOCHS
                        Integer. Number of epochs. Default=5
  -s SEED, --seed SEED  Integer. Seed for reproducibility. Default=1
```
Example Usage for non-private model:
```
python project_622.py --private=True --epsilon=1.0 --epochs=5 --seed=1
```

To run the code on the Semeval data, just replace the script name with the following.
```
project_622_semeval.py
```
Example usage for on the semeval data:
```
python project_622_semeval.py --private=True --epsilon=1.0 --epochs=100 --seed=1
```


To track the machine learning runs and the outputs, make sure the environment is activated and run 
```
mlflow ui
```
This will open up mlflow and you can track the experiments.

### Streamlit demo
Create a seperate conda environment and activate
Then install the requirements from the requirements.txt file.
NOTE: The packages in the requirements.txt file is only for the streamlit demo. For running the actual code, please follow the package installation steps in the previous section. The packages in the requirements.txt are only for the demo. Using the installed packages from the requirements.txt to run the actual training code may lead to errors.
```
conda create -n streamlit_lib python=3.9
conda activate streamlit_lib
pip install -r requirements.txt
```


### Hyperparameter comparison

| Hyperparameter              |Data: Table / Semeval         |
|:---------------------------:|:----------------------------:|
| LSTM/BiLSTM units           | 1                            |
| Batch Size                  | 16 / 8                       |
| Optimizer                   | Adam                         |
| Max Token Length            | 50                           |
| Learning Rate               | 0.001                        |
| Epochs                      | 5 / 100                      |
 
 
### Results
Results for the Table data.

![Results - Private vs Non-Private results for the Table data.](https://github.com/simpleParadox/Private-RE/blob/main/images/Private%20-%20Accuracy_F1%20vs%20Epsilon%20for%20Table.png?raw=true)


Results for the Semeval data.

![Results - Private vs Non-Private results for the Semeval data.](https://github.com/simpleParadox/Private-RE/blob/main/images/Private%20-%20Accuracy_F1%20vs%20Epsilon%20for%20Semeval.png?raw=true)
