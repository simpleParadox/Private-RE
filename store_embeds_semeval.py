import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#import torch.nn.functional as F
#import torch.optim as optim

# Import the transformers library for retrieving the BERT embeddings.
from transformers import BertModel, BertTokenizer


# Import pyvacy for privacy preserving optimizers.
#from pyvacy import optim as private_optim, analysis

# Import scikit-learn packages.
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.utils import shuffle
#from sklearn.metrics import f1_score
#from sklearn.utils import gen_batches


# Import scientific computing python packages.
import numpy as np      

# Additional packages.
from tqdm import tqdm

# Load custom functions.
from functions import load_semeval_data


# Using gpu if available.
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_bert_embeds_from_tokens(bert_model, encoded_inputs):
    all_bert_embeds = []
    bert_model = bert_model.to(device)  # Put the bert_model on the GPU.
    for i in tqdm(range(len(encoded_inputs))):
        encoded_input = encoded_inputs[i]
        encoded_input = encoded_input.to(device)  # Put the encoded input on the GPU.
        # print("encoded input: ", type(encoded_input))
        with torch.no_grad():
            outputs = bert_model(**encoded_input)

            # Getting embeddings from the final BERT layer
            #print(outputs.keys())
            token_embeddings = outputs[0]
            token_embeddings = torch.squeeze(token_embeddings, dim=0).cpu().detach()
        all_bert_embeds.append(token_embeddings) 
        encoded_input.to('cpu')
    all_bert_embeds = [t.numpy() for t in all_bert_embeds]
    return all_bert_embeds

def bert_tokenize(texts, tokenizer):
    all_encoded_inputs = []
    # bert_model = bert_model.to(device)
    
    for i in tqdm(range(len(texts))):
        text = texts[i]
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=50, truncation=True)
        #encoded_input = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        #encoded_input = tokenizer.encode(text)
        all_encoded_inputs.append(encoded_input)
        
    return all_encoded_inputs


"""### Define the BertTokenizer and the BertModel from the transformers library."""

# Define the BertModel and the BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/vocab.txt', model_max_length=50, padding_side='right', local_files_only=True)#, config='/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/tokenizer_config.json')
bert_model = BertModel.from_pretrained('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/', local_files_only=True)#, config='/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/config.json')

"""### Encode the inputs and store them so that we don't have re-encode everytime we run the model."""

# First get the train test splits on the sentences and the labels.


# First get the data here.

train_directory_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/train.txt'
test_directory_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/test.txt'

save_traincsvfile_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/train.tsv'
save_testcsvfile_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/test.tsv'

X_train_texts, y_train_classes, label_mappings = load_semeval_data(train_directory_path, save_traincsvfile_path)


X_test_texts, y_test_classes, map = load_semeval_data(test_directory_path, save_testcsvfile_path)

seeds = [0]   # Change the actual seed value here.
all_train_last_hidden_states = []
all_test_last_hidden_states = []
# NOTE: Since colab is running out of memory, you can process this in batches and then concatenate the results. See if this works. If not, then move to Compute Canada.
for seed in seeds:
    #X_train_texts, X_test_texts, y_train_classes, y_test_classes = train_test_split(sentences, y, random_state=seed, test_size=0.2)

    # slices = gen_batches(len(X_train_texts), 1000)
    # for batch_num, s in enumerate(slices):
        # print("Batch num: ", batch_num)

        # Now do the tokenization and the encoding process.
    train_tokens = bert_tokenize(X_train_texts, bert_tokenizer)

    test_tokens = bert_tokenize(X_test_texts, bert_tokenizer)

    # # Now get the encodings from BERT. NOTE: The get_bert_embeds_from_tokens function only returns the last_hidden_state vector for the input.
    last_hidden_states_train = get_bert_embeds_from_tokens(bert_model, train_tokens)
    del train_tokens
    last_hidden_states_test = get_bert_embeds_from_tokens(bert_model, test_tokens)
    del test_tokens

    # # Store the hidden states
    all_train_last_hidden_states.append(last_hidden_states_train)
    all_test_last_hidden_states.append(last_hidden_states_test)


    np.savez_compressed(f"embeds/semeval/train_embeds_seed_{seed}.npz", all_train_last_hidden_states)
    np.savez_compressed(f"embeds/semeval/test_embeds_seed_{seed}.npz", all_test_last_hidden_states)

    np.savez_compressed(f"embeds/semeval/train_labels_seed_{seed}.npz", y_train_classes)
    np.savez_compressed(f"embeds/semeval/test_labels_seed_{seed}.npz", y_test_classes)
