from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.utils import gen_batches
import pandas as pd
import numpy as np
import csv

import torch
from transformers import BatchEncoding
from tqdm import tqdm

import tokenization
import tensorflow_hub as hub
import sentencepiece

device = "cuda:0" if torch.cuda.is_available() else "cpu"

relation_to_id = [
    "other", 
    "Entity-Destination(e1,e2)",
    "Cause-Effect(e2,e1)",        
    "Member-Collection(e2,e1)",      
    "Entity-Origin(e1,e2)",        
    "Message-Topic(e1,e2)",        
    "Component-Whole(e2,e1)",       
    "Component-Whole(e1,e2)",       
    "Instrument-Agency(e2,e1)",     
    "Product-Producer(e2,e1)",     
    "Content-Container(e1,e2)",     
    "Cause-Effect(e1,e2)",          
    "Product-Producer(e1,e2)",       
    "Content-Container(e2,e1)",    
    "Entity-Origin(e2,e1)",          
    "Message-Topic(e2,e1)",        
    "Instrument-Agency(e1,e2)",       
    "Member-Collection(e1,e2)",      
    "Entity-Destination(e2,e1)"]   

def tf_bert_tokenize(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    all_tokenized_data = []
    
    for text in texts:
      current_tokenized_data = {}
      #tokenizer.add_special_tokens(added_special_token)
      text = text[:max_len-2]
      print(text)
      marked_text = "[CLS] " + text + " [SEP]"
      tokenized_text = tokenizer.tokenize(marked_text)
      print(tokenized_text)
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
      print(indexed_tokens)
      pad_len = max_len-len(indexed_tokens)
      
      tokens = tokenizer.convert_tokens_to_ids(indexed_tokens) + [0] * pad_len
      pad_masks = [1] * len(indexed_tokens) + [0] * pad_len
      segment_ids = [0] * max_len
      
      current_tokenized_data['input_ids'] = torch.Tensor([tokens]).long()
      current_tokenized_data['attention_mask'] = torch.Tensor([pad_masks]).long()
      current_tokenized_data['token_type_ids'] = torch.Tensor([segment_ids]).long()
      
      all_tokenized_data.append(BatchEncoding(current_tokenized_data))
      
      # all_tokens.append(tokens)
      # all_masks.append(pad_masks)
      # all_segments.append(segment_ids)
        
        
    # return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    return all_tokenized_data


def tf_tokenizer():
	m_url = "/home/rsaha/scratch/re_656_data/bert_en_uncased_L-12_H-768_A-12_4"
	bert_layer = hub.KerasLayer(m_url, trainable=False)

	vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
	do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
	tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
	return tokenizer




def get_bert_embeds_from_tokens(bert_model, encoded_inputs):
    all_bert_embeds = []
    batch_bert_embeds = []
    with torch.no_grad():
      for i in range(len(encoded_inputs)):
        batch_bert_embeds = []
        encoded_input = encoded_inputs[i]
        encoded_input = encoded_input.to(device)  # Put the encoded input on the GPU.
        outputs = bert_model(**encoded_input, output_hidden_states=True)
        hidden_states = outputs['last_hidden_state']
        hidden_states_detached = hidden_states.cpu().detach()
        hidden_np = hidden_states_detached.numpy()
        # print("hidden np size: ", hidden_np.size())
        del hidden_states_detached
        del encoded_input
        all_bert_embeds.append(hidden_np)
      # print("All bert embeds: ", all_bert_embeds)
    return np.concatenate(all_bert_embeds)

def bert_tokenize(texts, tokenizer, sequence_max_length):
    all_encoded_inputs = []
    # bert_model = bert_model.to(device)
    for i in range(len(texts)):
        text = texts[i]
        encoded_input = tokenizer(
            text, 
            return_tensors='pt', 
            padding="max_length", 
            max_length=sequence_max_length, 
            truncation=True)
        all_encoded_inputs.append(encoded_input)
        
    return all_encoded_inputs




def load_table_data(dataset_size='small'):
    """
    dataset_size: 'small'|'large'
    Return: The preprocessed data and the relation labels.
    """
    """## Read in Erin's tabular data and preprocess it."""

    # relations_path = '/content/drive/MyDrive/CMPUT 622 project/data/tabular_data/Input_all_29_relation.tsv'
    if dataset_size == 'small':
        relations_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/tabular_data/Input_500_29_relation.tsv'
    else:
        relations_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/tabular_data/Input_all_29_relation.tsv'

    train_data = pd.read_csv(relations_path, encoding='utf-8', sep = '\t')


    train_data.fillna("", inplace = True)

    # Shuffle data so that there is a higher chance of the train and test data being from the same distribution.
    train_data = shuffle(train_data, random_state = 1)
    
    sentences = train_data.iloc[:,:-1].values.tolist()

    sentences = [' '.join(sent).strip() for sent in sentences]

    label = preprocessing.LabelEncoder()
    y = label.fit_transform(train_data['relation'])
    label_mappings = integer_mapping = {i: l for i, l in enumerate(label.classes_)}
    
    return sentences, y, label_mappings

# helper functions to read sentence level text

def convertText_csv(path):
	output: List[List[str]] = []

	with open(path) as file:
		lines = file.read()
		lines =  lines.splitlines()

	for line in lines:
		line = line.strip()
		input = line.split(sep="\t")
		entity1 = input[0]
		entity2 = input[1]
		relation = input[2]
		sentence = input[3]

		# sentence = sentence.replace('<e1>', '')
		# sentence = sentence.replace('<e2>', '')
		# sentence = sentence.replace('</e1>', '')
		# sentence = sentence.replace('</e2>', '')
		
		output.append([sentence, entity1, entity2, relation])
	return output

def writeOutput(output, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["sentence", "entity1", "entity2", "relation"])
        for i in output:
            writer.writerow(i)


"""## **Read Sententence-level Data**"""
def load_semeval_data(inputFilename, outputFilePath):

    writeOutput(convertText_csv(inputFilename), outputFilePath)
    data = pd.read_csv(outputFilePath, encoding='utf-8', sep = '\t')

    data = shuffle(data, random_state = 1) 
    # xdata = shuffle(data, random_state = 1) 

    # data = xdata.loc[xdata['relation'] == 'other'][:5]
    # for r in relation_to_id:
    #     data = data.append(xdata.loc[xdata['relation'] == r][:5])

    features = data.iloc[:,:1].values.tolist()
    sentences = [' '.join(i).strip() for i in features]

    label = preprocessing.LabelEncoder()
    y = label.fit_transform(data['relation'])
    label_mappings = integer_mapping = {i: l for i, l in enumerate(label.classes_)}

    return sentences, y, label_mappings
  
  
def reformat(data, batch_size):
    reformated_data = []
    for i in range(batch_size):
        temp_formated_data_dict = {}
        temp_formated_data_dict['input_ids'] = torch.Tensor(data['input_ids'].numpy()[i]).long()
        temp_formated_data_dict['attention_mask'] = torch.Tensor(data['attention_mask'].numpy()[i]).long()
        temp_formated_data_dict['token_type_ids'] = torch.Tensor(data['token_type_ids'].numpy()[i]).long()
        reformated_data.append(BatchEncoding(temp_formated_data_dict))
    return reformated_data

      