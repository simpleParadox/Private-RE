import pandas as pd
import numpy as np
import csv

import torch
from transformers import BatchEncoding
from tqdm import tqdm

import tokenization
import tensorflow_hub as hub
import sentencepiece
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def tf_bert_tokenize(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    all_tokenized_data = []
    
    for text in texts:
      current_tokenized_data = {}
      text = tokenizer.tokenize(text)
        
      text = text[:max_len-2]
      input_sequence = ["[CLS]"] + text + ["[SEP]"]
      pad_len = max_len-len(input_sequence)
      
      tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
      pad_masks = [1] * len(input_sequence) + [0] * pad_len
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
    m_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    bert_layer = hub.KerasLayer(m_url, trainable=False)
    print("bert layer: ", bert_layer)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    print("vocab file: ", vocab_file)
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    print("tokenizer: ", tokenizer)
    return tokenizer




def get_bert_embeds_from_tokens(bert_model, encoded_inputs):
    all_bert_embeds = []
    batch_bert_embeds = []
    with torch.no_grad():
      for i in range(len(encoded_inputs)):
        encoded_input = encoded_inputs[i]
        print("encoded_input", encoded_input)
        encoded_input = encoded_input.to('cpu')  # Put the encoded input on the GPU.
        outputs = bert_model(**encoded_input, output_hidden_states=True)
        hidden_states = outputs['last_hidden_state']
        print("hidden_states", hidden_states.shape)
        hidden_states_detached = hidden_states.detach()
        hidden_np = hidden_states_detached.numpy()
        # print("hidden np size: ", hidden_np.size())
        del hidden_states_detached
        del encoded_input
        all_bert_embeds.append(hidden_np)
      # print("All bert embeds: ", all_bert_embeds)
    print("All bert embeds: ", len(all_bert_embeds))
    return np.concatenate(all_bert_embeds)

def bert_tokenize(texts, tokenizer):
    all_encoded_inputs = []
    # bert_model = bert_model.to(device)
    for i in range(len(texts)):
        text = texts[i]
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=50, truncation=True)
        all_encoded_inputs.append(encoded_input)
        
    return all_encoded_inputs




# def load_table_data(dataset_size='small'):
#     """
#     dataset_size: 'small'|'large'
#     Return: The preprocessed data and the relation labels.
#     """
#     """## Read in Erin's tabular data and preprocess it."""
#
#     # relations_path = '/content/drive/MyDrive/CMPUT 622 project/data/tabular_data/Input_all_29_relation.tsv'
#     if dataset_size == 'small':
#         relations_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/tabular_data/Input_500_29_relation.tsv'
#     else:
#         relations_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/tabular_data/Input_all_29_relation.tsv'
#
#     train_data = pd.read_csv(relations_path, encoding='utf-8', sep = '\t')
#
#
#     train_data.fillna("", inplace = True)
#
#     # Shuffle data so that there is a higher chance of the train and test data being from the same distribution.
#     train_data = shuffle(train_data, random_state = 1)
#
#     sentences = train_data.iloc[:,:-1].values.tolist()
#
#     sentences = [' '.join(sent).strip() for sent in sentences]
#
#     label = preprocessing.LabelEncoder()
#     y = label.fit_transform(train_data['relation'])
#     label_mappings = integer_mapping = {i: l for i, l in enumerate(label.classes_)}
#
#     return sentences, y, label_mappings

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

    features = data.iloc[:,:-1].values.tolist()
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





def get_reduced_label_mappings():
    reduced_label_mappings = {
        0: 'None',
        1: 'award-nominee',
        2: 'author-works_written',
        3: 'book-genre',
        4: 'company-industry',
        5: 'person-graduate',
        6: 'actor-character',
        7: 'director-film',
        8: 'film-country',
        9: 'film-genre',
        10: 'film-language',
        11: 'film-music',
        12: 'film-production_company',
        13: 'actor-film',
        14: 'producer-film',
        15: 'writer-film',
        16: 'political_party-politician',
        17: 'location-contains',
        18: 'musician-album',
        19: 'musician-origin',
        20: 'person-place_of_death',
        21: 'person-nationality',
        22: 'person-parents',
        23: 'person-place_of_birth',
        24: 'person-profession',
        25: 'person-religion',
        26: 'person-spouse',
        27: 'football_position-player',
        28: 'sports_team-player'
    }
    return reduced_label_mappings



def get_label_probs(probs):
    label_mappings = get_reduced_label_mappings()
    fig, ax = plt.subplots()
    ax.barh(list(label_mappings.values()), probs)

    return fig
