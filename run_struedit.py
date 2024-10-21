import json
from collections import defaultdict
import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM

import argparse
from texttable import Texttable
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd
from datetime import datetime

import time
import re


model_name = 'meta-llama/Llama-2-7b-chat-hf'

class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

class KnowledgeGraph:
    def __init__(self):
        self.kg = defaultdict(lambda: defaultdict(list))
    
    def add_triple(self, e1, triples, triples_label):
        if triples not in self.kg[e1]['triples']:
            self.kg[e1]['triples'].append(triples)
            self.kg[e1]['triples_label'].append(triples_label)
    
    def get_triples(self, e1):
        return self.kg[e1]['triples']
    
    def get_triple_labels(self, e1):
        return self.kg[e1]['triples_label']
    
    def display_kg(self):
        for e1, data in self.kg.items():
            print(f"Entity: {e1}")
            print("Triples:")
            for triple in data['triples']:
                print(f"  {triple}")
            print("Triple Labels:")
            for label in data['triples_label']:
                print(f"  {label}")
            print()
            
def set_stop_words(tokenizer, stop):
    stop_words = stop
    list_stop_word_ids = []
    for stop_word in stop_words:
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria
            
def call_llama(model, tokenizer, prompt, stopping_criteria, stop):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    sequences = model.generate(input_ids.cuda(), stopping_criteria = stopping_criteria)[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(sequences, skip_special_tokens=True)
    for stop_word in stop:
        length_to_remove = len(stop_word)
        if decoded[-length_to_remove:] == stop_word:
            decoded = decoded[:-length_to_remove]
    output_str = decoded.strip()
    return output_str


def reasoning_on_KG(model, tokenizer, stopping_criteria, stop, source_id, relation, new_kg, select_task_prompt):
    
    reasoning_path = []
    path_label = " "
    s = source_id
    for rela in relation:
        candidate_triple = new_kg.kg[s]["triples"]
        candidate_rela = [i[1] for i in candidate_triple]
                    
        if len(candidate_triple) == 0:
            # print("Candidate is NONE")
            return reasoning_path, path_label
        
        if len(candidate_triple) == 1:
            reasoning_path.append(candidate_triple[0])
            triple = candidate_triple[0]
            path_label += map_to_label[triple[0]] + ' -> ' + map_to_label[triple[1]] + ' -> ' + map_to_label[triple[2]] + " "
            s = candidate_triple[0][2]
            continue
        
        # select relation
        select_prompt = select_task_prompt + "Target relation: " + rela + "\nCandidate relations: " +\
        "; ".join(f"{index}: {map_to_label[can_rela]}" for index, can_rela in enumerate(candidate_rela)) + "\nChosen relation id: "
        
        gen = call_llama(model, tokenizer, select_prompt, stopping_criteria, stop)
        match = re.search(r'\d+', gen)
        if match:
            selected_id = int(match.group())
            if selected_id > len(candidate_triple) - 1:
                selected_id = 0
        else:
            selected_id = 0
        
        reasoning_path.append(candidate_triple[selected_id])
        triple = candidate_triple[selected_id]
        path_label += map_to_label[triple[0]] + ' -> ' + map_to_label[triple[1]] + ' -> ' + map_to_label[triple[2]] + " "
        s = candidate_triple[selected_id][2]
        
        
    return reasoning_path, path_label
    
    

with open('StruEdit/MQuAKE/datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
with open('StruEdit/prompts/prompt_extract.txt', 'r') as f:
    extract_task_prompt = f.read()
with open('StruEdit/prompts/prompt_select.txt', 'r') as f:
    select_task_prompt = f.read() 


new_kg = KnowledgeGraph()
map_to_id = {}
map_to_label = {}
results = []
tot = 0
cor = 0

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
model.cuda()
stop1 = ["Q:"]
stop2 = ["Target relation:"]
stopping_criteria1 = set_stop_words(tokenizer, stop1)
stopping_criteria2 = set_stop_words(tokenizer, stop2)


for d in dataset:
    for index, et in enumerate(d['orig']['new_triples']):
        et_label = d['orig']['new_triples_labeled'][index]
        new_kg.add_triple(et[0], et, et_label)
        map_to_id[et_label[0]] = et[0]
        map_to_id[et_label[2]] = et[2]
        map_to_label[et[0]] = et_label[0]
        map_to_label[et[2]] = et_label[2]
        map_to_label[et[1]] = et_label[1]
        

for _id, d in enumerate(tqdm(dataset)):
    tot += 1
    for q in d["questions"]:
        ans = ""
        # print(q)
        # Extract entity and relation
        extract_prompt = extract_task_prompt + 'Q: ' + q + '\n'
        
        gen = call_llama(model, tokenizer, extract_prompt, stopping_criteria1, stop1)
        # print(gen)

        entity_match = re.search(r'Entity:\s*"?([^\n"]+)"?', gen)
        relation_match = re.search(r"Sequential relation:\s*\[([^\]]+)\]", gen)

        source_entity = entity_match.group(1).strip() if entity_match else None
        relation = [rel.strip() for rel in relation_match.group(1).split(',')] if relation_match else None
        # print(q)
        # print(source_entity)
        # print(relation)
        
        # new_kg = KnowledgeGraph()
        for index, et in enumerate(d['orig']['new_triples']):
            et_label = d['orig']['new_triples_labeled'][index]
            new_kg.add_triple(et[0], et, et_label)
            map_to_id[et_label[0]] = et[0]
            map_to_id[et_label[2]] = et[2]
            map_to_label[et[0]] = et_label[0]
            map_to_label[et[2]] = et_label[2]
            map_to_label[et[1]] = et_label[1]
        
        if source_entity not in map_to_id:
            print("not found source entity")
            break
        source_id = map_to_id[source_entity]
        
        try:
            reasoning_path, path_label = reasoning_on_KG(model, tokenizer, stopping_criteria2, stop2, source_id, relation, new_kg, select_task_prompt)
        except (KeyError, IndexError) as e:
            print(f"Error occurred: {e}")
            print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
            reasoning_path = []
            path_label = ""
            
        try:
            ans = map_to_label[reasoning_path[-1][2]]
        except (KeyError, IndexError) as e:
            print(f"Error occurred: {e}")
            print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
        
        is_correct = ans == d["new_answer"] or ans in d["new_answer_alias"]
        if is_correct:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
        
