from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import gc 
import torch
from torch.utils.data import Dataset
import json
import os 



class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.file = json.load(open(path, "r"))

        
        self.data = [] # To store the dialogue 

        # Reading the dialogues 
        for idx, dialog in enumerate(self.file):
            self.data.append([]) # new dialogue
            for txt in dialog['dialog']:
                self.data[idx].append(txt['text'])
        # Formating the conversations 
        self.X = []
        int1 = ''
        int2 = ''
        for idx, dialog in enumerate(self.data):
            if len(dialog)==0:
                continue
            if len(dialog) % 2 == 0:
                int1 = '<per>'
                int2 = '<bot>'
            else:
                int1 = '<bot>'
                int2 = '<per>'
            tmp_str = '<sos>'
            for k,utt in enumerate(dialog):
                if k%2 == 0:
                    tmp_str += int1 + utt
                else:
                    tmp_str += int2 + utt
            tmp_str += '<eos>'
            # print(len(tmp_str))
            self.X.append(tmp_str)

        # max = 0
        # for dialog in self.X:
        #     if len(dialog)>max:
        #         max= len(dialog)
        # print(max)
    
        
        # Tokenization of the dataset 
        self.X_encoded = tokenizer(self.X,max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])