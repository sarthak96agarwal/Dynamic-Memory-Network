import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
random.seed(1024)


class DMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(DMN, self).__init__()
        
        self.hidden_size=hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fact_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.ques_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn_weights = nn.Sequential(nn.Linear(4*hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Softmax())
        
        self.epsisodic_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.ans_grucell = nn.GRUCell(2*hidden_size, hidden_size)
        
        self.ans_fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_p)
    
    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    def init_weight(self):
        nn.init.xavier_uniform(self.embedding.state_dict()['weight'])

        for name, param in self.fact_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.ques_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.attn_weights.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.epsisodic_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.ans_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_normal(self.ans_fc.state_dict()['weight'])
        self.ans_fc.bias.data.fill_(0)
        
    def forward(self, facts, facts_masks, question, question_masks, num_decode, episodes=3, is_training=True):
        #input module
        concated=[]
        for fact, fact_mask in zip(facts, facts_masks):
            embedded = self.embedding(fact)
            if(is_training):
                embedded = self.dropout(embedded)
            hidden = self.init_hidden(fact)
            output, hidden = self.fact_gru(embedded, hidden)
            hidden_real = []
            for i, o in enumerate(output):
                length = fact_mask[i].data.tolist().count(0)
                hidden_real.append(o[length-1])
            concated.append(torch.cat(hidden_real).view(fact.size(0), -1).unsqueeze(0)) 
        encoded_facts = torch.cat(concated)
        #question module
        hidden=self.init_hidden(question)
       
        embedded = self.embedding(question)
        if(is_training):
                embedded = self.dropout(embedded)
        output, hidden = self.ques_gru(embedded, hidden)

        if is_training == True:
            real_question = []
            for i, o in enumerate(output): # B,T,D
                real_length = question_masks[i].data.tolist().count(0)

                real_question.append(o[real_length - 1])

            encoded_question = torch.cat(real_question).view(questions.size(0), -1) # B,D
        else: # for inference mode
            encoded_question = hidden.squeeze(0) # B,D
            
        #episodic memory module
        
        memory = encoded_question
        T_C = encoded_facts.size(1)
        B = encoded_facts.size(0)
        for i in range(episodes):
            hidden = self.init_hidden(encoded_facts.transpose(0, 1)[0]).squeeze(0) # B,D
            for t in range(T_C):
               
                z = torch.cat([
                                    encoded_facts.transpose(0, 1)[t] * encoded_question, # B,D , element-wise product
                                    encoded_facts.transpose(0, 1)[t] * memory, # B,D , element-wise product
                                    torch.abs(encoded_facts.transpose(0,1)[t] - encoded_question), # B,D
                                    torch.abs(encoded_facts.transpose(0,1)[t] - memory) # B,D
                                ], 1)
                g_t = self.attn_weights(z) # B,1 scalar
                hidden = g_t * self.epsisodic_grucell(encoded_facts.transpose(0, 1)[t], hidden) + (1 - g_t) * hidden
                
            e = hidden
            memory = self.memory_grucell(e, memory)
        
        # Answer Module
        answer_hidden = memory
        start_decode = Variable(LongTensor([[word_to_index['<s>']] * memory.size(0)])).transpose(0, 1)
        y_t_1 = self.embedding(start_decode).squeeze(1) # B,D
        
        decodes = []
        for t in range(num_decode):
            answer_hidden = self.ans_grucell(torch.cat([y_t_1, encoded_question], 1), answer_hidden)
            decodes.append(F.log_softmax(self.ans_fc(answer_hidden),1))
        return torch.cat(decodes, 1).view(B * num_decode, -1)