import torch
import torch.nn as nn
import transformers

# +
class Roberta_Model(nn.Module):
    def __init__(self,pretrained_model,dropout = 0.1,word_embedding = 'second_last'):
        super(Roberta_Model, self).__init__()
        self.model = transformers.RobertaModel.from_pretrained(pretrained_model)
        self.last = nn.Linear(self.model.config.hidden_size, 1) #
        self.dropout = dropout
        self.word_embedding = word_embedding
        # Functions
        self._init_fc()
        
    def _init_fc(self):
        torch.nn.init.normal_(self.last.weight, std=0.02)
                        
    def _embeddings(self,input_ids,attention_mask):
        # Embedding from the second-to-last hidden layer
        outputs = self.model(input_ids, attention_mask,output_hidden_states = True)
        if self.word_embedding == 'second_last':
            sent_embed   = outputs[2][-2]
        elif self.word_embedding == 'last_four_layers':
            hidden_states = outputs[2]
            sent_embed    = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]],dim = -1)
        sent_embed   = torch.mean(sent_embed, dim=1)
        return sent_embed
            
    def forward(self, input_ids, attention_mask):
        _, out = self.model(input_ids = input_ids, attention_mask = attention_mask,return_dict=False)
        if self.dropout > 0:
            out    = nn.Dropout(self.dropout)(out)
        out    = self.last(out)
        return out
    
class XLMRoberta(nn.Module):
    def __init__(self,pretrained_model,dropout = 0.1,word_embedding = 'second_last'):
        super(XLMRoberta, self).__init__()
        self.model = transformers.XLMRobertaModel.from_pretrained(pretrained_model)
        self.dropout = dropout
        self.last = nn.Linear(self.model.config.hidden_size,1)
        self.word_embedding = word_embedding
    
    def _embeddings(self,input_ids,attention_mask):
        # Embedding from the second-to-last hidden layer
        outputs = self.model(input_ids, attention_mask,output_hidden_states = True)
        if self.word_embedding == 'second_last':
            sent_embed   = outputs[2][-2]
        elif self.word_embedding == 'last_four_layers':
            hidden_states = outputs[2]
            sent_embed    = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]],dim = -1)
        sent_embed   = torch.mean(sent_embed, dim=1)
        return sent_embed
    
    def forward(self,input_ids,attention_mask):
        _, out = self.model(input_ids = input_ids, attention_mask = attention_mask,return_dict=False)
        if self.dropout > 0:
            out    = nn.Dropout(self.dropout)(out)
        out    = self.last(out)
        return out

