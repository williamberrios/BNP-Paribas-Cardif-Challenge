import torch
import torch.nn as nn
import transformers

class Roberta_Model(nn.Module):
    def __init__(self,pretrained_model,dropout = 0.1,reinit_layers = 0):
        super(Roberta_Model, self).__init__()
        self.model = transformers.RobertaModel.from_pretrained(pretrained_model)
        self.last = nn.Linear(self.model.config.hidden_size, 1) #
        self.dropout = dropout
        # Functions
        self._init_fc()
        
    def _init_fc(self):
        torch.nn.init.normal_(self.last.weight, std=0.02)
                        
    def _embeddings(self,input_ids,attention_mask):
        outputs = self.model(input_ids, attention_mask,output_hidden_states = True)
        sent_embed   = outputs[2][-2]
        sent_embed   = torch.mean(sent_embed, dim=1)
        return sent_embed
            
    def forward(self, input_ids, attention_mask):
        _, out = self.model(input_ids = input_ids, attention_mask = attention_mask,return_dict=False)
        if self.dropout > 0:
            out    = nn.Dropout(self.dropout)(out)
        out    = self.last(out)
        return out

