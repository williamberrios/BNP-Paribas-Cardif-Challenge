# +
import torch
class BNPParibasText(torch.utils.data.Dataset):
    def __init__(self,df,max_length,tokenizer,column):
        self.df = df
        self.max_length = max_length
        self.tokenizer  = tokenizer
        self.column     = column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = str(self.df[self.column][idx])
        sentence = " ".join(sentence.split())
        target = self.df.target[idx]
        encoded_dict = self.tokenizer.encode_plus(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True, # Add '[CLS]' and '[SEP]',
                              truncation=True,
                              max_length = self.max_length,           # Pad & truncate all sentences.
                              padding = "max_length",
                              return_attention_mask = True,   # Construct attn. masks.
                              return_tensors = 'pt',     # Return pytorch tensors.
                        )
        ids     = encoded_dict['input_ids'].flatten()
        mask    = encoded_dict['attention_mask'].flatten()
        #token_type_ids = encoded_dict["token_type_ids"].flatten()
        return {
              'ids': torch.as_tensor(ids, dtype=torch.long),
              'mask': torch.as_tensor(mask, dtype=torch.long),
              'target': torch.as_tensor(target, dtype=torch.float),
              #'token_type_ids' : torch.as_tensor(token_type_ids, dtype=torch.long),
          }
    
    
class BNPParibasTextFeatures(torch.utils.data.Dataset):
    def __init__(self,df,max_length,tokenizer,features):
        self.df = df
        self.max_length = max_length
        self.tokenizer  = tokenizer
        self.features    = features 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = str(self.df[self.column][idx])
        sentence = " ".join(sentence.split())
        target = self.df.target[idx]
        features = self.df.loc[idx,self.features].values.astype('float')
        encoded_dict = self.tokenizer.encode_plus(
                              sentence,                      # Sentence to encode.
                              add_special_tokens = True, # Add '[CLS]' and '[SEP]',
                              truncation=True,
                              max_length = self.max_length,           # Pad & truncate all sentences.
                              padding = "max_length",
                              return_attention_mask = True,   # Construct attn. masks.
                              return_tensors = 'pt',     # Return pytorch tensors.
                        )
        ids     = encoded_dict['input_ids'].flatten()
        mask    = encoded_dict['attention_mask'].flatten()
        
        #token_type_ids = encoded_dict["token_type_ids"].flatten()
        return {
              'ids': torch.as_tensor(ids, dtype=torch.long),
              'mask': torch.as_tensor(mask, dtype=torch.long),
              'features': torch.as_tensor(features, dtype=torch.float),
              'target': torch.as_tensor(target, dtype=torch.float)
              #'token_type_ids' : torch.as_tensor(token_type_ids, dtype=torch.long),
          }
