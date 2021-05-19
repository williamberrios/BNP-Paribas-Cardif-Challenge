# +
import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
from transformers import AdamW
import sys
module_path = "../"
if module_path not in sys.path:
    sys.path.append(module_path)

from models.models import *


# +
def fetch_optimizer(name,lr,params):
    if name =='AdamW':
        optimizer =  AdamW(params, lr = lr)
    elif name =='Adam':
        optimizer =  torch.optim.Adam(params, lr = lr)
    else:
        print('optimizer Not implemented Yet')
        sys.exit() 
    return optimizer
        
def fetch_scheduler(name,optimizer,params):
    if name == 'linear_with_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = int(params['WARMUP_RATIO']*params['NUM_TRAIN_STEPS']),num_training_steps=params['NUM_TRAIN_STEPS'])
    elif name == None:
        scheduler = None
    else:
        print('scheduler Not implemented Yet')
        sys.exit() 
    return scheduler
