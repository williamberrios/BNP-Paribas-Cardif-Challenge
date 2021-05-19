# +
#from tqdm.notebook import tqdm
from tqdm import tqdm
from .AverageMeter import AverageMeter
import torch
def train_fn(data_loader,model,criterion,optimizer,device,scheduler,mode_sched = 'on_epoch'):
    if (mode_sched == 'on_epoch')|(mode_sched == 'on_batch'):
        print('Mode Scheduler: OK')
    else:
        print(f"Error!! {mode_sched} doesn't exist, choose another mode")
    # Put model in train mode
    model.train()
    # Initialize object Average Meter
    losses = AverageMeter()
    tk0 = tqdm(data_loader,total = len(data_loader))
    for b_idx,data in enumerate(tk0):
        for key,value in data.items():
            data[key] = value.to(device)
        optimizer.zero_grad()
        output = model(data['ids'],
                       data['mask'])

        loss = criterion(output,data['target'].view(-1,1))
        loss.backward()
        optimizer.step()
        # Scheduler on_batch
        if (scheduler is not None) & (mode_sched == 'on_batch'):
            scheduler.step()  
        losses.update(loss.detach().item(), data_loader.batch_size)
        tk0.set_postfix(Train_Loss=losses.avg,LR=optimizer.param_groups[0]['lr'])
    # Scheduler on_epoch
    if (scheduler is not None) &(mode_sched == 'on_epoch'):
        scheduler.step()
    
    print(f'Training -> Loss: {losses.avg}')
    return losses.avg 

# Valid Loop
def valid_fn(data_loader,model,criterion,device,steps = None):
    # Put model in train mode
    model.eval()
    # Initialize object Average Meter
    losses = AverageMeter()
    tk0 = tqdm(data_loader,total = len(data_loader))
    for b_idx,data in enumerate(tk0):
        for key,value in data.items():
            data[key] = value.to(device)
            
        output = model(data['ids'],
                      data['mask'])
      
        loss = criterion(output,data['target'].view(-1,1))
                
        losses.update(loss.detach().item(), data_loader.batch_size)
        tk0.set_postfix(Eval_Loss=losses.avg)
    
    print(f'Validation -> Loss: {losses.avg}')
    return losses.avg
