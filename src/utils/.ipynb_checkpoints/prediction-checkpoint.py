# +
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from .utils import post_processing
import pandas as pd
import transformers


def get_prediction(data_loader, model, device):
    # Put the model in eval mode
    model.eval()
    # List for store final predictions
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(device)
            predictions = model(data['ids'],
                      data['mask'])
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return np.vstack(final_predictions)

def get_embedding(data_loader, model, device):
    # Put the model in eval mode
    model.to(device)
    model.eval()
    # List for store final predictions
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(device)
            predictions = model._embeddings(data['ids'],data['mask'])
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return np.vstack(final_predictions)

def predict(df,models,features):
    probs = 0
    for i in models:
        probs = probs + (i.predict(df[features]))
        print('fin_predict')
    y_test_pred = probs/5.0
    df['target'] = y_test_pred
    post_processing(df)
    return df