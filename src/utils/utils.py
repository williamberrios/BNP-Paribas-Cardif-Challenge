# -*- coding: utf-8 -*-
# +
import pandas as pd
import lightgbm as lgbm
import numpy as np
import os 
import random
import torch 
def extract_time(df,column):
    date    = pd.to_datetime(df[column])
    df[f'{column}_year']   = date.dt.year.values
    df[f'{column}_month']  = date.dt.month.values
    df[f'{column}_day']    = date.dt.day.values
    df[f'{column}_hour']   = date.dt.hour.values
    df[f'{column}_minute'] = date.dt.minute.values
    df[f'{column}_second'] = date.dt.second.values
    df.drop(columns = [column],inplace = True)

def multiple_extract_time(df,columns):
    for i in columns:
        extract_time(df,i)
        
def resumen_vars(df,columns):
    for i in columns:
        print(f"{'*'*20} Column: {i} {'*'*20}")
        print(f'- Value counts:')
        print(df[i].value_counts(dropna = False))
        print(f'- Nunique values {i}:\n',df[i].nunique())
        
def porc_na(df):
    df_na = pd.DataFrame({'Variable'      : df.columns.values,
                          'Missing (%)'     : np.round(100 * df.isna().sum().values / df.shape[0] ,3)})
    return df_na

def isNaN(string):
    return string != string

def split_double_dot(row,column):
    if isNaN(row[column]):
        return np.nan,np.nan
    elif len(row[column].split(':'))==1:
        return np.nan,row[column]
    else:
        list_split = row[column].split(':')
        return list_split[0],list_split[1]
    
def split_by_comma(row,column):
    list_split = row[column].split(',')
    return list_split[0],list_split[1],list_split[2],list_split[3],list_split[4],list_split[5],list_split[-6],list_split[-5],list_split[-4],list_split[-3],list_split[-2],list_split[-1]

def func_formatting(row):
    if isNaN(row):
        return np.nan
    else:
        row = row.replace('-',' ').lower()
        row = " ".join(row.split())
        return row
    
def func_geolocation(row):
    if isNaN(row['first_packaging_code_geo']):
        return np.nan,np.nan
    else:
        return row['first_packaging_code_geo'].split(',')[0],row['first_packaging_code_geo'].split(',')[1]
    
def func_sort_split_comma(row):
    if isNaN(row):
        return np.nan
    else:
        row = row.replace('-',' ').lower()
        row = " ".join(row.split(' '))
        row = row.split(',')
        row.sort()
        row = ",".join(row)
        return row    

def func_states(row,column):
    lista_rm = []
    splits   = row[column].replace('en:','').lower().split(',')
    indexes  = ['nutrition facts','ingredients','expiration date','packaging-code-','characteristics','categories','brands','packaging','quantity','product name']
    dict_ = {}
    for sentence in splits:
        for idx in indexes:
            if idx in sentence:
                if idx == 'packaging' and ('packaging-code-' in sentence):
                    continue
                else:
                    dict_.update({idx:sentence.split(f'{idx}')[1].strip().replace('-',' ')})
                    lista_rm.append(sentence)
    
    list_keys  = list(dict_.keys())
    list_falta = list(set(indexes)-set(list_keys))
    
    for i in list_falta:
        dict_.update({i:np.nan})
    #Complete first 2 and last 2
    for i in set(lista_rm):
        splits.remove(i)
    
    lista_rm = []
    indexes_2 = ['check','complete','validate','upload']
    for sentence in splits:
        for idx in indexes_2:
            if idx in sentence:
                if (idx=='check')|(idx=='complete'):
                    dict_.update({f'general_{idx}':sentence})
                    lista_rm.append(sentence)
                    
                if (idx=='validate')|(idx=='upload'):
                    dict_.update({f'photo_{idx}':sentence})
                    lista_rm.append(sentence)
    list_keys  = list(dict_.keys())
    list_falta = list(set(['general_check','general_complete','photo_validate','photo_upload'])-set(list_keys))
    
    for i in list_falta:
        dict_.update({i:np.nan})
    
    # Sort dictionary
    dict_ = dict_.items()
    dict_ = sorted(dict_)
    dict_ = dict(dict_)

    # Return
    #'brands': 'completed',
    #'categories': 'completed',
    #'characteristics': 'completed',
    #'expiration date': 'to be completed',
    #'general_check': 'to be checked',
    #'general_complete': nan,
    #'ingredients': 'completed',
    #'nutrition facts': 'completed',
    #'packaging': 'completed',
    #'packaging-code-': 'to be completed',
    #'photo_upload': 'photos uploaded',
    #'photo_validate': 'photos validated',
    #'product name': 'completed',
    #'quantity': 'completed'

    #['brands','categories','characteristics','expiration date','general_check','general_complete',
    # 'ingredients','nutrition facts','packaging','packaging-code-','photo_upload','photo_validate',
    #'product name','quantity']
    
    return tuple(dict_.values())

def generate_col_unique(df,cols_encode):
    col_unique = ''
    suma       = ''
    for i in cols_encode:
        col_unique = col_unique + i + '_'
        suma = suma + ', '+df[i]

    df[col_unique] = suma
    return col_unique

def seed_everything(seed=42):
    '''
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

