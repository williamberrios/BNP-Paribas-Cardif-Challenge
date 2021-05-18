# -*- coding: utf-8 -*-
# +
import pandas as pd
import lightgbm as lgbm
import numpy as np
import os 
from sklearn import preprocessing

def label_encoding(df,label_cols = [], drop_original = True, missing_new_cat = True):
    '''
    Function to get the encoding Label of a variable
    Input:
    -df         : Dataframe al cual se le aplica one hot encoding
    -label_cols : Variables categoricas
    '''
    from sklearn import preprocessing
    df_     = df.copy()
    df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
    dict_le ={}
    if missing_new_cat:
        print('Mode: Missing as new category')
        for i in df_cols.columns:
            le = preprocessing.LabelEncoder()
            print('Label Encoding: ',i)
            df_cols[i] = df_cols[i].astype('str')
            le.fit(df_cols[i])
            df_cols[i] = le.transform(df_cols[i])
            var_name = i
            dict_le[var_name] = le
    else:
        print('Mode: Missing as -1')
        for i in df_cols.columns:
            df_cols[i] = df_cols[i].fillna('NaN')
            df_cols[i] = df_cols[i].astype('str')
            le = preprocessing.LabelEncoder()
            print('Label Encoding: ',i)
            a = df_cols[i][df_cols[i]!='NaN']
            b = df_cols[i].values
            le.fit(a)
            b[b!='NaN']  = le.transform(a)
            df_cols[i] = b
            df_cols[i] = df_cols[i].replace({'NaN':-1})
            var_name = i
            dict_le[var_name] = le

    df_ = pd.concat([df_ , df_cols], axis = 1) 
    if drop_original:
        df_.drop(columns = label_cols, inplace = True)
    return df_,dict_le

def apply_label_encoder(df,dict_label_encoder,drop_original = True, missing_new_cat = True):
    from sklearn import preprocessing
    df_     = df.copy()
    label_cols = [i[6:] for i in list(dict_label_encoder.keys())]
    df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
    if missing_new_cat:
        print('Mode: Missing as new category')
        for i in df_cols.columns:
            print('Applying Label Encoding: ',i)
            df_cols[i] = df_cols[i].astype('str')
            le = dict_label_encoder[i]
            df_cols[i] = le.transform(df_cols[i])
  
    else:
        print('Mode: Missing as -1')
        for i in df_cols.columns:
            df_cols[i] = df_cols[i].fillna('NaN')
            df_cols[i] = df_cols[i].astype('str')
            print('Applying Label Encoding: ',i)
            a = df_cols[i][df_cols[i]!='NaN']
            b = df_cols[i].values
            le = dict_label_encoder[i]
            b[b!='NaN']  = le.transform(a)
            df_cols[i] = b
            df_cols[i] = df_cols[i].replace({'NaN':-1})

    df_ = pd.concat([df_ , df_cols], axis = 1) 
    if drop_original:
        df_.drop(columns = label_cols, inplace = True)
    return df_  
# -


