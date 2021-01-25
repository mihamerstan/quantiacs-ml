import quantiacsToolbox
from quantiacsToolbox import loadData, fillnans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, metrics


## DATA PREP
def generate_returns(CLOSE,RINFO):
    RET = np.zeros([1,CLOSE.shape[1]])
    RET = np.append(RET,np.float64(CLOSE[1:, :] - \
                    CLOSE[:-1, :] - RINFO[1:, :]) / CLOSE[:-1, :],axis=0)
    RET = fillnans(RET)
    RET[np.isnan(RET)] = 0
    return RET

def generate_returns_dict(quant_dict,col='CLOSE'):
    '''Wrapper that calls generate_returns and adds the RET outcome to quant_dict.'''
    
    quant_dict['RET'] = generate_returns(quant_dict['CLOSE'],quant_dict['RINFO'])
    return quant_dict

def quant_dict_to_df(quant_dict,settings):
    '''NOT ACTIVE CURRENTLY, but should work. 
    Function transforms quant_dict returned from loadData function (plus returns)
    into dataframe. Each row in dataframe is a day for a particular market.'''

    # MARKET and DATE must be in col_list
    if 'markets' not in list(settings.keys()):
        print("MARKET must be in col_list")
        return
    if 'DATE' not in list(quant_dict.keys()):
        print("MARKET must be in col_list")
        return

    # Set up df by market and date
    df = pd.DataFrame([[m,d] for m in settings['markets'] for d in quant_dict['DATE']],columns=['MARKET','DATE'])

    # Flatten out the data and append columns
    for col in quant_dict.keys():
        if col != 'DATE':
            # Fillnans with zero for now
            quant_dict[col][np.isnan(quant_dict[col])] = 0
            df[col] = quant_dict[col].flatten('F')

    return df


## MODEL EVALUATION
def eval_model(reg, X_train, X_val, y_train, y_val, 
               metric_dict, binary_list, metric_df = None):
    '''eval_model() generates train and val evaluation metrics for reg
    regression model.
    Inputs:
    reg - Pre-trained regression model 
    X_train, X_val - Feature Dataframes
    y_train, y_val - Returns to predict
    metric_dict - Dictionary of metrics to calculate
    binary_list - List of metrics that operate on sign(y)
    metrif_df - optional, if provided then metrics for reg will be appended
    
    Outputs:
    metric_df - DataFrame w/ rows of metric_list and columns train and val
    '''
    # Predictions
    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)

    # Define the empty metrics dataframe (if not provided)
    if metric_df is None:
        df_ix = [a+b for b in ['_train','_val'] for a in metric_dict]
        metric_df = pd.DataFrame(index=df_ix)
    
    # Metrics calculation loop
    reg_name = str(reg)[:15]
    for met in metric_dict:
        if met in binary_list:
            metric_df.loc[met+'_train',reg_name] = metric_dict[met]( \
                          np.sign(y_train),np.sign(y_train_pred))
            metric_df.loc[met+'_val',reg_name] = metric_dict[met]( \
                          np.sign(y_val),np.sign(y_val_pred))
        else:
            metric_df.loc[met+'_train',reg_name] = metric_dict[met]( \
                          y_train,y_train_pred)
            metric_df.loc[met+'_val',reg_name] = metric_dict[met]( \
                          y_val,y_val_pred)
    return metric_df	


### FEATURE GENERATION
# LAG
def lag(RETURNS,lag):
    '''lag returns a pd.Series of the lagged values in col, and the df with the lag column appended. 
       First #(lag) rows are zeroes (may change this later to remove these columns)'''

    out = np.zeros([lag,RETURNS.shape[1]])
    out = np.append(out, RETURNS[:-lag,:],axis=0)
    return out

def lag_feats(RETURNS, lag_list, remove_rows=True):
    '''lag_feats generates lag features, as defined by lag_list, 
    and removes initial rows that do not have previous information for lag.
    Output is a list of arrays with RETURNS.shape.'''
    # Add lag columns
    out_list = []
    for l in lag_list:
        out_list.append(lag(RETURNS,l))
    if remove_rows:
        # Remove the first max_lag rows from each list
        max_lag = np.max(lag_list)
        for i,ll in enumerate(out_list):
            out_list[i] = ll[max_lag:]            
    return out_list

def shape_lag_features(data_dict):
    '''This function reshapes data_dict into X and y, accounting for lags.
    Should eventually be expanded to handle other features.'''
    
    max_lag = np.max(data_dict['LAG_LIST'])

    # Create X and y
    y = data_dict['RET'][max_lag:,:].flatten('F')
    full_lag_arr = np.array(data_dict['LAG_FEATURES'])
    # Remove early returns that don't have lag
    lag_arr = full_lag_arr[:,max_lag:,:]

    # Reshape for model
    lag_arr = lag_arr.reshape((lag_arr.shape[0],-1),order='F').T
    y = y.reshape(-1,order='F')

    return lag_arr, y



