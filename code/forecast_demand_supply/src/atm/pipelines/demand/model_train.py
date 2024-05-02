import os
import numpy as np
import pandas as pd
from sklearn.model_ selection import ParameterGrid
from sklearn.model_selection import GridSearchcv
from sklearn.metrics import mean_absolute percentage_error
from statsmodels.tsa.exponential_ smoothing. ets import ETSModel
from pmdarima . arima import auto_arima
from statsmodels.tsa.holtwinters import Exponential Smoothing
from statsmodels.tsa. statespace.sarimax import SARIMAX
from fbprophet inport Prophet
import warnings
from datetime import datetime
import collections
import holidays
from xgboost import XGBRegressor
from sklearn.model selection import GridSearchCV
import warnings
from google.cloud import bigquery
warnings. filterwarnings"ignore"

#cLass to supress prophet Logs
class suppress_ stdout stderr(object) :
'''
A Context manager for doing a deep Suppressionof stdout and stderr in
Python, i.e. will suppress all print, even if the print originates in a
compiled C/Fortran sub-function.
This will not suppress raised exceptions since exceptions are printed
to stderr ust before a Script exits, and after the context manager has
exited (at least, I think that is why it lets exceptions through).
'''

def __init__(self) :
# Open a pair of nullfiLes
self.null_fds [os.open(os.devnull,, os.0_RDWR) For x in range (2) ]
# Save the actual stdout(1) and stderr(2) file descriptors.
self.save_fds (osdup(1), os.dup(2))

def __enter__(self):
# Assign the nulL pointers to stdout and stderr
os.dup2(self.null_fds[0], 1)
Os.dup2 (self. null_fds[1], 2)

def_exit_(self,*_):
#Re-assign the real. stdout/s tdenn back to (1) and (2)
os.dup2 (self. save_fds[o], 1)
os.dup2(self.save fds[1], 2)
# CLose the nulL fiLes
o5.close( self. null fds [0])
os.close(self.nullfds[1])

#Defining method for mape metnic this is to use as scoring method for autoarima as mape is not supported there by default
def mape(y true,y_ pred):  
  return mean_absolute percentageerror(y true y pred)
  
#hyper parameter tuning For ETS
def tune model ets(data train, data test,param grid):
'''
Perform hyper parameter tuning for ETS model
input: Train and test data and parameter to perform tuning on
Output: Best model, best parametems
'''
data_train['value']=data_train['value'].astype(float)
data_test['value']=data_test['value'].astype(float)
best score=np.inf
best params={}
for params in ParameterGrid (param_grid):
  model_ets=ETSModel(data_train['value'],
                     trend=params['trend'],
                     seasonal=params['seasonal'],
                     seasonal_periods=params['seasonal_periods'],
                     initialization_method='heuristic')
fitted_model_ets =model_ets.fit(smoothing_level=params['smoothing_level'],
                                smoothing_trend=params['smoothing trend'],
                                smoothing_seasonality=params['smoothing'],
pred=fitted_model_ets.forecast(len(data_test))
                                
score=mean_absolute_pencentage_error(data_test['value'],pred)

if best_score>score:
  best_Score=score
  best_params['param_best']=params
  best_model_ets=fitted_model_ets
return best_model_ets, best_params


# hyper parameter tuning for SARIMA
def tune_model_sarima (df_train, df_test, param_grid) :
  '''
  Perform hyper parameter tuning for SARIMA model
  input : Train and test data and parameter to perform tuning on
  output: Best model best parameters
  '''
  best_params={}
  #Training auto arima with seasonal component, so sarima modeling
  model_sarima=auto_arima(df_train["value"],
                          start p=param _grid['p_min'],
                          d=paramLgrid['d_min'],
                          start_qwparam_grid['q_min'],
                          max_p=param_grid['p_max'],
                          max_d=par am_grid['d_max'],
                          max_qparam_grid['q_max'],
                          start_p=param_grid['P_min'].
                          D=param_grid['D_min'],
                          start_Q-paran_grid['Q_min'],
                          max_P=param_grid['P_max'],
                          max_ D=par am grid['D_max 1'],
                          max_Q=param grid['Q_max'],
                          m=param_grid['m'],
                          seasonal=param_ grid['seasonal']
                          scoring=mape
                          )
  # get best parameters
  best_params_grid=model_sarima.get_params()
  best params['param_best']=best_params_grid
  model=SARIMAX(df_train['value'],
        order=best_params_grid['order'],
        seasonal_order=best_params_grid['seasonal_order'] )
  with suppress_stdout stderr():
    fitted_model_sarima=model.fit()
  return fitted_model_sarima,best_params


# Defining function for hyper parameter tuning of Triple exponential smoothing modeL
def param_tuning_tes(data_train, data_test, param_grid):
    '''
    Perfors hyper paraneter tuning for Triple Expoent ial Seoothing nodel
    inout Teain and test dsta and parameters to perforn tuning on
    outpt Best model, best parameters
    '''
    warnings.flterwarnings('ignore')
    best_score=np.inf
    best_params_tes={}
    for params in ParameterGrid(param_grid) :
        model=ExponentialSmoothing(data_train['value'],
                                  trend=params['trend'],
                                  seasonal=params['Seasonal'],
                                  seasonal_period=params[ 'seasonal_periods']
                                  )
      
        with suppress stdout_stderr() : 
          fitted_model=model.fit(smoothing_level=params['smoothing_level'], 
                                 smoothing_trend=params['smoothing_trend'])
        pred=fitted_model.forecast(len(data_test))
        mape_score=mean_absolute_percentage_error(pred, data_test['value'])
        if best_score>mape_score:
        best_score=mape_score
        best_params_tes['param_best']=params
        best_model_tes=fitted_model
    return best_model_tes, best_params_tes
    
# Defining method to prepare data for stacking

def prep data stack(level,forecast_input,selected_rule_type, best_models,train_start_date, train_end_date,test_start_date,test_end_date,num_logs):
    '''
    Prepare data in the form neady for stacking modeling (XGB0ost Regresion)
    Make upto 6 lag data as attribute of predicting data 1ike lagi, lag2, lag3, lag4, lag5, lag6
    Get average of last 3 months data as one attribute of predicting data
    Get Average of last 3 months prediction of ets and sarima model as 2 attributes of predicting data
    
    input : Dataframe, number of lag ets and sarima fitted imodel, training starting date
    output: Training predecting data training target data, testing predecting data, testing ta
    '''
    stack dict = collections.defaultdict(dict)
    train_start_date=pd.to_datetime(train_start_date)
    train_end_date=pd.to_datetime(train_end_date)
    test_start_date=pd.to_datetime(test_start_date)
    test_and_date-pd.to_ datetime(test enddate)
    df=forecast_input[forecast_input['rule_type']==selected_rule_type]
    data_df_Stack=df.copy()
    my dict = (tup['0']: tup[1l for tup in best_models[selected_rule_type)
    # Taking 6 Lags for each month data as trLbute
    for i in range(1, num_lags+1):
    data df_stack['lag_{}'.format(i)=data_df_stack['value'].shift(i)
    
    # Filling missing value created due to taking Lag Filled issing vaLue by Lag walue of same year
    # getting List of coLumns starts wth Lag
    cols_lag=[col for col in data_df_stack.columns if col.startswith('lag')]
    for n in range(num_lags):
        index=data_df_stack.index[n]
        for i,col in enumerate(cols_lag[n:]):
            if(level=='monthly"):
                new_index_lag=index-pd.DateOffset(months(i+n+1)+ pd.date0ffset(months=12)
            elif(level=='daily'):
            new_index_lag= index - pd.DateOffset(days=(i+n+1) +pd.DateOffset(months=12)
            data_df_stack.loc[index,col]=data_df_stack.loc[new_index_lag,'value']
    






