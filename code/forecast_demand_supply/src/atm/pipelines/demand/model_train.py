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

    # Taking moving average window 3 (Last 3 months data) as an attribute
    data_df_stack['ma_3']=(data_df_stack['lag_1']+data_df_stack['lag2']+data_df_stack['lag3'])/3

    # Taking ETS and sarima model prediction as attribute

    ets_pred_stack=my_dict['ETS'].predict(start=train_start_date, end=test_end_date)
    sarima_pred_stack=my_dict["SARIMA"].predict(start train start_date, end=test_end d
    prophet_pred_stackmy_dict['PROPHET'].predict(pd. DataFrame(pd.date_range(start=train_start_date,end=test_end_date),columns=['ds']))[['ds','yhat']]
    prophet_pred_stack.set_index('ds',inplace-True)

    ets_pred_stack=pd.Dataframe(ets pred stack)
    ets_pred_stack.columns=['ets_prediction']

    sarima_pred_stack=pd.Dataframe(sarima pred stack)
    sarima_pred_stack. columns=['sarima_ prediction']

    prophet_pred_stack=pd.Dataframe(prophet_pred_stack)
    prophet_pred_stack.columns=['prophet_prediction']

    ets_pred_last_3_avg-ets_pred_stack.rolling (window=3).mean().shift(1)
    # FilLing missing value created due to roLLing and shifting to get avg ets prediction for last 3 months
    m=3
    for w in range(m) :
        ets_pred_last_3_avg.iloc[w, ]=ets_pred_stack.iloc[w, ][0]
    # FiLLing missing value created due to rolling and shifting to get avg samima prediction for last 3 months
    sarima_pred_last_3_avg-sarima_pred_stack.rolling (window=3).mean().shift(1)
    for w in range(m) :
        sarima_pred_last_3_avg.iloc[w, ]=sarima_pred_stack.iloc[w, ][0]
      
    prophet_pred_last_3_avg=prophet_pred stack.rolling(windoW=3).mean().shift(1)
    # FiLLing missing value created.due to roliing and shifting to get avg tes prediction for last 3 months
    m=3
    for w in range(m):      
      prophet_pred_last3_avg.iloc[w, ]=prophet_pred_stack.iloc [w, ][0]
      
    # Merging 6 Lags data, moving average data, ets prediction and sarima prediction
    data_df_stack=data_df_stack.merge(ets_pred_last_3_avg,left_index=True,right_index=True).merge(sarima_pred_last_3_avg,left_index=True,right_index=True).merge(prophet_pred_last3_avg,left_indexe=True,right_index=True)
    
    # Splitting data into train and test data
    traindf_stack=data_df_stack[ :train_end_date]
    test_df_stack=data_df_stack[test_start_date:]
      
    cols_stack_x=cols_lag+[ 'ma_3', 'ets_prediction','sarima_prediction','prophet_prediction']
    x_train_stack=train_df_stack[cols_stack_x]
    y_train_stack=train_df_stack['value']
      
    x_test_stack=test_df_stack[cols_stack_x]
    y_test_stack=test_df_stack['value']
      
    stack_dict[selected_rule_type] ['x_train_stack' ]=x_train_stack
    stack dict[selected_rule_type]['y_train_stack']=y_train_stack
    stack_dict[selected_rule_type]['x_test_stàck']=x_test_stack
    stack_dict[selected_rule_type]['y_test_stack']=y_test_stack
      
    return stack_dict
      
#Defining method to train and tune stacking (XG8)
def tune_model_stack (level, forecast_input, selected_rule_type, best_models, train_start_date,train_end_date,test_start_date,test_end_date,num_lags,params_stack):
    '''
    Performing hyper parameter tuning of stacking modeling
    Input : x_train,y-train,x_ test,y_test, parameters
    Output : best stack model best parmeters conbination
    '''
    
    stack_dict=prep_data_stack(level, forecast_input, selected_rule_type, best_models,train_start_date,train_end_date,test_start_date,test_end_date,num_lags)
    x_train=stack_dict[selected_rule_type]['x_train_stack']
    y_train=stack_dict[selected_rule_type ]['y_train_stack']
    x_test=stack_dict[selected_rule_type]['x_test_stack']
    y_test-stack_dict[selected_rule_type]['y_test_stack']
  
    model_xgb_sample-XGBRegressor()
    grid_search_stack-GridSearchCV(estimatormodel_xgb_sample, param_grid=params_stack, cv=2,n_jobs=-1)
    grid_search_stack.fit(x_train,y_train)
    best_params_stack=grid_search_stack.best_params_
    best_model_stack=grid_search_stack.best_estimator_
  
    return selected_rule_type, best_model_stack, best_params_stack,x_train,y_train,x_test,y_test
  
def get_mape_stack (model_stack, x_test,y_test) :
    y_pred=model_stack.predict(x_test)
    mape_stack=mean_absolute_percentage_error(y_test,y_pred)
    return mape_stack
  
def train_model_ets(data , params) :
    '''
    Train the ETS model
    This function is used to retrain the model with whole available data
    Arguments :
                Data (Dataframe) - whole avilable data
                parmametrs - best parameters
    Output : ETS Model trained on whole available data with
    '''
    data['value']=data['value'].astype(float)
    model_ets=ETSModel(data['value'],
                      trend=params['trend'],
                      seasonal=params['seasonal'],
                      seasonal_periods=params['seasonal_periods'],
                      initialization_method='heuristic')
    with suppress_stdout_stderr():
       fitted_model ets-model_ets.fit(smoothing_level=params['smoothing level'],
                                      smoothing_trend=params['smoothing_trend'],   
                                      smoothing_seasonality=params['smoothing_seasonality']
    return fitted_model_ets
      
def train_model_sarinax(data,params):
    '''
    Train the SARIMAX model
    This function is used to retrain the model with shole available data
    Arguments :
                Data (Dataframe) - whole avilable data
                parmametrs - best parameters
    Output : SARIMAX Model trained on whole available data with
    '''
    model_sarima =SARIMAX(data['value'],
                          order=params['order'],
                          seasonal_order=params['seasonal_order'])
    with suppress_stdout_stderr():
        fitted model_sarima=model_sarima.fit()
    return fitted_model_sarima


def train_model_tes (data, params):
  '''
    Train the Triple Exp Smoothing model
    Thỉs function is used to retrain the model with whole available data
    Arguments :
    Data (Dataframe) - whole avil able data
    parmametrs - best pan ametems
    '''
    model_tes=ExponentialSmoothing(data['value'],
                                    trend=params['trend'],
                                    seasonal=panams['seasonal'],
                                    seasonal_periods=params['seasonal_periods']
    with suppress_stdout_stderr():
       fitted model_tes=model_tes.fit(smoothing_level=params['smoothing_level'],
                                      smoothing_trend=params['smoothing_trend'])
    return fitted_model_tes

def train_model_stack(x_train,x_test,y_train,y_test, params):
    '''
    Train the stack model
    This function is used to retrain the model with whole available data
    Arguments :
    Data (Dataframe) - whole available data , predicting data and target data
    parmametrs- best parameters
    '''
    from xgboost import XGBRegresşor
    X=pd.concat([x_train, x_test])
    y=pd.concat( [y_train, y_test])
    fitted_model_xgb=XGBRegressor(max_depth=params['max_depth'],
                                  learning_rate=params['learning_rate'],
                                  n_estimators=params['n_estimators'])
    fitted_model_xgb.fit(x, y)
    return fitted_model_xgb
                                    
def prep_data_stack_1_period (level,y_test_stack, num_lags, ets_fitted, sarima_fitted,prophet_fitted):
    '''
    Prepare data for stacking for future one period
    args :
    return :
    data (Data frame)- predicting data (x) for one future period (month)
    '''
    if(level=='monthly') :
        forecast_date=y_test_stack.index[-1]+pd.DateOffset(months=1)
    elif (level='daily' ):
        forecast_date=y_test_stack.index[-1]+pd.Dateoffset(days=1)
    forecast_start_date_temp=forecast_date-pd.DateOffset(months=3)
    forecast_end_date_temp=forecast_date- pd.DateOffset(months=1)
  
    data_forecast_temp=pd.DataFrame([])

    # Taking 6 Lags for each month data as atrribute
    for i in range(1, num_lags +1):
       data_forecast_temp['lag_{}'.format(i)] =[y_test_stack[-i]]
      
    data_forecast_temp['ma_3']=(data_forecast_temp['lag_1']+data_forecast_temp['lag_2']+data_forecast_temp['lag3'])/3
    # Taking average of Last 3 months forecasting of ets, sarima and tes mode'
    data_forecast_temp['ets_prediction' ]=ets_fitted.predict(start=forecast_start_date_temp, end=forecast_end_date_temp).mean()
    data_forecast_temp['sarima_prediction' ]=sarima_fitted.predict(start=forecast_start_date_temp, end=forecast_end_date_temp).mean()
  
    data_forecast_proph=prophet_fitted.predict(pd.Dataframe(pd.date_range(start=forecast_start_date_temp, end=forecast_end_đate_temp), colums=['ds']))[['ds','yhat']]
    data_forecast_proph.set_index('ds',inplace=True)
    data_forecast_temp['prophet_prediction']=data_forecast_proph.mean()
  
    return data_forecast_temp, forecast_date
  
def forecast_stack(level,y_train_stack,y_test_stack, num_lags,ets_fitted, sarima_fitted, prophet_fitted, fitted_model_stack, forecast_period):
    y_stack=pd.concat([y_train_stack,y_test_stack]),
    stack_forecasted_data-pd.Series ([])
    for i in range(forecast period) :
    # Prepare data for stacking for future one period
        x_forecast_temp, forecast_date_temp=prep_data_stack_1_period (level, y_stack, num_lags,ets_fitted, sarima_fitted, prophet_fitted)
    # forecasting by stack for future 1 period
        y_forecast_temp=fitted_model_stack.predict(X_forecast_temp)

        # Setting index on forecaşting x variables and y variabLe, index in forecasting date
        y_forecast_temp=pd.Series (y_forecasttemp, index=[forecast_date_temp])
        y_stack=y_stack.append(y_forecast_temp)
        stack_forecasted_data=pd.concat([stack_forecasted_data, y_forecast_temp])
    return stack_forecasted_data
  
def tune_model_prophet (data_train, data_test, param_grid):
    data_train.reset_index(inplace=True)
    data_test.reset_index(inplace=True)
    data_train.rename(columns={'call_answer_dt':'ds','value':'y'),inplace=True)
    data_test.rename (columns={'call_answer_dt': 'ds','value':'y'},inplace=True)
    #Logging. getLogger('fbprophet'). setLevel (Logging.WARNING)
    best_score=float('inf')
    best_params_prophet-{}
    for params in list (ParameterGrid(param_grid)):
        model=Prophet(**params)
   with suppress_stdout_stderr():
     model.fit(data_train)
     
   forecast=model.predict(data_test.drop(columns='y')


    mape_score=mean_absolute_percentage_error(data_test['y'], forecast['yhat'])

    if best score>mape SCore:
        best_Score=mape_score
        best_params_prophet['param_best']=params
        best_model_prophet=model

        return best_model_prophet, best_params prophet

def train_model_prophet(data train, params):
   '''
    Train the Prophet mOdel
    This Function i5 U5ed to retrain the model with nhole available data
    Arguments
    Data (Dataframe)- whole avilable data
    parmametrs best parameters
    Output: Prophet Model trained on whole available data with
    '''
    data_train=data_train.reset_index()
    data_train.rename(columns=('call_answer_dt':'ds','value':'y'}, inplace=True)
    model=Prophet (
                    seasonality_mode=params ['seasonality_mode'],
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                  )
    with suppress_stdout_stderr():
        model.fit(data train)

    return model 




        









                      



    
    
    
    
    
       
        
    
    
    
    
    
