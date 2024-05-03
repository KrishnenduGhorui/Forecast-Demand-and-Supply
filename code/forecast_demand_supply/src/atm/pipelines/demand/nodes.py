from typing import Any, Dict
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib. pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.dates as mdates
from datetime import date
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import timedelta
from joblib import Parallel, delayed
from hyperopt import STATUS_OK, Trials, fmin, tpe
from fbprophet import Prophet
from hyperopt import hp
from hyperopt.pyll import scope
from fbprophet . diagnostics import cross_validation, performance_ metrics
from .model_train import tune_model_ets,tune_model_sarima, param_tuning_tes, mape, tune
from .model_train import train_model_ets, train_model_sarimax, train model_tes
from .model_train import prep_data_stack, tune_model_stack, train_model_stack, prep_dat
from .model_train import get_avg_volume_holiday, handle_volume_holiday
from sklearn.metrics import mean absolute percentage_error
import collections
from .etl import sql_table_mapping, prepare_sql_strings
import logging

def last_day_month(run_date):
    date_obj = datetime.strptime (run date,"SY-%m-%d")
    first_day_next_month = date_obj.replace(day=1) + timedelta(days=32)
    last_day_current_month= first_day_next_month.replace(day=1)-timedelta(days=1)
    date_str = last_day_current_month.strftime('%Y-%m-%d')
    rpt_dt = pd.to_datetime(date_str).date()
    return rpt_ dt
def load_data(sq1_file_path, ba_project, parameters):
    '''
    input_param : bq project
    output data frame with bgco historic call values
    This function will will load data from big query tables
    '''
  
    client = bigquery.client (project=bq_project) 
    with open(sql_file_path, 'r') as file:
      sql_read=file.read()
    table_map=sql_table_mapping(parameters)
    query=prepare_sql_strings(table_map, sql_read, parameters)
    df = client.query(query).to_dataframe()
    logging.info('Loading data is completed')
  
    return df
  
def load_demand_data_queue (df,key_arg_queue):
    '''
    This function will will load data from big query tables
    input_param : bq_project
    output : data frame
    '''
  
    df_demand=df[[' call_answer_dt', 'rule_type ', key_arg_queue]]
    df_demand['call_answer_dt']=pd.to_datetime (df_demand['call_answer_dt'])
    df_demand=df_demand.sort_values(by=['rule_type','call_ansiwer_dt'])
    df_demand.rename(columns={key_arg_queue:'value'},inplace=True)
    df_demand.dropna(inplace=True)
    df_demand.set_index('call_answer_dt',inplace=True)
    return df_demand
  
def input_validation (df, level, parameters) :
    function to validate input for demand
    #checking number of values in input df
    def diff_month(d1, d2):
        return (dl.year - d2.year)* 12 + dl.month d2. month
    run_date = pd.to_datetime(parameters['run_date'])
    if(level==monthly'):
        month_back = run date -relativedelta(months=1)
        month_back_run_date=month_back.replace(day=1)

    df.set_index('call_answer dt',inplace=True)
    df=df.loc [df.index <=month_back_run_date)

    #checking empty df
    Count-df.shape[]

    months=abs(diff_month(pd.to_datetime(parameters['sql_start_date']), month_back_run_date))+1
    total_rows=months*len(parameters['rule type list'])
    error msg =""
    try:
    #checking number of values in tnput df
        assert count == total_ rows, Input Data is insufficient
    except AssertionError as e:
        if not error_msg:
            error msg += e.args[0]
        else:
            error msg += "\n" + e.args[0]
    try:
    #check empty df
        assert count != 0, "No Data in input table"
      
    except AssertionError as e:
        if not error_msg:
            error msg + e.args[0
        else:
    error msg = e.args[0]

    try:
        #checking nulLs in df
        assert not df.isnull().values.any(), Input data contains null values 
          
    except AssertionError as e:
        if not error_msg:
            error_msg += e.args[0]
        else:
            error_msg += "\n" + e.args [0]
    if not error_msg:
        #assert True, "Input validation Success"
        print("Input validation Success")
    else:
    print (error_msg)
    print("Input validation Failure")
    logging.info("Input validation is completed.")

def train_forecast_models (forecast_input,
    demand_rule_type,train_start_date, train_end_date, test_start_date,test_end_date, params_ets, params_sarima,params_tes,params_grid_prophet,demand_model):
    '''a wrapper to train prophet forecasting model by rule_type and select & save the best hyper-parameters
    Args:
        forecast input (dataframe): time-series data of historical calls by rule type / date
        train_end date (date): train end date of whole time-series dataset
    Returns:
        best params dict (dict) best hyper parameter input per rule type based on past train
    '''
    best params_dict = {}
    returned list = Parallel(n jobs=-1)(delayed(train_forecast_model_rule_type)
                                        (selected rule,
                                         forecast_input, 
                                         train start_date,
                                         train_end_date, 
                                         test_start_date, 
                                         test_end_date, 
                                         params_ets,
                                         params_sarima,
                                         params_tes,
                                         oarams_grid_prophet,
                                         demand_model) for selected_rule in demand_rule_type)
                                                    
    for selected_rule, model_result_list in returned_list:
        if selected_rule not in best_params_dict:
            best_params_dict[selected_rule] = ()
        best_params_dict[selected_rule] = model_result_list
    return best_params_dict

def train_forecast_model_rule_type(selected_rule_type, forecast_input,train_start_date, train_end_date, test_start_date, test_end_date, params_ets,params_sarima, params_tes, params_grid_prophet,demand_model) :
    """
    train the model by rule type, and select the
    best hyper parameters
    Args:
    selected_rule_type: selected rule type
    forecast input (dataframe): time- series data of historical demand
    by rule type
    parameters (dictionary)
    Returns:
    selected_ruletype: selected rule type
    best_params (dict): best hyper parameter input
    """
    df_temp=forecast_select_data (forecast_input, selected_rule_type)
    model_result_list= Parallel(n_jobs=1) (
                       delayed(fitted_model_return)(
                        train_start_date, train_end_date, test_start_date, test_end_date, params_ets, params_sarima, params_tes, params_grid_prophet)
              for selected_model in demand_model)
    
    return selected_rule_type, model_result_list
    
def fitted model_return(model_selected, forecast_input,train_start_date, train_end_date, test_start_date, test_end_date, params_ets,params_sarima,params_sarima,params_tes,params_grid_prophet):
    '''
    Perform model training , hyper parmeter tuning , provide best paramters
    Input :
            model name: string,
            data : dataframe,
            parameters : dictionary
    Output:
            model name, best trained model object, best paramters
    '''
    train_start_date=pd.to_datetime(train_start_date)
    train_end_date=pd.to_datetime(train_end_date)
    test_start_date=pd.to_datetime (test_start_date)
    test_end_date=pd.to_datetime(test_end_date)
    data_train=forecast_input.loc[ (forecast_input.index >= train_start_date) & (forecast_input.index <= train_end_date)]
    data_test=forecast_input.loc[(forecast_input.index >= test_start_date) & (forecast_input.index <= test_end_date)]

    if model1_selected=='ETS':
        param grid_ets-params_ets
        best_model_ets, best params_ets=tune_model_ets (data_train, data_test, param_grid_ets)
        return 'ETS',best_model_ets, best_ params_ets,
        
    elif model_selected=='SARIMAX':
        param_grid_sarima=params_sarima
        best_model_sarima, best_params_sarima=tune_model_sarima(data_train, data_test, param_grid_sarima)
        return 'SARIMA', best_model_sarima, best_params_sarima
   


        
        
        
        
        
        
        
              
            
            
            
            
            
            




