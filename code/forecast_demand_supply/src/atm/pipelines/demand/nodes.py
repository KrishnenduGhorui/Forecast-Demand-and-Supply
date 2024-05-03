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

    elif model_selected=='TES' :
        param_grid_tes=params_tes
        best_model_tes, best_params_tes=param_tuning_tes(data_train, data_test, param_grid_tes)
        return 'TES' ,best_model_tes, best_params_tes
    elif model_selected=='PROPHET':
        param_grid_prophet=params_grid_prophet
        best_model_prophet, best_params_prophet=tune_model_prophet (data_train, data_test, param_grid_prophet)
        return 'PROPHET',best_model_prophet, best_params prophet
        
def outlier_treatment (datacolúmn):
    sorted (datacolumn)
    Q1,Q3 = np.percentile (datacolumn , [25,75])
    IOR = Q3- Q1
    lower_range = Q1- (1.5 * IOR)
    upper_range = Q3+ (1.5 IOR)
    return lower_range, upper_range
    
def wmape (forecast, actual):
    # we take two series and calculate an output a wmape from it
    forecast=forecast.reset index)
    actual=actual.reset_index()
    forecast = forecast.iloc[:,1]
    actual = actual.iloc[: ,1]
    
    #make a sertes called mape
    se_mape=abs(actual-forecast) /actual
    
    #get a float of the sum of the actual
    ft_actual_sum = actual.sum()
    
    print('se_mape',se_mape)
    print('ft_ actual sum',ft_actual_sum)
    
    # get a series of the multiple of the actual & the mape
    se_actual_prod_ mape = actual* se_mape
    # summate the prod of the actual and the mope
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()
    
    # float: wmape of forecast
    ft_wmape_forecast = ft_actual_prod_mape_sum/ft_actual_sum
    ft_wmape_forecast=round(ft_wmape_forecast,2)
    
    # return a float
    return ft_wmape_forecast
    
def forecast_select_data(forecast input, selected ruletype):
    """
    select the forecast. input data of specific negion programgroup for forecasting model training
    Args:
        forecast input (dataframe) : time-series data of historical calls by ruletype/date.
        selected region (str): selected region
        selected_program group (str): selected program group
    Returns:
        df_forecast (dataframe ): tine-series data of historical calls for selected rule_type
    """
    df_forecast = forecast_input [forecast_input["rule_type"] == selected_rule_type]
    lowerbound, upperbound = outlier_treatment (df_forecast.value)
    df_forecast['value'] =np.where(df_forecast['value']>upperbound,upperbound, np.where(df_forecast['value']<lowerbound, lowerbound,df_forecast['value']))

    return df_forecast
                                                                                        
def holdout_metrics (forecast_input, best_params_dict, returned_list,test_start_date,test_end_date, demand_mode1):
    '''
    Getting mape metrics value for trained models
    Args:
    forecast_ínput (dataframe): time-series data of historical calls
    by rule_type / date
    best_params dict (dictionary) :
    Returns:
    Dictionary with mape metrics value of all models
    '''
    test_start_date=pd.to_datetime(test_start_date)
    test_end_date=pd.to_datetime(test_end_date)

    model_mape_dict = collections.defaultdict(dict)
    for key in best_params_dict:
        df_temp = forecast_select_data(forecast_input, key)
        df_temp=df_temp. loct (df_temp.index >= test_start_date) & (df_temp. index <= test_end_date)]
        actual=df_temp ['value']
        for i in range(len (demand_model)):
            if(best_params_dict [keyj [i][O]) =='ETS':
                forecast-best_params_dict [key ] [i] [1].forecast (actual. shape [0])
            if(best _params_dict [key] [i][O]) ==' SARIMA' :
                forecast-best_params_dict [key ] [i] [1].forecast (actual.shape[0])
            if (best_params_dict [key] [iJ[O)==TES':
                forecast-best_params_dict[key][ij[1) .forecast (actual.shape[0])
            if(best_params_dict [key] [ij[o])=='PROPHET':
                test_df=pd.DataFrame (df_temp.index) .rename (columns-'call answer_dt':"ds'})
                forecast-best_params_dict [key] [i)[1).predict(test_df) ['yhat']
            
            mape = mean_absolute_percentage_error(forecast, actual)
            model_mape_dict [key ] [best_params_dict [key] [ij[e]]-mape
        for i in returned_list:
            if (i[e]==key):
                forecast=i[1]. prediçt(i[5] )
                mape = mean_absolute_percentage_error(forecast, actual)
                model_mape_dict [key]['STACK']=mape
    return model_mape_dict
    
    
    
    
        
        


   


        
        
        
        
        
        
        
              
            
            
            
            
            
            




