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

def select_model_forecast(metric,model_mape_dict, sCore model_id, run_key, output_project, output_db name, output_metrics_tb_name,run_date):
    '''
    Selecting best nodel for each group(rule type) that is having least mape
    args :
    returns :
            rule_type_model_dict (dictionary) : Contain group name (rule type) as key, best model algo name for corresponding group(rule type) as value
    '''
    output metrics_dict = collections.defaultdict (dict)
    rule_type_model_dict = collections.defaultdict (dict)
    for rule_type in model_mape_dict:
        sub_dict-model_ mape_dict [rule_type]
        best_model=min(sub_dict, key-sub_dict.-get)
        Shich mOdel algorithm to use as per minimum mape per rule type
        #sub dict has group name and best model has which model to use
        rule_type_modeldict[rule_type]=best_model
        output_metrics_dict[rule_type][best_model]=sub_dict[best_model]
    cols = ['queue', 'metric_value','model']
    lst1=[]
    for key in output_metrics_dict:
        for k, v in output_metrics_dict[key].items():
            value=float(v)
            model=k
        lstl.append([key, round (value, 3), model])
    output_metrics_df=pd. DataFrame (1st1, columns cols)
    output_metrics_df['forecast_date']=datetime.now().date()
    output_metrics_df['metric']=metric
    output_metrics_df['metric_name']='wmape'
    output_metrics_df['rpt_dt' ]=last_day_month(run_date)
    output_metrics_df['score_model_id']=score_model_id
    output_ metrics_df[' run_key']=run_key
    output_metrics_df-output_metrics_df[['forecast_date', 'queue','metric', 'metric_name', 'metric_value', 'model','rpt_dt', 'score_model_id','run_key']]
    #write output to bq
    project_id=outputproject
    db_name=output_db_name
    tb_name=output_metrics_tb_name
    table_string=db_name+'.'+tb_name
    output_metrics_df.to_gbq (table_string, project_id, if_exists='append' )
    return rule_type_model1_dict
    
def forecasting_model_selection(metric,level, returned_list, best_params_dict, rule _type_model_dict, forecast_input, train_start_date, train_end_date, test_start_date, test_end_date, forecast_period,demand_model, num_lags, score_model_id, run key, run_date) :
    '''
    Perform retraining of the model algo that is selected as best for à rule type, retraining done on shole avilable data, provides that trained model
    Input :
            model name: string,
            data : dataframe,
            parameters' : dictionary
    Output : model name, best trained model object
    '''
    logging.info(f'Received best parameters - {best_params_dict}')
    output_df=pd.DataFrame()
    train_start_date=pd.to_datetime(train_start_date)
    train_end_date=pd.to_datetime(train_end_date)
    test_start_date=pd.to_datetime(test_start_date)
    test_end_date=pd.to_datetime(test_end_date)
    for key in rule_type_model_dict:
        df_tempforecast_select_data(forecast_input, key)
        data_train-df_temp. loc[ (df_temp. index >- train_start_date) & (df_temp . index <- test_end_date) ]
        if(level-=' monthly'):
            forecast_start_date = pd.to_datetime(test_end_date)+pd. Date0ffset (months=1)
            forecast_end_date = pd.to_datetime (test_end_date) +pd.Date0ffset (months=forecast_period)
        elif(level==' daily' ):
            forecast_start_date = (max(data_train.index)+pd.Date0ffset (days=1)).date()
            forecast_end_date = (max (data_train.index) +pd.DateOffset (days=forecast_period)) .date()
    
        my_dict = (tup[0]: tup[2][' param_best'] for tup in best_params_dict [key]}
        if rule_type_model_dict[kęy]=='ETS':
            param_grid_ets-my_dict['ETS']
            best model_ ets=train model_ets (data_ train, param grid_ets)
            forecast-best model ets.forecast(forecast period)
            forecast-round(forecast)
            df=pd.DataFrame(forecast)
            calc_std=round(forecast.stdO)
            df.rename(columns={df.columns[0]:'predicted_mean'},inplace=True)
            df['lb']=df['predicted mean']-calc std
            df['ub']=df['predicted mean']+calc std
            df.index=pd.to_datetime (df.index)
        elif rule_type_model_dict[key]=='SARIMA' :
            param grid_sarima=my dict['SARIMA' ]
            best model sarima-train_model sarimax(data train, param grid_sarima)
            forecast-best model sarima.forecast(forecast period)
            forecast-round(forecast)
            df-pd.DataFrame (forecast)
            calc_std=round(forecast.std())
            df.rename(columns-{df.columns[O]: 'predicted_mean'}, inplace=True)
            df['1b']=df['predicted_mean']-calc_std
            df['ub' ]=df['predicted_mean']+calc_std
            df.index=pd.to_datetime (df.index)
            
        elif rule_type_model_dict[key]=='TES':
            param_grid_tes=my_dict['TES]
            best_model_tes-train_model_tes (data_train, param grid_tes)
            forecast-best_model_tes.forecast(forecast_period)
            df.index=pd.to_datetime(df.index)
        elif rule_type_model _dict[key ]=-PROPHET' :
            param grid prophet=my_dict['PROPHET']
            best_model_prophet=train_model_prophet(data_train, param_grid_prophet)
            if(level=='monthly' ):
                forecast=best_model_prophet.predict(pd.DataFrame(pd.date_range (start=forecast_start_date, end=forecast_end_date, freq="MS"),columns=['ds']))[['ds', 'yhat','yhat_lower','yhat_upper']]
            elif(level=='daily'):
                forecast-best_model_prophet. predict (pd. DataFrame (pd.date_range(start=forecast_start_date, end=forecast end_date, freq='D'), columns=['ds']))['ds', 'yhat','yhat_lower','yhat_upper']]
                forecast=round(forecast)
                df=pd.DataFrame(forecast)
                df.set_index('ds',inplace=True)

        elif rule_type_model_dict[key]=='STACK":
            for i in retunned_ list:
                if(i[0] ==key) :
                    param_grid_stack=i[2]
                    best_model_stack=train_model_stack(i[3],i[5],i[4],i[6],param_grid_stack)
                    for j in range(len(demand_model) ) :
                        if(best_params_dict[key][j][0])=='ETS':
                            ets_fitted=best_params_dict[key][j][1]
                        if (best_params_ dict [key][j][O])=='SARIMA":
                            sarima_fitted=best_params_dict[key][j][1]
                        if (best params_dict [key] [j][O])==PROPHET':
                            prophet_fitted=best_params_dict [key][j][1]
                        
        forecast=forecast_stack(level, i[4],i[6],num_lags, ets_fitted, sarima_fitted, prophet_fitted, best_model_stack, forecast_period)
                    forecast=round(forecast)
                    df-pd.DataFrame(forecast)
                    calc_std=round (forecast.std() )
                    df.rename (columns={df.columns [0] : 'predicted mean'}, inplace-True)
                    df['lb' ]=df['predicted_mean']-calc_std
                    df['ub']=df['predicted_mean']+calc_std

        if(level=='daily'):
            df=handle_volume_holiday(data_train,df)
        df.rename(columns={df.columns[0]: metric}, inplace=True)
        df.rename(columns-{df.columns[1]: metric+'_lower_bound'}, inplace=True)
        df.rename(columns=(df.columns[2]: metric+'_upper_bound'}, inplace=True)
        df[metric+'_lower_bound'] = df[metriç+'_lower_bound' ].round ()
        df[metric+'_upper_bound'] = df[metric+'_upper_bound'].round()

        df[metric][df[metric] <= 0] = np.nan
        
        category_mean=df [metric]. mean()
        df[metric].fillna (category_mean, inplace=True)
        df[metric] = df[metric] .round()
        df[metric+'lower_ bound'] = df [metric+'_lower bound'].clip(lowe=0)
        df[metric+'_upper_bound'] = df[metric+'_upper_bound'].clip(l0wer=0)
        df[forecast_ date'] = df.index
        df.reset_index(drop=True)
        df['rule_type']=key

        output_df-pd.concat( [output_df, df], axis-0)
        #outputdfl'as_of_date']=datetime. now().date()
        output_df['rpt_dt']=last_day_month(run_date)
        output_ df['score_modelid']=score_mode1_id
        output_df[' run_key']=run_key
        output_df-output df[['forecast_date' ,'rule_type', metric, metric+'_lower bound', metric+'_upper_bound', 'rpt_dt','score_model_id', 'run_key' ]
    
    return output_df

def stack_model (level, forecast_input, best_models, demand_rule_type, train_start_date,train_end_date, test_start_date, test_end_date, num lags, params_stack):
    returned list = Parallel(n_jobs=-1)(
                    delayed(tune_model_stack) 
                    (level,forecast_input,selected_rule_type, best_models,train_start_date, train_end_date,test_start_date,test_end date, num lags, params_stack)
                    for selected rule_type in. demand_rule_type)

    return returned_list

def holdout_metrics_without_stack(metric, forecast_input, best_params_dict, test_start_date,test_end date,demand_model,score_model id, run key,output_project, output_db_name, output_metrics_inter_tb_name, run_date):
    '''
    Getting mape metrics válue for trained models
    Args:
        forecast_input (dataframe): timę-series data of historical calls by rule _type / date
        best_params_dict (dictionary) :
    Returns:
        Dictignary with mape metrics value of all models
    '''
    output_df = pd.DataFrame()
    test_start_date=pd.to_datetime(test_start_date)
    test_end_date=pd.to_datetime(test_end_date)
    model mape_dict = collections.defaultdict(dict)
    for key in best_params_dict:
        df_temp = forecast_select_data(forecast_input, key)
        df_temp=df_temp.loc[ (df_temp.index >= test_start_date) & (df_temp. index <= test_end_date) ]
        actual=round (df_temp['value'], 2)
        for i in range(len (demand_model)):
            if (best_params_dict[key] [i][0]) =='ETS' :
                forecast-best_params_dict [key][i][1].forecast (actual.shape [0])
            if(best params_dict[key][i][0])=='SARIMA':
                forecast-best_params_dict [key] [ij[1].forecast (actual.shape[0])
            if (best _params_dict [key][i][0])=='TES':
                forecast-best_params_dict [key] [ij [1].forecast (actual.shape [o])
            if(best_params_dict [key][i][0]) =='PROPHET':
                test_df=pd.DataFrame(df_temp.index).rename(columns={'call_answer_dt': 'ds'})
                forecast=best_params_dict [key][ij[1].predict(test_df)['yhat']
            forecast=pd.Series (round (forecast ))
            forecast.rename ("forecast", inplace=True)
            mape = Wmape(forecast, actual)
            model_mape_dict [key] [best_params_dict[key][i][O]=mape
            
            ind=pd.DataFrame(index=actual.index)
            ind-ind.reset_index()
            df=pd.concat([ind.reset_index(drop=True), actual.reset_index (drop-True), forecast.reset_index(drop-True) 1, axis=1)
            df['wmape']=round (mape, 2)
            df['rule_type']=key
            df['metric']=metric
            output_df=pd.concat([output_df, df], axis=0)
    outputdf = output_df.rename(columns-'value': 'actual'})
    output_df=output_df.groupby(['call_answer_dt','rule_type', 'metric', 'actual'])["wmape", 'forecast'].min()
    output_df=output_df.reset_index()
    output_df=output_df.sort_values (by='rule_type' )
    output_df-output df.reset_ index()
    output_df-output_df.sort_valuęs(by='rule_type')
    output_df['score_model_id']=score_model_id
    output_ df['runkey']=run_key
    output_df['rpt_dt']=last_day_month(run_date)
    output_df=output_df['call_answer_dt', 'rule_type','metric', 'actual','forecast ', 'wmape','score_model_id', 'run_key", "rpt_dt']
    output_df['actual'] = output_df['actual1']. astype (float)
    output_df['forecast'] = output_df['forecast'].astype(float)
    db_name=output_db_name
    tb_name=output_metrics_inter_tb_name
    table_string-db_name+'.'+tb_name
    project_id=output project
    output_df.to_gbq(table_string, project_id,if_exists='append')

    return model_mape_dict

def forecasting_model_selection_without_stack (metric,level, best_params_dict, rule_type_model_dict, forecast_input,train_start_date, train_end_date,test_start_date, test_end date, forecast_period,demand_model, num_lags,score_model_id, run key, run dáte):
    '''
    Perform retraining of the model algo that is selected as best for a rule type, retraining done on whole avilable data, provides that trained model
    Input :
            model name: string,
            data : dataframe,
            parameters : dictionary
    Output : model name, best trained model object
    '''
    output_df = pd.DataFr ame()
    train_start_date=pd.to_datetime (train_start_date)
    train_end_date-pd.to_datetime(train_end_date)
    test_start date=pd,to_datetime(test _start_date)
    test_end_datepd.to_datetime (test_end_date)
    
    for key in rule type model dict:
        df_temp=forecast_select_data(forecast_input, key)
        data_train=df_temp.loc[(df_temp.index >= train_start_date) & (df_temp.index <= test_end_date)]
    if(level=='monthly'):
        forecast _start_date = pd.to_datetime(test_end_date) +pd.Date0ffset (months=1)
        forecast_end_date = pd.to_datetime (test_end_date) +pd.Date0ffset (months-forecast_period)
    elif(level==daily'):
        forecast_start_date = (max(data_train.index) +pd. DateOffset (days=1) ) . date()
        forecast_end_date = (max (data_ train.index) +pd.Date0ffset (days=forecast_period) ).date()
    my_dict = (tup[O]: tup[2] ['param_best'] for tup in best_params_dict[key]}
    if rule_type_model_dict [key] =='ETS':
        param_grid_ets=my_dict['ETS']
        est_model_ets=train_model_ets (data_train, param_grid_ets)
        forecast=best_model_ets.forecast (forecast_period)
        forecast=round (forecast)
        df-pd.Data Frame (forecast )
        calc_std=round (forecast.std())
        df.rename (columns={df.columns[0]: 'predicted_mean'}, inplace=True)
        calc_std=round (forecast. std()
        df.rename (columns=(df, columns [0]: 'predicted_mean'},inplace=True)
        df['lb' ]-df[' predicted_ mean']-calc_std
        df['ub' ]-df['predicted_mean' ]+calc_std
        df.index=pd.to_datetime (df. index)
           
    elif rule_type_model_dict[key] =='SARIMA':
        param_grid_sarima=mydict['SARIMA']
        best_model_sarima=train_model_sarimax(data_train, param_grid_sarima)
        forecast=best_model_ sarima.forecast (forecast_period)
        forecast=round (forecast)
        df=pd.DataFrame (forecast)
        calc_std=round (forecast.std())
        df.rename (columns={df.columns[0]: 'predicted mean'},inplace-True)
        df['lb'=-df['predicted mean']-calc std
        df['ub']=df['predicted_mean']+calc_std
        df.index=pd.to_datetime(df. index)
    elif rule_type_model_dict [key ] =='TES':
        param_grid_tes=my_dict['TES']
        best_model_tes=train_model_tes (data_train, param_grid_tes)
        forecast=best model_tes.forecast (forecast_period)
        df.index=pd.to_datetime(df.index)
    elif rule_type_model_dict[key] =='PROPHET':
        param_grid_ prophet=my_dict["PROPHET"]
        param grid prophet-my_dict['PROPHET']
        best_model_prophet=train_model_prophet (data_train, param_grid_prophet)
        if(level=='monthly' ) :
            forecast-best_model_prophet.predict (pd.Dataframe (pd.date_range(start =forecast_start_date, end=forecast_end_date, freq='MS'),columns('ds']))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        elif(level=-'daily' ) :
            forecast=best_model_prophet. predict (pd.DataFrame (pd.date_range (start=forecast_start_date, end=forecast_end_date, freq='D'), columns["ds"]))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast=round ( forecast)
            df=pd.Dataframe (fore cast)
            df.set_index('ds' ,inplace=True)
    df.rename (columns={df.columns [O]: metric), inplace-True)
    df.rename (columns={df.columns [1]: metric+'_lower_bound'},inplace-True)
    df.rename (columns=(df. columns[2]: metric+'_upper_bound'}, inplace-True)
    df[metric+"_lower_bound"] = df[metric+'_lower_bound'].round ()
    df[metric+'_upper_bound'] = df [metric+'_upper_bound']. round()
    
    df[metric}[df[metric] <= 0] = np.nan
    categorY_me an-df[metric].mean()
    df[metric].fillna(category_mean, inplace=True)
    df[metric] = df[metric].round()
    df[metric+'_lower_bound'] = df[metric+'_lower_bound'].clip (lower =0)
    df[metric+'_upper_bound'] = df[metric+'_upper_bound'].clip (lower=0) 
    df[ forecast_date'] = df.index
    df.reset_index(drop=True)
    df['rule_type']=key
    output_df=pd.concat ([output_df, df], axis=0)
    output_df['rpt_dt' ]=last_day_month (run_date)
    output_df['score_model_id']=score_model_id
    output_df['run_key']=run_key
output_df-output_df[['forecast_date', 'rule_type', metric, metric+'_lower_bound ' , metric+" _upper_bound", 'rpt_dt', 'score_model_id',"run_key"]

return output_df

def output_validation(df, level, parameters):
    #'function to validate output for demand'    
    count=df.shape[]
    if(level='monthly'):
        months=parameters['monthly_forecast_period']
        total_rows=months*len(parameters['rule_type_list'])*3 #number of metrics
    error_msg=""    
    try:
    #checking number of values in input df
    assert count != 0,"No Data in output table"
    
    except AssertionError as e:
        if not error msg:
            error msg += e.args[O]
        else:
            error_ msg += "\n"+e.args[0]
    try:
    #checking number of values in input df
    assert not df.isnull().values.any(), 'Output table contains null values'
    except AssertionError as e:
        if not error msg:
            error_msg += e. args[0]
        else:
            error_msg += "\n"+e.args[0]
    try:
    #checking number of values in input df
    assert count == total_rows, 'Output table data insufficiency'
    
    except AssertionErrr as e:
        if not error msg:
            error msg +=e.args [O]
        else:
            erron msg+="n" +e.args 0]

    if not error_msg:
    # assert True, "Input validation Success"
        print( "Output validation Success"*)
    else:
        print(error_msg)
        print ("Output validation Failure")
    logging.info("Output validation is completed.")
    
def train_test_splitting (forecast_input,demand_rule_list, train_start_date,run_date, level, split ratio) :
    for i in demand_rule_list:
        df_forecast = forecast_input [forecast_input ["rule__type"] = i]
        break
    train_start_date=pd.to_datetime (train_start_date)
    run _date = pd.to_datetime (run_date)
    if(level=='monthly' ):
        month_back = run _date - relativedelta(months-1)
        test_end_date-month_back.replace(day-1)
    if(level=='daily'):
        test _end_date = run_date - timedelta(days=2)
    test_end_date-pd.to_datetime (test_end_date)
    df_forecast-df_forecast.loc[ (df_forecast.index => train_start_date) & (df_forecast.index <= test_end_date)]
    train, test = train_test_split (df_forecast, test_size=split_ratio, shuffle=False)
    
    train_end_date=train.index.max()
    test_end_date=test.index.max()
    
    return train_start_date,tarin_end_date,test_start_date,test_end_date
    
    
    

    






        
    
    
    
    
    
    

        








            


    
    
    
        
    
    
    
        
        


   


        
        
        
        
        
        
        
              
            
            
            
            
            
            




