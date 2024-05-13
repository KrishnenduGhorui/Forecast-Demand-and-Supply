import pandas as pd
import numpy as np
from gocgle.cloud import bigquery
from datetime import datetime, timedelta
import math
import logging
from ..demand.model_train import store_data_bq_f

def last_day_month(run_date):
  date_obj = datetime.strptime (run_date, "Y - %m-%d" )
  first_day_next_month = date_obj.replace (day=1) + timedelta (days-32)
  last_day_current_month = first_day_next_month.replace (day-1) timedelta (days =1)
  date_str = last_day_current_month. strftime ( '%Y-%m-%d')
  rpt_dt = pd.to_datetime (date _str).date()
  return rpt_dt
  
def load_daily_forecast (df_daily_forecast):
  # remove timestamp from date
  df_daily_forecast.forecast date-pd.to_datetime (df_daily_forecast.forecast_date)
  df_daily_forecast.forecast_date=df_daily_forecast.forecast_date.dt.date.
  df_daily_forecast.rename (column s={'forecast_date':'call_answer_dt '},inplace-True)
  df_daily_forecast-df_daily_forecast[['call_answer_dt', 'rule_type', 'Volume', 'Volume_lower_bound', 'Volume_upper_bound', 'AHT','AHT_lower_bound','AHT upper_bound', 'OCC', 'OCC_lower_bound', 'OCC_upper_bound'])
  return df_daily_forecast

# Function to Fill missing interval
def handle_missing_interval(value_dist_wd, value_dist_wnd, value):
  interval_time_start=['00:00:00', '00:30:00', '01:00:00','01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00' , '04:00:00',
'04:30:00', '05:00:00', '05:30:00','06:00:00', '06:30:00','07:00:00', '07:30:00', '08:00:00', '08:30:00', '09:00:00', '09:30:00',
'10:00:00', '10:30:00', '11:00:00', '11:30:00','12:00:00','12:30:00', '13:00:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00' ,
'15:30:00', '16:00:00', '16:30:00', '17:00:00', '17:30:00', '18:00:00', '18:30:00', '19: 00:00', '19:30:00', '21:00:00','21: 30:00', '22:00:00', '22:30:00','23:00:00', '23:30:00']

  value_dist_wd.index=value_dist_wd.index.astype('str')
  value_dist_wnd.index=value_dist_wnd.index.astype('str)
  if(value=='Volume):
    mean_value_wd-0
    mean_value_wnd=0
  else:
    mean_value_wd=value_dist_wd. mean()
    mean_value_wnd=value_dist_wnd.mean()
  if (len(value_dist wd) <48):
    missing_iterval_wd= [x for x in interval_tine_start if x not in value_dist_wd. index. to_list () ]
    srs_missing _interval_wd=pd.Series (mean_value_wd, index-missing_interval_wd)
    value_dist_wd=value_dist_wd.append (srs_missing_interval_wd) .sort_index()
    #value_dist_wd=Value_dist_wd.sort_index()
  if (len(value_dist_wnd) <48):
    missing interval_wnd-(x for x in interval_time_start if x not in value_dist_wnd. index .to_list ()01
    srs_missing_interval_wnd-pd.Series (mean_value_wnd, index-missing interval_wnd)
    value_dist_wnd-value_dist_wnd.append(srs_missing_interval_wnd)
    value_dist_wnd=value_dist_wnd.sort_index()
  return value_dist_wd, value_dist_wnd


# function to calculate call distribution
def get_call_dist (date, df_intraday):
  '''
  Func - It will get call đistribution 30 mins interval wise for a day, distribution for weekday and weekend will be there
  individually.
  input :
  Date - date from the day forecasting is required
  df_intraday (dataframe) - Data of call volume 30 mins interval wise
  Output :
  callRatio_avg wd (Series) - call distribution for weekday
  callRatio_avg wnd (Series) - call distribution for weekend
  '''
  df_intraday['call_answer_dt' ]=pd.to_datetime(af_intraday[call_answer_dt'1)
  date_today=pd.to_datetime (date)
  # Get date of 90 days past
  date_start-date_today-pd. Date0ffset (days=90)
  df_intraday=df_intraday[ (df_intraday[' call _answer_dt']>=date_start) & (df_intraday['call_answer_dt']<date_today) ]
  df_intraday['weekday -weekend']=[ 'weekday' if wd<5 else 'weekend' for wd in df_intraday.call_answer_dt.dt.weekday]
  df_intraday_wd=df_intraday [df_intraday[ 'weekday-weekend' ]=='weekday']
  df_intraday_wnd=df_intraday [df_itraday ['weekday-weekend']=='weekend']
  # Get distribution of catl volume for weekday
  df_pvt_wd-df_intraday_wd.pivot
  (index='call_answer_dt',columns='interval_start',values='Volume)
  df_pvt_wd.fillna (0,inplace-True)
  df_callRatio_wd=df_pvt_wd.div(df_pvt_wd. sum(axis-1), axis=0)
  # Get distribution of call volume for weekend
  df_pvt_wnd=df_intraday_wnd . pivot (index="callanswen dt ', columns='interval_start', values-'Volume' )
  df_pvt_wnd. fillna (0, inplace=True)
  df_callRatio_wnd=df_pvt_wnd.div(af_pvt_wnd. sum(axis-1), axis-0)
  # Get average of call ratio for Last B months
  callRatio_avg wd=df_cal1Ratio_wd.mean()
  callRatio_avg_ wnd=df_cal1Ratio_wnd .mean()
  # FilLing missing interval
  callRatio_avg_wd, cal1Ratio _avg_wnd =handlemissing interval(cal1Ratio_avg_wd , callRatio_avg_ wnd, "Volume') ,
  return callRatio_avg_wd, call Ratio_avg wnd

# function to get AHT distribution
def get_aht_dist (date, df_intraday):
  '''
  Func - It will get AHT distribution 30 mins interval wise for a day , distribution for weekday and weekend will be there
  individually.
  input :
  Date - date from the day forecasting is required
  df_ intraday (dataframe) - Data of AHT 30 mins interval wise
  Output:
  aht_avg_wd (Series) - AHT distribution for weekday
  aht_avg_wnd (Series) - AHT distribution for weekend
  '''
  df_intraday['call_answer_dt']=pd.to_datetime (df_intraday['cąll_answer_dt'])
  date_today=pd.to_datetime (date)
  # Get date of 90 days past
  date_start-date_today-pd.DateOffset (days=90)
  df_intraday=df_intraday[ (df_intraday [' call_answer_dt']>date_start) & (df_intraday['call_answer_dt']<date_today)]
  df_intraday['weekday -weekend']=['weekday' if wd<5 else 'weekend' for wd in df_intraday.call_answer_dt.dt.weekday]
  df_intraday_wd=df_intraday [df_intraday[ 'weekday-weekend']=='weekday']
  df_intraday_wnd=df_intraday [df_intraday [ 'weekday- weekend']=='weekend']

  # Get distribution of call vol ume for weekday
  df_pvt_wd=df_intraday_wd. pivot (index='call_answer_dt', columns='interval_start',values='AHT')
  df_pvt_wd.fillna (0, inplace=True)
  # Get distribuution of call vol ume for weekend
  df_pvt_ wnd=df_intraday_wnd.pivot (index='call_an swer_dt', columns='interval_start',values="AHT)
  df_pvt_wnd. fillna (0, inplace=True)
                                     
  # Get average of call ratio for last 3 months
  aht_avg_Wd=df_pvt_wd.mean()
  aht_avg_Wnd=df_pvt_wnd. mean()
  aht_avg_Wd, aht_avg_wnd=handle_missing_interval (aht_avg_wd, aht_avg wnd, ' aht')
  return aht_avgWd, aht_avg_wnd
  
# function to get 0CC distribution
def get_occ_dist (date, df_intraday_wnd):
  '''
  Func - It will get 0CC distribution 30 mins interval wise for a day , distribution for weekday and weekend will be there individually.
  input :
        Date - date from the day forecasting is required
        df_intraday (dataframe) -Data of 0CC 30 mins interval wise
  Output :
        occ_avg wd (Series) - 0CC distribution for weekday
        occ_avg wnd (Series) - oCC distribution for weekend
  '''
  df_intraday_wnd['call_answer_dt']=pd.to_datetime (df_intraday_wnd['call_ansiwer_dt'])
  date_today=pd.to_datetime (date)
  # Get date of 90 days past
  date_start-date_today-pd. DateOffset (days=90)
  df_intraday_wnd=df_intraday_wnd [ (df_intraday_wnd['call _answer_dt' ]>=date_start) & (af_intraday_wnd [ ' cal1_answer dt']<date_today)]
  df_intraday_ wnd [ "weekday-weekend" ]-[ 'weekday' if wd<5 else 'weekend' for wd in df_intraday_wnd. call_answer_dt. dt . weekáay]
  df_intraday_ wnd_wd=df_intraday_wnd [df_intraday_wnd [ 'weekday-weekend ']== 'weekday']
  df_intraday_wnd_wnd=df_intraday_wnd[df_intraday_wnd['weekday -weekend']==' weekend']
  # Get distribution of calL volume for weekday
  df_pvtwd=df_intraday_wnd_ wd.pivot (index='call_answer_dt',columns='interval_start', values='0CC')
  df_pvt wd.fillna(0, inplace=True)
  # Get distribution of call volume for weekend
  df_pvt_wnd=df_intraday_wnd_wnd. pivot (index='call_answer_dt', columns='interval_start', values='0CC')
  df_pvt_wnd.fillna (0, inplace=True)
  # Get average of call ratio for Last 3 months
  occ_avg_wd=df_pvt_wd.mean()
  occ_avg_wnd=df_pvt_wnd.mean()
  oCc_avg_Wd, occ_avg_wnd=handle_missing_interval(occ_avg wd, occ_avg_wnd, 'occ')
  return occ_avg wd, occ_avg_wnd

# function to calculate value for next 60 days from distribution
def forecast_value (value_dist_wd, value_dist_wnd, value_daily, value):
  '''
  Func : It will forecast the value volume intraday wise from the distribution and forecasted daily value volume
  Input :
  value dist wd (Series) - value distribution intraday wise for weekday
  value_ dist wnd(Series) - value distribution intraday wise for weekend
  value daily - Forecasted value volume daily for future days
  Output :
  df_intraday_value (dataframe) - forecasted value 30 min interval wise for future day
  '''
  valve_daily_c=value_daily. copy()
  value_daily_c['call_answer_dt']=pd.to_datétime (value_daily_c['call_answer_dt'])
  value_daily_c.set_index('call_ answer_dt',inplace-True)
  df_intraday_value=pd.DataFrame([])
  for indx in list (value_daily_c.index):
    if indx.weekday() <=4:
      if(value=='Volume' or value='Volume_lower_bound' or value='Volume_upper_bound')
         if(value=="Volume" or value=='Volume lower_bound or value=-"Volume_upper_bound):
            row=value_daily_ciloc[indx, value]value_dist wd
        else:
            row=value_dist_wd
    else:
      if(value=='Volume' or value='Volume_lower bound' or value=="Volume_upper_bound"):
        row=value_daily_c.loc[indx, value]*value_dist_wnd
      else:
        row=value_dist_wnd
      
    df_intraday value-dfintradayvalue.append (row, ignore_index=True)
  
df_intaday_value.index=value_daily_c.index
return df_intraday_value

# function to calculate intraday Head count for next 60 days
def calc_hc_intraday(call_fore_intraday, aht_fore_intraday, occ_fore_intraday, shr=0.4):
  '''
  Func : Calculate headcount for 30 mins interval wise
  input :
      call fore intraday(dataframe) - forecasted call volume 30 mins interval wise for future day
      aht fore intraday(dataframe)- forecasted AHT 30 mins interval wise for future day
      occ_fore_intraday (dataframe)- forecasted occ 30 mins interval wise for future day
      shr - a fixed numver 0.4
  Output:
      df_hc(dataframe )- calculated head count 30 mins interval wise for future day
  '''
  df_hc=np. ceil((call_fore_intraday* aht_fore_intraday) /3600/occ_fore_intraday/ (1-shr))
  return df_hc

# function to get output dataframe
def get_output_result (call_forecasted_intraday, aht_forecasted_intraday, occ_forecasted_intraday, hc_forecasted_íntraday, rule_type, df_call_fore_
intraday_lb, df_aht_fore_intraday_lb, df_occ_fore_intraday_lb, df_call_fore_intraday_ub, df_aht_fore_intraday_ub, df_occ_fore_intraday_ub):
   '''
  Func :- Consolidate forecasted call volume, AHT, 0CC, head count 30 mins interval wise into one dataframe
  Input:
      Forecasted call volume intraday, Forecasted AHT intraday, Forecasted occ intraday, calculated head count intraday, rule
      type(string)
  Output:
      Dataframe with future date and corresponding call volume, AHT, OCC, Head count 30 mins interval wise
 '''
  # print(aht forecasted_intraday)
  df_call_intraday_fore-call_forecasted_intraday.stack().reset_index()
  df_call_intraday_fore.columns=['call_answer_dt', 'interval_start', 'Volume']
  df_call_intraday_fore['Volume']=df_call_intraday_fore['Volume'].apply (1ambda x: math.ceil(x))

  df_aht_intraday_fore-aht_forecasted _intraday. stack() .reset_index()
  df_aht_intraday_fore.columns-['call_answer_dt', 'interval_start', 'AHT']
  df_aht_intraday_fore['AHT']=df_aht_intraday_fore[ 'AHT' ].apply (lambda x: round (x, 2))
  std_aht_intraday-df_aht_intraday_fore['AHT'].std()
  
  


      






  


  


