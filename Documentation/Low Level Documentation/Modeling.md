
 Model algorithm explored for Time Series forecasting – 

   * FBProphet
   * Sarimax  
   * ETS 
   * TES (Triple Exponential smoothing), DES (Double Exponential smoothing)
   * Stacking (XGBoost)
   * Regression based approach - Random Forest
   * LSTM

**Stacking** – 

 * Meta model - XGBoost regressor  
 * Base models - FBProphet, Sarimax, LSTM, ETS

Predicting features / attribute in stacking XGBoost model – 
   * prediction of model FBProphet, Sarimax, LSTM, ETS
   * lag_1, lag_2, lag_3,...,lag_6,
   * lag_7, lag_14 (for daily only)
   * moving average (window 3 i.e mean of lag1, lag2, lag3)


Daily - 
From PACF plot of those Time Series, have seen upto 6 lag significant correlation is there. 

And for seasonality lag 7, lag14 also taken as attributes. As for daily data, weekly seasonality is there.

Taking lag7 ,lag14 as attribute make sense. because for example the volume on today suppose Monday is mostly similar to Monday of last week. So, most dependency there on that day. 

Filled missing value created due to taking lag, by lag value of same year. For example data there from 2019 Jan, for due to taking lag , lag1 value for 2019, jan should be 2018 ,Dec, but no data there, so lag1 value is 2019 December. 



**Regression based forecasting approach (XGBoost & Random Forest regressor)**– 

Predicting variable/attributes are  – 

·       lag\_1(last month value), 

·       lag\_2(2<sup>nd</sup> last month value),

·       lag\_3(3rd last month value),

·       average of last 3 month value,

·       value of same month of last year 

·       value of same month of 2<sup>nd</sup> last year 





**Hyper parameter Tuning** - 

Grid SearchCV/ randomsearchcv doesn’t support ETS, TES,FBProphet model for hyper parameter tuning. 

So, to tune these models , manually model to be running on iteration using for loop on various parameters combination and select best parameters. 

And to tune sarima, autoarima is used.

On Stacking , XGBoost is supported by GridSearchCV. So, tuning done through GridSerchCV. 

