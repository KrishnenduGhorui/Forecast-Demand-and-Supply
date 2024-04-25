Model algorithm explored– 

* FBProphet

* Sarimax

* ETS 

* TES (Triple Exponential smoothing), DES (Double Exponential smoothing)

* Stacking 

* Regression 



Stacking – 

XGBoost regression is used as model algorithm.

Take prediction of all other model as attributes.

Predicting features – 

·       lag\_1, lag\_2, lag\_3…, lag\_6,

·       moving average (window 3 i.e mean of lag1, lag2, lag3)

·       prediction of other model (ETS, TES, sarimax)             



Now, here in stacking , used XGBoost regression as forecasting model . And attributes are forecasting of model sarimax, fbprophet, ets and lag1,lag2,lag3,lag4,lag5,lag6. This is for daily. 

From PACF plot of those Time Series, have seen upto 6 lag significant correlation is there. 

And for seasonality lag 7, lag14 also taken as attributes. As for daily data, weekly seasonality is there.

Taking lag7 ,lag14 make sense . 

for example the volume on today suppose Monday is mostly similar to Monday of last week. So, most dependency there on that day. 



filled missing value created due to taking lag, by lag value of same year. For example data there from 2019 Jan, for due to taking lag , lag1 value for 2019, jan should be 2018 ,Dec, but no data there, so lag1 value is 2019 December. 





**Regression** based (XGBoost)– 

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

Stacking , XGBoost is supported by GridSearchCV. So, tuning done through GridSerchCV. 

