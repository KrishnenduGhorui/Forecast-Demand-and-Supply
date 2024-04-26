
 **Time series forecastibility check** – 

**White noise** - Time series is not a white noise.

There is auto-correlation in each time series data , checked by **ACF/PACF** plot.
Time series are not white noise , also checked by **Ljung Box hypothesis test**

Conclusion - All Time series are forecastable 



**Decomposition** – 

Contribution of trend and contribution of seasonality is checked by decomposing the time series. So, Seasonality term is also important for forecasting target variable. 

Conclusion - So, SARIMA would be better than ARIMA algorithm . 

 
**Time series stationarity check by ADF test** – 

ADF test is done to check if original time series is stationary or not. If not then differencing is required to make it stationary to use that in Sarima model. 

**Other EDA prospect** – 

* Total Call volume monthly, Quarterly 
* Average Call volume on weekday and weekend month wise – weekend has very less call compared to weekday 
* Call volume trend on holiday vs non-holiday

**Conclusion /information retrieved from EDA** – 

Shift optimization - 

* The time range when most call is raised to call centre / what is peak time when maximum call raised 
* The time range in a day for which how much agent should be present 
* The time range when no call or very less call comes / When no need of any agent 
  

