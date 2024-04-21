# Forecast-Demand-and-Supply

**Objective** –
Develop an AIsystem that will predict required manpower in support call center forfuture at monthly, daily & intraday level by TimeSeries forecasting ofDemand and supply. 

* Demand - Call volume
* Supply – AHT (Average Handling Time), OCC (Occupancy), SHR(Shrinkage)
* Calculated metric – Head count of support agent, SLA (Service Level Agreement)
There are 5 different types of call, so time series forecasting for each 5 groups for each attribute (call volume, AHT, OCC, SHR) individually are done.  

**Impact** -
* This helps call centre management take decision to employenough human resource to handle all call efficiently with very less waitingtime for customer. This will improve customer experience.
* It helps to cut the extra man resource that is notrequired as per the forecasted count of call volume. So, cost saving can be achieved. 
* By implementing this project, average NPS improved from87 (earlier was at 69) and 18% reduction in expenditure on humanresource per quarter.

**Steps** -

* Load data from multiple BigQuerytables running a file containing aggregated sql query to join them. * Perform EDA from various businessperspective and performed data pre-processing like missing value handling,outlier handling, aggregated daily data to monthly (for monthlyforecasting)
* Perform hyper parameter tuning andtrained model algorithm like FBProphet, SARIMA, ETS, Triple ExponentialSmoothing, Stacking with training data.
* Evaluate performance of each modelalgorithm with validation data based on validation metric MAPE and pickup best model and corresponding best parameters value for each attribute ineach call type. Average model accuracy is 88%. 
* Train again the selected best modelwith whole available data (train + validation data) with corresponding bestpicked parameters value and use that model for future forecasting. 
*
  * Develop a pipeline to automatewhole process (step 1-5, except EDA part) by Kedro framework on GCPDomino platform,
  * Make processing paralyzed for each call type & each metricto reduce run time
  * Make whole code parameterized using yaml file anddeployed on Google cloud platform.

 
Term & definition - 
* AHT (Average Handling Time) – On average how muchtime a single support agent takes to handle a call. 
* OCC (Occupancy) - % of support agent available inwork out of whole support agent count.= (Count ofsupport agent active in work/Count of total support agent)*100
* SHR (Shrinkage) - % of inactive work time for asupport agent
* SLA (Service Level Agreement ) - % of call attended within30 seconds
                           = (Count of call attended within 30 seconds/count of total call offered)*100
* Call_type/rule_type = call related to various business segment/department So , in data call_type/rule_type is available.
* Headcount – Count of support agent required to handle forecasted callefficiently with minimised waiting period.
   Formula to calculate headcount – HC monthly = (Call Volume * AHT)/3600/OCC/(1-SHR) HC daily = (Call Volume * AHT)/3600/OCC/(1-SHR) /Days inMonth HC intraday = (Call Volume * AHT)/3600/OCC/(1-SHR) /Days inMonth


**Forecasting period** – 
Monthly – future 18 months out 
Daily – future 60 days outIntraday – each 30 mins interval in a day , future 60 days out   

**Some special feature of this project are as below**–  Details - 
1.      Parameterised code – Yaml file we used to parameterised wholecode, so that we can control the processing from outside and lessen the hardcoding. Like the table name where forecastingoutput, other intermediate metrics are stored ,combination of parameters ofalgorithm , forecasting period , train, test data split ratioAll these are parameter are changeable justfrom the yaml file.  2.      Automatic process by pipeline – Whole process is automatically carried out in pipeline by kedroframework. Data loading from bigquery table, then some preprocessing, modeltraining, evaluation, picking up best model for each attribute for eachrule-type and forecasting. 3.      Real time model training –Model is always trained with data up torecent date and recent data pattern will be reflected in forecasting.Because of this there is less chance themodel to be obsolete, model is always up to date.  4.       Parallel processing – we code for parallel processing using joblib library, Paralleland delayed method. For each rule type 4s models for metrics (Call Volume, AHT, OCC, SHR)will be trained in parallel. Now for each rule type training is done parallelly. So, total 5 call type there, for each call type, 4 attribute there.Total 20 model being trained in parallel. 
