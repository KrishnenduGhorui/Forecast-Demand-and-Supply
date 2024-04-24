# Forecast-Demand-and-Supply

### Objective –
Develop an AI system that will predict required manpower in support call center for future at monthly, daily & intraday level by TimeSeries forecasting of Demand and supply. 

* Demand - Call volume
* Supply – AHT (Average Handling Time), OCC (Occupancy), SHR(Shrinkage)
* Calculated metric – Head count of support agent, SLA (Service Level Agreement)
  

### Impact -
* This helps call centre management take decision to employe nough human resource to handle all call efficiently with very less waiting time for customer. This will improve customer experience.
* It helps to cut the extra man resource that is not required as per the forecasted count of call volume. So, cost saving can be achieved. 
* By implementing this project, in overall call support team,
   * average **NPS** improved improved to **87** from **69**
   * **18% reduction in expenditure** on human resource per quarter
   * **21%** reduction in escalation to senior superviser due to long waiting time.

### Steps -

1. Load data from multiple BigQuery tables running a file containing aggregated sql query to join them.
2. Perform EDA from various business perspective and performed data pre-processing like missing value handling,outlier handling. 
3. Perform hyper parameter tuning and trained model algorithm like **FBProphet, SARIMA, ETS, Triple ExponentialSmoothing,LSTM, Stacking** with training data.
4. Evaluate performance of each model algorithm with validation data based on validation metric **MAPE** and pickup best model and corresponding best parameters value for each attribute ineach call type. Average model **accuracy** is **88%**. 
5. Train again the selected best model with whole available data (train + validation data) with corresponding bestpicked parameters value and use that model for future forecasting. 
6. 
  * Develop a pipeline to automate whole process (step 1-5, except EDA part) by Kedro framework on GCP Domino platform,
  * Make processing paralyzed for each call type & each metricto reduce run time
  * Make whole code parameterized using yaml file and deployed on Google cloud platform.


![image](https://github.com/KrishnenduGhorui/Forecast-Demand-and-Supply/assets/77465776/fccf8107-6a92-4bc4-9a0e-a6ea9607666c)


### Term & definition - 
* AHT (Average Handling Time) – On average how much time a single support agent takes to handle a call. 
* OCC (Occupancy) - % of support agent available inwork out of whole support agent count.

                   => (Count ofsupport agent active in work/Count of total support agent)*100
* SHR (Shrinkage) - % of inactive work time for asupport agent
* SLA (Service Level Agreement ) - % of call attended within30 seconds

                                 => (Count of call attended within 30 seconds/count of total call offered)*100
* Call_type/rule_type = call related to various business segment/department So , in data call_type/rule_type is available.
* Headcount – Count of support agent required to handle forecasted call efficiently with minimised waiting period.

                 HC monthly => (Call Volume * AHT)/3600/OCC/(1-SHR)
              
                 HC daily = (Call Volume * AHT)/3600/OCC/(1-SHR)/Days in Month
              
                 HC intraday = (Call Volume * AHT)/3600/OCC/(1-SHR)/Days in Month


![image](https://github.com/KrishnenduGhorui/Forecast-Demand-and-Supply/assets/77465776/83bbecd4-97fc-4964-93a5-7d41a7e40a7e)

There are 5 different types of call, so time series forecasting for each 5 groups for each attribute (call volume, AHT, OCC, SHR) individually are done. 
So, total 5*4=20 models there to forecast. 
Now for each attribute 3-4 model algorithms are trained to 


**Forecasting period** – 
Monthly – future 18 months out 

Daily – future 60 days out

Intraday – each 30 mins interval in a day , future 60 days out   


### Some special feature of this project are as below –  

1. **Parameterised code** –  Yaml file is used to parameterised whole code, so that we can control the processing from outside and lessen the hardcoding. Like the table name where forecasting output, other intermediate metrics are stored ,combination of parameters of algorithm , forecasting period , train, test data split ratio. All these are parameter are changeable just from the yaml file.  

2. **Automate process by pipeline** – Whole process is automatically carried out in pipeline by **kedro framework**. Data loading from bigquery table, then some preprocessing, model training, evaluation, picking up best model for each attribute for each rule-type and forecasting. 

3. **Real time model training** – Model is always trained with data up to recent date and recent data pattern will be reflected in forecasting.Because of this there is less chance the model to be obsolete, model is always up to date.  

4. **Parallel processing** – we code for parallel processing using **joblib library, Paralleland delayed method**. For each rule type 4s models for metrics (Call Volume, AHT, OCC, SHR)will be trained in parallel. Now for each rule type training is done parallelly. So, total 5 call type there, for each call type, 4 attributes there.Total 20 model being trained in parallel. 
