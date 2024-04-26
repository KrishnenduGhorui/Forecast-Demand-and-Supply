**Data pre-processing -**  

Missing data was there in time series data- two type of missing data, 

·       One like date was missing ,like after 2022,6 Jan , 8 Jan there , 7<sup>th</sup> Jan is missing , date is not available in dataset.

·       Date is available but data like volume is nan. 

So, here as it is TS data, we cannot directly remove data record which are nan, that approach is wrong . As TS data should be of consecutive date. 

So, now at first missing dates were found out, then added into time series dataset. 

As per business, for those case where date is missing or data value is nan, that means those days call volume was 0, that why system store data on those days in that way. 

Similarly for intraday data, in each 30 mins interval , where call volume 0 , was missing, that interval inteself missing in data set. 

So, ideally we should fill up missing volume by 0. But as MAPE(Mean Absolute Percentage Error) is used as evaluation metrics and in calculation of MAPE , denominator is actual value, so if actual value 0 is in denominator, then mape calculation will be wrong , it will be huge number. So, instead of 0 , filled missing volume by 1. 

<a name="m_-8844058215201302453__hlk162478035"></a>Same done with intraday 30 minutes interval volume. 

And missing data for AHT, OCC,SHR are imputed by average value of that rule type of that month.



