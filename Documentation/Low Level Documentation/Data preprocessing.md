﻿**Data pre-processing -** 

Regarding pre-processing, 

Missing data was there in time series data- two type of missing data, 

·       One like date was missing ,like after 2022,6 Jan , 8 Jan there , 7<sup>th</sup> Jan is missing , date is not available in dataset.

·       Date is available but data like volume is nan. 

So, here as it is TS data, we cannot directly remove data record which are nan, that approach is wrong . As TS should be consecutive. 

We got to know from business like for those case where date is missing or data value is nan, that means those days call volume was 0, that why system store data on those days in that way. 

Similarly for intraday data, in each 30 mins interval , where call volume 0 , was missing, that interval inteself missing in data set. 

So, ideally we should fill up missing volume by 0. But we used MAPE(Mean Absolute Percentage Error) as evaluation metrics. And in calculation of MAPE , in the denominator ,  actual value is there, so if actual value 0 is in denominator, then mape calculation will be wrong , it will be huge number. So, instead of 0 , filled missing volume by 1. 

<a name="m_-8844058215201302453__hlk162478035"></a>Same done with intraday 30 minutes interval volume. 

And missing for AHT, OCC,SHR by average value of that rule type of that month.


