**Model performance evaluation** - 

Then did evaluation of model performance for validation period data. 

And evaluation is done based on evaluation metrics MAPE. So, MAPE we used because like if MAPE value 10% , then it means 10% deviation is there from actual. 

And here we evaluating for various attributes. Like volume for call\_type1, call\_type2, call\_type3..

So, I need to get for volume of call\_type1 which model is best, similarly volume for call\_type2 which model is best, similar way for other call\_type. Now, here scale/range of volume of various call\_type is different. For example , call type, monthly volume in 30-40 lac range .for call\_type\_2 3-4 lac  , call\_type\_3 in 30-40 thousand. 

So, if we used mse or mae or rmse , then I couldn’t get idea about how much the model good is with respect to other rule type. 

In this case I can get idea like for example for rule\_type 1, MAPE is 0.18 , so 18% deviation there, for rule type\_2 volume MAPE is 16% , so overally both are of same performance. 

If used mase/mae/rmse then we get metric value in different range as data value volume is in different range . 

And if we AHT , that is in second , so like 1000 second in that range . 

So, by using MAPE as a metric we can compare all model how they are. 

But for intraday volume forecasting only couldn’t use MAPE as metrics because , intraday , in many 30 mins interval , call volume is 1,2 . So, suppose that actual is 1, and it did forecasted 3 , then MAPE value will 2, 200% , 200% mape value is considered as bad model , but in reality we can’t say model is bad where where actual is 1 and forecast is 3. So , in this case only used rmse as evaluation metrics as call volume in small value. 





Along with mape value , for evaluation, also checked lineplot of actual vs forecast on testing period and checked how close they are.

This is very important. Because for one model it came like forecasting constant ,but mape value is not so high. So, from mape value it would be concluded like model forecast is good. But actually model didn’t learn any trend, seasonal pattern. So, this plot is also necessary to watch. s



