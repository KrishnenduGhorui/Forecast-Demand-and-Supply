from typing import Dict
from kedro.pipeline import Pipeline
from atm.pipelines.demand.pipelinė import demand_data_pipeline, load_data_pipeline, input_validation_pipeline, output_validation_pipeline
from atm.pipelines.supply.pipeline import supply_data_pipeline, supply_data_pipeline, aggregator_pipeline, whatif_pipeline
from atm.pipelines.intraday. pipeline import intraday_pipeline,intraday_pipeline_whatif

def register_pipelines (**kwargs) -> Dict[str, Pipeline] :
    '''
    Create the project's pipeline.
    Args:
      kwargs: Ignore any additional arguments added in the future.
    Returns:
      A mapping from a pipeline name toaPipeline object.
    '''
    data_pipeline_daily = load_data_pipeline(level='daily')
    data_pipeline_monthly = load_data_pipeline(level-'monthly')
    data_pipeline_intraday = load_data_pipeline(level='intraday')
  
    monthly_input_validation = input_validation_pipeline(level='monthly')
    monthly_output_validation=output_validation_pipeline(level='monthly')
  
    Volume_pipeline_daily = demand_data_pipeline(stack_flag-0, suffix='Volume',level='daily')
    Volume_pipeline_monthly = demand_data_pipeline (stack_flag-0, suffix='Volume',level='monthly')
  
    calls_answered_within_30_pipeline_daily = denand_data_pipeline (stack_flag=0, suffix='calls_answered_within_30', level='daily')
    
    aht_pipeline_daily = supply_data_pipeline(suffix='aht', stack_flag=0, level='daily')
    aht_pipeline_monthly = supply_data_pipeline(suffix='aht', stack_flag=0, level='monthly')

    occ_pipeline_daily = supply_data_pipeline(suffix='occ', stack_flag=0, level='daily')
    occ_pipeline_monthly = supply_data_pipeline(suffix='occ', stack_flag=0, level='monthly')

    aggregator_pipeline_daily=aggregator_pipeline(level='daily')
    aggregator_pipeline_monthly=aggregator_pipeline(level='monthly')

    intraday_pipe=intraday_pipeline()

    whatif_pipeline_daily=whatif_pipeline(level='daily')
    whatif_pipeline_monthly=whatif_pipeline(level='monthly')
    whatif_pipeline_intraday=intraday_pipeline_whatif()

    return {
    #these pipelines won't write data in tables
    "demand_pipeline_daily": data_pipeline_daily+Volume_pipeline_daily,
    "demand_pipeline_monthly": data_pipeline_monthly+Volume_pipeline_monthly,
    "calls_answeredwithin_30_pipeline_daily": data_pipeline_daily+calls_answered_within_30_pipeline_daily,
    "supply_aht_pipeline_daily" : data_pipeline_daily+aht_pipeline_daily,
    "supply_aht_pipeline_monthly" : data_pipeline_monthly+aht_pipeline_monthly,
    "supply_occ _pipeline_daily": data_pipeline_daily+occ_pipeline_daily,
    "supply_occ_pipeline_monthly": data_pipeline_monthly+occ_pipeline_monthly,
    "dd": data _pipeline_intraday,
    # these pipelines will write data, into tables
      
    # Only for daily
    "daily_pipeline":
                     data pipeline_daily
                   + Volume_pipeline_daily
                   + calls_answered_within_30_pipeline_daíly
                   + aht_pipeline_daily
                   + occ_pipeline_daily
                   + aggregator_pipeline_daily
                   + whatif_pipeline_daily,
      
    #for daily + intraday
    "daily_intraday_pipeline":
                     data_pipeline_daily
                   + Volume_pipeline_daily
                   + calls_answered_within_30_pipęline_daily
                   + aht_ pipeline_daily
                   + occ_pipeline_daily
                   + aggregator_pipeline_daily
                   + data_pipeline_intraday
                   + intraday_pipe
                   + whatif_pipeline_intraday,
    # Only for monthly
    "monthly_pipeline":
                      data_pipeline_monthly 
                    + monthly_input_validation
                    + Volume_pipeline_monthly
                    + aht_pipeline_monthly
                    + occ_pipeline_monthly
                    + aggregator_pipeline_monthly
                    + monthly_output_validation 
                    + whatif_pipeline_monthly,
    #for compLete pipeline
"complete_pipeline":
                    data_pipeline_daily
                  + Volume_pipeline_daily
                  + calls_answered_within_30_pipeline_daily 
                  + aht _pipeline_daily
                  + occ_pipeline_daily
                  + aggregatar_pipeline_daily
                  + data_pipeline_intraday
                  + intraday_pipe
                  + data_pipeline_monthly
                  + Volume_pipeline_monthly
                  + aht_pipeline_monthly
                  + occ_pipeline monthy
                  + aggregator_pipeline_monthly
}


      


  


    
  


