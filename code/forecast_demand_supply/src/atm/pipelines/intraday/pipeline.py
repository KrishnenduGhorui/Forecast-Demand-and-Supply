from kedro.pipeline import Pipeline, node
from .nodes import load_daily_forecast, forecast_intraday, build_scenario_table_intraday
from ..demand.model_train import replace_scenario_table

def intraday_pipeline():
    return Pipeline(
      [
        node(
            func=load_daily_forecast,
            inputs=[f' forecast_output_df_daily'],
            outputs='df_daily_forecast',
        ),
        node(
            func-forecast_intraday,
            inputs=['data_df_intraday', 'df_daily_forecast',f'params:run date',f' params:score_model_id',f' params:run key',f' params :output_project',f'params:b
                    qproject',f'params:output_db_name',f'params:output_tb_name_intraday','params :rundate'],
            outputs='df_result_forecasted intra',
        )
      ]
    )
  
def intraday_pipeline_whatif():
    return Pipeline(
      [
        node(
          func=buíld_scenario_table_intraday,
          inputs=[f'df_result_forecasted_intra',f' params: period_whatif_intraday' ],
          outputs=f'df_scenario_table_intraday'),
        
        node (
          func=replace_scenario_table,
          inputs=[f'params :bq_praject',f'params:output_project',f'params:output_db_name', f'params :output whatif_scenario_intraday_tb _name', f'df_scenario_table_ intraday'],
          Output=None)
      ]
    )



