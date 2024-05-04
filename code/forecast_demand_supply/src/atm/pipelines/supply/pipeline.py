from kedro.pipeline import Pipeline, node

from .nodes import
load_ supply_ data_queue,train_forecast_models, holdout_metrics, select_model_forecast, forecasting_model_selection,stack_odel, holdout metric
s_without_stack, forecastìng_model_selection_without_stack, final_output_daily, final_output_monthly
from ..demand.nodes import train_test_splitting
from ..demand.model_train import build_scenario_table_daily, build_scenario_table_monthly, replace_scenario_table

def supply_data_pipeline( suffix, stack_flag, level) :
    if(stack_flag = 0):
      return Pipeline (
        [
          node(
              func=load_supply_data_queue,
              inputs-[f' data_df_{level}', f' params :key_arg_{suffix}'],
              outputs=f'supply_df_{suffix}_{level}',
          node(
              func=train _test_splitting,
              inputs=[f'supply_df_{suffix}_{level}', f'params :rule_type_list',f' params:supply_train_start_date',f'params:run _date', f' params :
              {level}',f' params:supply_split_ratio_{level}'],
              outputs=[f'train_start_date_{suffix}_{level}',f' train_end_date_{suffix}_{level}',f'test_start _date_{suffix}_{level} ', f'test_end_date_ {suffix}_{level}'],
              ),
          node(
              func=train_forecast_models,
              inputs=[f'supply_df_{suffix}_{level}',f'params:rule_type_list',f'train_start date_{suffix}_{level}',f'train_end_date_{suffix}_{level}',f'test_start_date_{suffix}_{level}',f'test_end_date_{suffix}_{level}',f'params_{level}_params_ets',f' params :
                      level} params_sarima,'f' params:params_tes',f'params : params_grid_prophet',f'params :model_{level}'],
              outputs=f'best_models_{suffix}_{level}',
              ),
          node(
              func=holdout_metrics_without_stack,
              inputs=[f'params:key_arg_{suffix}',f'supply_df_{suffix}_{level}',f' best_models_{suffix}_{level}', f' test_start_date_{suffix}_{level}',f'test_ end
              date_{suffix}_{level}',f' params :model_{level}' ,f'params :score_model_id',f'params :run key',' params :output_project',f' params:output_db_name',f'params:output_metrics_{level}_intermediate_tb_name','params:run_date'],
              outputs=f'model_mape_dict {suffix}_{level}'
               ), 
          node(
              func=select_model_forecast,
              inputs=[f'params :key_arg_ {suffix}',f'model_mape_dict_{suffix}_{level}',f'params:score_model_id',f' params :run_key',f'params :output_project',f'params:output_db name',f' params :output_ metrics_{level}_tb_name','params :run_date'],
              outputs=f' rule_type_model_dict_{suffix}_{level)'
              ),  
         node (
              func=forecasting_model_selection_without_stack,
              inputs=[f'params : key_arg_{suffix}',f'params:rule_type_model_dict_{suffix}_{level}',f'supply_df_{suffix}_{level}',f'train_start_date_{suffix}_{level}',f'best_models_{suffix}_{level}',f'x}:{level}',f'train_end_date_(suffix}_(level}',f'test_start_date_(suffix}_(level}',f'test_end date_{suffix)_(level}', f'params:
                     {level}_forecast_period',f'params : model_{level}',f' params :score_model_id',f'params :run _key',f'params:run_date',
              outputs=f'output_df_{suffix}_{level}'
              )
            ]
         )

    elif(stack_flag == 1):
        return Pipeline(
                        node(
                            func=load_supply_data_queue,
                            inputs=[f'data_df_{level}',f' params : key_arg{suffix} ',
                            outputs=f'supply_df_{suffix}_{level}',
                            ),
                        node(
                            func=train_test_splitting,
                            inputs=[f'supply_df_{suffix}_{level}',f'params : rule_type_list',f'params : supply_train_start_date',f'params :run_date',f'params :
                                   {level}',f'params :supply_split_ratio_{level} ']
                            outputs=[f'train_start_date_{suffix}_{level}',f' train_end_ date_{suffix}_{level}', f'test_start_date_{suffix}_{level}',f'test_end_date_{suffix}_{level}'].
                            ),
                          node(
                            func=train_forecast_models,
                            inputs=[f'supply_df_{suffix}_{level}',f'params:rule_type_list',f'train_start date_{suffix}_{level}',f'train_end_date_{suffix}_{level}',f'test_start_date_{suffix}_{level}',f'test_end_date_{suffix}_{level}',f'params_{level}_params_ets',f' params :
                                    {level}_params_sarima','f' params:params_tes',f'params : params_grid_prophet',f'params :model_{level}'],
                            outputs=f'best_models_{suffix}_{level}',
                              ),
                          node(
                            func=stack_model,
                            inputs=[f' params:{level}',f'supply_df_{suffix}_{level}',f'best_models_{suffix}_{level}' ,f'params :rule_type _list',f'train_start_date{suffix}_{level}',f'train_end_date_{suffix}_{level}', f'test_start_date_{suffix}_{level}',f'test_end_date_{suffix}_{level}",f'params : num_lags',f'params : params_stack'],
                            outputs=f'returned_list_{suffix}_{level}',
                            ),
                          node(
                            func=holdout_metrics,
                            inputs=[f'supply_df_{suffix}_{level}',f'best_models_{suffix}_{level}', f' returned_iist_{suffix}_{level}',f'test_start_date_{suffix}_{leve1}',f'test_end_date_{suffix}_{level}',f' params :model_{level}',
                            outputs=f'model_mape_dict_{suffix}_{level}'
                              ),
                          node(
                            func=select_model_forecast,
                            inputs=[f'params :key_arg_ {suffix}',f'model_mape_dict_{suffix}_{level}',f'params:score_model_id',f' params :run_key',f'params :output_project',f'params:output_db_name',f' params :output_ metrics_{level}_tb_name','params :run_date'],
                            outputs=f' rule_type_model_dict_{suffix}_{level}'
                             ),
                          node(
                              func=forecasting_model_selection,
                              inputs=[f' params :key_arg_{suffix}', f' params:{level}',f'returned_1ist_{suffix}_{level}',f' best_models_{suffix}_{level}',f'rule_type_model_dict_{suffix}_{level}',f'supply_df_{suffix}_{level}',f'train_start_date_{suffix}_{level}', f'train_end_date_{suffix}_{level}',f'test_start_date_{suffix}_{level}',f'test_end_date_{suffix}_{level}',
                              f' params:{level}_forecast_period',f' params:model_{level}',f'params :num_lags',f' params:score_model_id',f'params :run_key', 'params :run_date'],
                              outputs=f'output_df_{suffix}_{level}'
                              )
                        ]
                    )

def aggregator_pipeline (level):
    if(level=='daily'):
        return Pipeline(
            [
                node(
                    func=final_output_daily,
                inputs=[f'output_df_aht(level},f' outputdf_occ_(levely ',f' output_df_Volume_{level}',f'output_df_calls_answered within_30(level}",f'params :(level},f'params:output_project',f' params :output_db_name',f' params :output_tb_name_daily'],
                outputs=[f'final_output_df_daily',f' forecast_output_df_daily'])
            ]
            )
    elif(level=='monthly'):
        return Pipeline
                [
                node(
                    func=final_output_monthly,
                    inputs=[f'output_df_aht_{level}',f'output_df_occ_{level}', f'output df_Volume_{level}',f'params :{level}',f'params :output project',f' paräms :output_db_name',f'params : output_tb_name_monthly'],
                    outputs=f'final_output_df_{level})
                ]
                )

                            
                            



                            
                          









