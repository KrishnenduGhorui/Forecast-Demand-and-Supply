from kedro.pipeline import Pipeline, node
from .nodes inport
input_validation, load_data,load_demand_data_queue, train forecast_models, holdout_metrics, select_model_forecast, forecasting model_selectior,stack_model, holdout_metrics_without_stack, forecasting model_selection_ without_stack,output_validation, train_test_splitting

def load_data _pipeline (level1):
    return Pipeline(
           [
            node(
                func=load_data,
                inputs=[f' params:(level}_sq1_file _path',f' params: bq_project',' parameters'],
                outputs=f'data_df_{level}',
                )
           ])

def input_valídatíon_pipeline(level) :
    return Pipeline(
          [
          node(
                func=input_validation,
                inputs=[f'data_df_{level}',f'params: {level}', ' parameters'],
                outputs=None,
              )
          ])
def output_validation_pipeline(level):
    return Pipeline(      
                    node(func=output_validation,
                    inputs=[f'final_output_df_{level}', f'params :{level}','parameters' ],
                    outputs=None)
                    ])
  
def demand_data_pipeline (stack_flag, suffix, level):
    if(stack_flag == 0):
        return Pipeline(
          [
            node(
                func=load_demand_data_queue,
                inputs=[f'data _df_{level}',f'params:{suffix}' ],
                outputs=f'demand_df_{suffix}_{level}',
            ),
           node(
                func=train_test_splitting,
                inputs=[f'demand df_{suffix}_{level}',f'params : rule_type_list',f' params : Volume_train _start _date', f' params : run_date', f' params :{level}',f'params :demand_ split_ratio_{level}'],
                outputs=[f'train_start_date _{suffix}_{level}' ,f'train_end _date_{suffix}_{level}', f'test _start _date_{suffix}_{level}', f'test_end_date_{suffix}_{level}'],
           ),
            node(
                func=train_forecast_models,
                inputs=[f'demand_df_{suffix}_{level}', f'params : rule_type_list',f'train_start_date_{suffix}_{level}',f'train_end dste_{suffix}_{level}',f'test_start_date_{suffix}_{level}', f'test_end_date {suffix}_{level}', f'params : {level}_params_ets',f'params:{level}_params_sarima', f'params : perams_tes',f'params : params_grid_prophet' ,f'params :model_{level}'],
                outputs=f'best_models_{suffix}_{level}',
            node(
                func=holdout_metrics_without_stack,
                inputs=[f' params :{suffix}',f' demand_df_{suffix}_{level}', f' best_models_{suffix}_{level}',f'test_start_date_{suffix}_{level}', f'test_end_date_ {suffix}_{level}',f'params:model_{level}' , f'params:score_model_id',f' params :run key', f'params: output_ project',f'params :output db_name',
                        f'params:output_metrics_{level}_intermediate_tb_name', 'params :run_date'],
                outputs=f'model_mape_dict_{suffix}_{level}'
                ),
            node(
                func=select_model_forecast,
                inputs=[f'params:{suffix}' , f'model_mape_dict_{suffix}_{level}',f'params : score_nodel_id', f'params : run_key ',f'params : output_project',f'params :output_db_name'
                        ,f'params :output_metrics_{level}_tb_name', 'params : run_date '],
                outputs=f' rule_type_model_dict_{suffix}_{level}'
               ),
            node (
                func=forecasting_model_selection_without_stack,
                inputs[f'params :{suffix}',f'params : {level}',f' best_models_{suffix}_{level}',f'rule_type_model_dict_{suffix}_{level}', f' demand_df_{suffix}_{level}',f'train_start_date_{suffiX}_{level}', f'train_end_date_{suffix}_{level}',f'test_start_date_{suffix}_{level}',f'test_end_date_ {suffix}_{level},f'params:
                        {level}_forecast period',f'params:model_{level}',f'params :num_lags',f' params:score_model_id',f' params :run_key', 'params run_date'],
                outputs=f'output_df_{suffix}_{level}'
                )
                ]
            )
    elif(stack_flag==1):
                return Pipeline(
                    [
                     node(
                func=load_demand_data_queue,
                inputs=[f'data _df_{level}',f'params:{suffix}' ],
                outputs=f'demand_df_{suffix}_{level}',
            ),
                   node(
                        func=train_test_splitting,
                        inputs=[f'demand df_{suffix}_{level}',f'params : rule_type_list',f' params : Volume_train _start _date', f' params : run_date', f' params :{level}',f'params :demand_ split_ratio_{level}'],
                        outputs=[f'train_start_date _{suffix}_{level}' ,f'train_end _date_{suffix}_{level}', f'test _start _date_{suffix}_{level}', f'test_end_date_{suffix}_{level}'],
                   ),
                    node(
                        func=train_forecast_models,
                        inputs=[f'demand_df_{suffix}_{level}', f'params : rule_type_list',f'train_start_date_{suffix}_{level}',f'train_end dste_{suffix}_{level}',f'test_start_date_{suffix}_{level}', f'test_end_date {suffix}_{level}', f'params : {level}_params_ets',f'params:{level}_params_sarima', f'params : perams_tes',f'params : params_grid_prophet' ,f'params :model_{level}'],
                        outputs=f'best_models_{suffix}_{level}',
                    node(
                        func=holdout_metrics,
                        inputs=[f'demand_df_{suffix}_{level}', f' best_models_{suffix}_{level}','returned_list',f'test_start_date_{suffix}_{level}', f'test_end_date_ {suffix}_{level}',f'params:model_{level}'],
                        outputs=f'model_mape_dict_{suffix}_{level}'
                        ),
                    node(
                        func=select_model_forecast,
                        inputs=[f'params:{suffix}' , f'model_mape_dict_{suffix}_{level}',f'params : score_nodel_id', f'params : run_key ',f'params : output_project',f'params :output_db_name'
                                ,f'params :output_metrics_{level}_tb_name', 'params : run_date '],
                        outputs=f' rule_type_model_dict_{suffix}_{level}'
                       ),
                    node (
                        func=forecasting_model_selection,
                        inputs[f'params :{suffix}',f'params : {level}',f' best_models_{suffix}_{level}',f'rule_type_model_dict_{suffix}_{level}', f' demand_df_{suffix}_{level}',f'train_start_date_{suffiX}_{level}', f'train_end_date_{suffix}_{level}',f'test_start_date_{suffix}_{level}',f'test_end_date_ {suffix}_{level},f'params:
                                {level}_forecast period',f'params:model_{level}',f'params :num_lags',f'params:score_model_id',f'params :run_key', 'params: run_date'],
                        outputs=f'output_df_{suffix}_{level}'
                        )
                ]
            )
                        
                






