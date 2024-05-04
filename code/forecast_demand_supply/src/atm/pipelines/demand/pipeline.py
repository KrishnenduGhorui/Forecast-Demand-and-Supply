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
    if(stack_flag == 8):
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
func-train_forecast_models,
ooutr




