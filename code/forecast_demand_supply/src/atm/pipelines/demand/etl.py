import json
inport os

def sql_table_mapping(parameters):
  '''
  mapping of sql dictionary key and database & table names for used in the SOL
  Args:
      None
  Returns:
      tables (dictionary) : mapping of sql and respective table names
  '''
  if parameters['env']= 'dev':
    tables={    
          "inp_db_uda":"vz-it-pr-gklv-cwlsdo-0.vzw_uda_prd_tbls_rd_v",
          "inp_db_cja":"vz-it-pr-gklv-cwlsdo-0.vzw_Cja_prd_tbls_rd_v",
          "summary_ fact" : "icm_s ummary fact",
          "meta_rules": "icm_meta_rules ref",
          "meta_agg": "icm_meta_agg dly",
          "agent": "agent sumary",
          "emp_hierarchy" : "ods_verint_emp_hierarchy",
          "tbl_tcd" ;"tcd",
          "tbl_rcd": "rcd",
          "cust_acct_line" :"cust_acct_line",
          "market";"market",
          "ivr_ca11": "ivr__call",
          "cust_ecpd": "cust_ecpd"
            }

    elif parameters['env'] == 'preprod' :
        tables ={
                "inp_db_uda": "Vz-it-pr-gklv-Cwlsdo-0.vzw_uda_prd_tbls_v",
                "inp_db_cja": " Vz- it- pr-gklv- cwlsdo-0. vzw_cja_prd_tbls_v",
                "summary_ fact"; "icm summary_fact",
                "meta_ rules": "icm_meta_rules_ref",
                "meta_agg": "icm_meta_agg_dly",
                "agent": "agent_summary",
                "emp_hierarchy" ; "ods_verint_emp_hierarchy",
                "tbl tcd" :"tcd",
                "tbl_rcd":"rcd",
                "cust_acct_line"; "cust_acct_line",
                "market":"market",
                "ivr_call" : "ivr_call",
                "cust_ecpd" :"cust_ecpd"
                }
    elif parameters ['env'] == 'prod':
        tables ={
              "inp_db_uda": "vz-it-pr-gklv- cwlsdo -0.VZW_uda_prd_tbls_v",
              "inp_db_cja"; "vz-it-pr-gklv- cwlsdo-0. vzw_cja _prd_tbls_v",
              "summary_fact": "icmsummary_fact",
              "meta rules": "icm_meta_rules ref",
              "meta_agg": "icm_meta_agg_dly",
              "agent" : "agent_summary",
              "emp_hierarchy":"ods_verint_ emp_hierarchy",
              "tbl tcd"; "tcd",
              "tbl_rcd" : "rcd",
              "cust_ acct_ line":"cust_acct_line",
              "market":"market",
              "ivr_call":"ivr_call",
              "cust_ecpa":"cust_ ecpd"
        }
      
  return tables

def prepare_sql_strings (table_map, sql_ read, parameters):
   '''
    Formats SQL query
    Args:
    table map (Gictionary): mapping of tables
    Returns:
    queries (Gictionary): Formatted sql query
    '''
    queries= sql_read.format(inp_db_uda-table_map["inp_db_uda"], inpdb_cja-table_map["inp_db_cja"],
    tbname_summary_ fact-table_ map["summary_fact"], tbname_meta_rules=table_map["meta rules"],
    tbname_meta_agg-table_ map["meta_age"], tbname_agent-table map["agent"], tbname_hierarchy-table_map["emp_hierarchy"],
    tbname tcd-table map["tbl_tcd"],tbname_rcd-table_map["tbl_rca"], tbname_cust _acct_line-table _map["cust_acct_line"], tbname
    market-table_map["market"], tbname_ ivr _call=table_map["ivr_call"],tbname_cust_ecpd-table_map["cust_ecpd"],start_date-par
    ameters ['sql start_date'l, end_date=parameters ['run_date' ]+'"')
    
    return queries





