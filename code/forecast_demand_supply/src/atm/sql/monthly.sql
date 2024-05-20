
with DT AS
(
/** Adjust the date here for the below queries ***/
Select
Date({start_date}) as RecordStart,
Date({end date}) as RecordEnd,
),
ICM As
(
With ICH_VT AS
SELECT I.*,
r.rule_type as r_rule_type,
r.rule_audit_ ind AS rrule _audit_ ind,
/*****a* Below code deats the time without dcmf half_ hour intervals table******/
CAST (TRIM( Case
When i.answer_half_hr = 27 Then '13:00:00'
When i.answer_half_hr = 29 Then '14:00:00'
When i.answer_ half_hr = 21 Then '10:00:00'
When i.answer_half_hr =8 Then '03:30:00'
When i.answer_half_hr = 2 Then '0:30:00'
When i.answer_half_hr = 25 Then'12:00:00'
When i.answer_half_hr = 6 Then '02:30:00'
When i.answer_half_hr = 4 Then '01:30:00'
When i.answer_half_hr = 44 Then'21:30:00'
When i.answer_half_hr= 46 Then '22:30:00'
When i.answer_half_hr=12 Then '05:30:00'
When i.answer_half_hr = 48 Then'23:30:00'
When i.answer_half_hr = 31 Then '15:00:00'
When i.answer_half_hr = 41 Then '20:00:00'
When i.answer_half_hr =33 Then '16:00:00'
When i.answer_half_hr = 10 Then '04:30:00'
When i.answer_half_hr= 18 Then '08:30:00'
When i.answer_half_ hr 22 Then '10:30:00'
When i.answer_half_hr = 39 Then '19:00:00' 
bihen i.answer_half_hr = 24 Then '11:30:00'
When i.answer_half_hr = 16 Then '07:30:80'
ihen i.answer_half_hr = 30 Then '14:30:08'
ihen i.answer_half_hr = 11 Then '05:00:00'
lhen i.answer_half_hr = 9 Then '04:00:00'
lhen i.answer_half_hr = 23 Then '11:00:00'
When i.answer_half_hr = 32 Then '15:30:00'
When i.answer_half_hr = 26 Then '12:30:00'
When i.answer_half_hr = 28 Then '13:30:00'
When i.answer_half_hr = 5 Then '02:00:00'
When i.answer_half_hr= 45 Then '22:00:00'
When i.answer_half_hr= 47 Then '23:00:00'
When i.answer_half_hr = 43 Then '21:00:00'
When i.answer_half_hr = 15 Then '07:00:00'
When i.answer_half_hr = 40 Then '19:30:00'
When i.answer_half_hr = 7 Then '03:00:00'
When i.answer_half_hr = 1 Then '00:00:00'
When i.answer_half_hr =3 Then '01:00:00'
When i.answer_half_hr = 37 Then '18:00:00'
When i.answer_half_hr= 20 Then '09:30:00'
When i.answer_half_hr = 17 Then '08:00:00'
When i.answer_half_hr = 34 Then '16:30:00'
When i.answer_half_hr = 38 Then '18:30:00'
When i.answer_half_hr = 36 Then '17:30:00'
When i.answer_half_hr = 35 Then '17:00:00'
ihen i.answer_half_hr= 13 Then '06:00:00'
ihen i.answer_half_hr = 19 Then '09:00:00'
When i.answer_half_hr= 14 Then '06:30:00'
When i.answer_half_hr = 42 Then '20:30:00'
End) AS TIME) AS interval_start,
CAST(TRIM(CASE
When i.answer_half_hr = 27 Then '13:30:00'
When i.answer_half_hr = 29 Then '14:30:00'
When i.answer_half_hr = 21 Then '10:30:00'
When i.answer_half_hr = 8 Then '04:00:00'
When i.answer_half_hr = 2 Then '01:00:00' 
When i.answer_half_hr = 25 Then '12:30:00'
When i.answer_half_hr =6 Then '03:00:00'
When i.answer_half_hr = 4 Then '02:00:00'
When i.answer_half_hr = 44 Then '22:00:00'
When i.answer_half_hr= 46 Then '23:00:00'
When i.answer_half_hr= 12 Then '06:00:00'
When i.answer_half_hr= 48 Then '00:00:00'
When i.answer_half_hr= 31 Then '15:30:00'
When i.answer_half_hr = 41 Then '20:30:00'
when i.answer_half_hr = 33 Then '16:30:00'
When i.answer_half_hr = 10 Then '05:00:00'
When i.answer_half_hr = 18 Then '09:00:00'
When i.answer_half_hr = 22 Then '11:00:00'
When i.answer_half_hr =39 Then '19:30:00'
When i.answer_half_hr=24 Then '12:00:00'
When i.answer_half_hr=16 Then '08:00:00'
When i.answer_half_hr=30 Then '15:00:00'
When i.answer_half_hr = 11 Then '05:30:00'
When i.answer_half_hr = 9 Then '94:30:00'
ihen i.answer_half_hr= 23 Then '11:30:00'
When i.answer_half_hr = 32 Then '16:00:08'
When i.answer_half_hr = 26 Then '13:00:00'
When i.answer_half_hr = 28 Then '14:00:00'
When i.answer_half_hr = 5 Then '02:30:00'
When i.answer_half_hr = 45 Then '22:30:90'
When i.answer_half_hr = 47 Then '23:30:00'
When i.answer_half_hr = 43 Then '21:30:00'
When i.answer_half_hr= 15 Then '07:30:00'
When i.answer_half_hr = 40 Then '20:00:00'
When i.answer_half_hr =7 Then '03:30:00'
When i.answer_half_hr= 1 Then '00:30:00'
When i.answer_half_hr = 3 Then '01:30:00'
When i.answer_half_hr = 37 Then '18:30:00'
When i.answer_half_hr = 20 Then '10:00:00'
When i.answer_half_hr =17 Then '08:30:00'
When i.answer_half_hr = 34 Then '17:00:00'
When i.answer_half_hr = 38 Then '19:00:00'
When i.answer_half_hr=36 Then '18:00:00'
When i.answer_half_hr=35 Then '17:30:00'
When i.answer_half_hr=13 Then '06:30:00'
When i.answer_half_hr = 19 Then '09:30:00'
When i.answer_half_hr= 14 Then '07:00:00'
When i.answer_half_hr= 42 Then '21:00:00'
END) AS TIME) AS interval_end
FROM {inp_db_uda}.{tbname_summary_fact} i
LEFT JOIN
SELECT* except (R) From(
SELECT row_number() OVER (partition by bus_rule order by cast (bus_rule_id as numeric) desc as R,*
FROM {inp_db_cja}. {tbname_meta_rules} )
WHERE R= 1
) r
ON r.bus_rule = i.bus_rule
--AND i.CaLl_answer_dt BETWEEN r.eff_dt AND r.exp_dt /*** James: No need as we SELECT the newest rules
LEFT JOIN DT ON 0=0
WHERE Date(i.call_answer_dt) BETWEEN DT.RecordStart AND DT.RecordEnd
  ),
ICM_PATCHNA_VT AS
(
SELECT
recoverykey,
recover_bus_rule
FROM
SELECT
i.recoverykey,
i.IVR_Call_ID,
i.CALL_ ANSNER_ DT ,
i.BUS_RULE as orig_bus_rule,
r.VARIABLE8 as recover_bus_rule,
ROW_NUMBER() OVER (PARTITION BY i.recoverykey ORDER BY r.DATETIME asc) as rownumber
FROM {inp_db_uda}.(tbname_summary_fact} i
INNER JOIN
inp_db_uda}.(tbname_ tcd} T
ON T.RECOVERYKEY = i.RECOVERYKEY
-- do not alttempt to optimize the joins on WHERE CLause below, each item plays a critical role in the patch
INNER JOIN inp- db_uda}.(tbname_rcd} r
ON r.ROUTERCALLKEYDAY = t.ROUTERCALLKEYDAY
AND r. ROUTERCALLKEY = t.ROUTERCALLKEY
AND r.ROUTEID = t.ROUTEID
LEFT JOIN DT ON 0=0
--AND r.DIALEDNUMBERSTRING <> 'CTIOS_XFR' /*** James: Hide as it is a redacted field ***f
/******* Mask the VARIABLE8 filter as it is redacted *******/
WHERE --r. VARIABLE8 <> i.BUS_RULE /*** Jamęs: Hide as it is a redacted field ***/
--AND
( length(i.bus rule). < 3 or ( length(i.bus_rule) =3 AND i.AGENT_ GROUP_ID = 13090)) --AFNI MESS
--dont destroy CTI Calls WHERE the 3 Letter rule is gk!
--AND r. VARIABLE8 not Like %;% --get rid of a fw double xfers, pv8 is already Lost AND has the format ACSSREP; /***

--AND r. VARIABLE8 not Like 'S K--get rid of any that contain spaces, cannot be valid; *** James: Hide as it is a
AND Date(i.call_answer_dt) BETWEEN DT.RecordStart AND DT RecordEnd
WHERE rownumber = 1 --THIS ENSURES WE NEVER CREATE MORE THAN 1 row per RECOVERYKEY
SELECT
rule_type,
DATE_ TRUNC(call_answer_dt, MONTH) as call_ month,
call_ answer dt,
eid,
count() call_volume,
Sumanswered cnt) as answered volume,
Sum(ansr_38_cnt) as answered_within_30 volume
from
(
SELECT
Case when r.rule_type In ('Tech Adv Call', 'Global Call') Then 'Tech Adv Call' Else r.rule_type End as rule_type,
r.rule_dept,
r.rule_desc,
a.ag_desc,
i,*
from
(
SELECT
i.recoverykey recovery_key,
i.ivr call_id ivr_call_id,
1.acd area nm,
i.call_end_dt,
i.call_end_tm,
i.call_answer_dt,
i.call_answer_tm,
i.route_value,
i.icm_acct_type_cd,
mkt.vzmkt_cd,
cal.line_type_cd,
i.eqp_prod_id,
i.cust_value,
i.lang_pref_ind,
i.cacs_state_cd,
i.first_bill_cd,
i.transfer_point,
--CASE WHEN i. high_risk_ ind = 'H' THEN "H' ELSE 'N' END AS high_risk_ind,
CASE WHEN i.high_risk_ind is null or i.high_risk_ind = THEN 'U' ELSE i.high_risk_ind END AS high risk_ind,
i.cacs_Work_state_cd,
i.ivr_cust_src_cd,
COALESCE(
1*** James: Dealing with the ";" and " " rules ***/
CASE When patchbr.recover_bus_rule like '%;%' Or patchbr.recover_bus_rule like '% %' Then NÙLL Else
patchbr.recover_bus_rule END,
CASE WHEN (i.BUS_RULE='NA' OR i.BUS_ RULE is NULL) and i.eccr_dept_nm IN ('BGCO Tech', 'Business - FA', ' BSC') THEN
'CATCH_NA_BGCO'
WHEN (r_RULE_TYPE='Routing Fallout Call') and i.eccr dept_nm IN ('BGCO Tech', 'Business - FA', 'BSC') and
i.icm_acct_type_cd='FEDC' THEN 'CATCHFED'
WHEN (r_RULE_TYPE='Routíng Fallout Call') and i.eccr_dept_nm IN ('BGCO Tech', 'Business - FA", 'BSC') and
i.icm_acct_type_cd='GOVC' THEN 'CATCH_ GOVC'
WHEN (r_RULE_TYPE='Routing Fallout Call') and i.eccr_dept_nm IN (BGCO Tech', 'Business - FA', 'BSC') THEN
'CATCH FALLOUT BGCO'
ELSE i.bus_rule END) AS bus_rule,
i.script_nm,
CAST(i.eccr_line_bus_nm AS STRING) AS eccr_line_bus_nm,
i.eccr_super_line_bus_nm,
i.eccr_dept_nm,
i.mtn,
i.acd_appl_id,
CASE WHEN(i.answered_cnt = 0 and i.call_offered_cnt = 1 and i.AGENT_GROUP_ID is null) THEN 999999
ELSE i.AGENT_GROUP_ID
END AS agent_group_id,
i.transfer_flag,
i.call_duration_seconds,
i,ring_tm_seconds,
i.delay_tm_seconds,
i.tine_to_abandseconds,
i.hold_tm_seconds,
i.talk_tm_seconds,
i.work_tm_seconds,
i.local_q_tm_seconds,
i.handle_tm_seconds,
i.delay_answer_seconds,
i.call_offered_cnt, 
i.interval_start,
i.interval_end,
i.abandons_cnt,
i.answered_cnt,
i.ansr_30_cnt,
i.ansr_30_to_40_cnt,
CASE WHEN (1.answered_cnt = 0 AND i.call_offered_cnt 1 AND i.AGENT_GROUP ID IS NULL) THEN 999
ELSE i. callcenterid_agent END AS callcenterid agent,
i.ECCR_CALL_CTR_CD,
CASE WHEN (i.answered_cnt = 0 and i.call_offered_cnt = 1 and i.AGENT_GROUPID IS NULL) THEN 999
ELSE i.CALLCENTERID
END AS callcenterid,
i.sor_id,
i.cust_id,
i.cust_line_seq_id,
i.acss_call_id,
iv.DNIS_CD,
COALESCE(cae. ecpd_profile_id, '0') ecpd_profile_id,
i.eid
FROM ICM_VT i
/***************8
Patch broken NA Rule BEGINS here
*****/
/*
NOTE THIS IS A LEFT OUTER JOIN TABLE, do nọt inner join on it or records wilL be Lost. TCD AND RCD tables onty go back
90 DAYS
because TCD AND RCD. are so Limited by dates (90 days) we dont really have to put a date range in the WHERE clause.
This code is slow, but only because the tables in DW Lack AND valuable indexing - do not attempt to alter the below
methods each
has been tested extensively to ensure clean data
*/
LEFT OUTER JOIN ICM PATCHNA_ VT patchbr
ON patchbr.recoverykey = i.recoverykey -- END LEFT OUTER JOIN TABLE (PATCH)
/*************
Patch broken NA RuLe ENDS here
******************/
LEFT JOIN {inp_db_uda}.{tbname_cust_acct_line} as cal
ON cal.sor id = i.sor_id
AND cal.cust_id = i.cust_id
AND cal.cust_line_seq_id = i.cust_line_seq_ id
LEFT JOIN {inp_db_uda}.{tbname market} as mkt
ON mikt.mikt_cd = cal.mkt_cd
LEFT JOIN {inp db_uda}.{tbname ivrcall} iv
ON iv.IVR_CALL ID = i.IVR_CALL_ID
AND iv.TRANSFER_POINT IS NOT NULL
  
LEFT JOIN {inp_db_uda}.{tbname cust_ ecpd} cae
ON i.cust id = cae.Cust id
AND CASTi.call_ anser dt AS DATE) >= CAST(cae.ecpd eff_dt AS DATE)
AND CAST (i.call_an swer_dt AS DATE), <= CAST(Cae.ecpd_term_dt AS DATE)
LEFT JOIN (
SELECT* except (R) From (
SELECT row_number() 0VER (partition by Ag_Branch order by cast (Ag_Rcd_Id as numeric) desc) as R,
FROM {inp_db_cja}. {tbname_meta_agg} )
WHERE R = 1
pXpX
SELECT row_number() OVER (partition by Ag Branch order by cast (Ag Rcd_Id as numeric) desc) as R, *
FROM (inp_db_cja).(tbname_meta_agg} )
WHERE R = 1
)a
ON CAST(i.AGENT_GROUP_ID AS STRING) = a.AGENT_GROUP_ID
--AND i.call answer dt BETWEEN a.eff_dt AND a. exp_dt /*** James: No need as we SELECT the newest ag ***/
AND
CAST(COALESCE (i.callcenterid_agent, i.CALLCENTERID) AS STRING) = a.call_center_id -- the coalesce is important!
OR i.ECCR CALL CTR_CD = a.ECCR_CALL_CTR_CD
OR (i.callcenterid_agent IS NULL AND a.AG_CENTER_BYPASS='Y')
)
WHERE (
--r_rule_audit_ind = "y' OR a. audit_ind = 'y OR
i.ECCR_DEPT_NM IN (BGCO Tech', 'BSC', 'Business - FA')
--OR i.icm_acct_type_cd IN ("MAJX", "FEDC', 'SMBC', 'NATX, GOVC', 'CHAJ')
AND Date(i. call_answer_dt) BETWEEN (start_date} AND (end_date)
)i
/******* Rejoin the Meta Rules table to find the rule_type, rule_dept, and rule_desc by the adjusted bus_rule *******/
LEFT JOIN (
SELECT * except (R) From(
SELECT row_number ) oVER (partition by bus_rule order by cast(bus_rule_id as numeric) desc) as R,
FROM (inp_db_cja).(tbname_meta_rules }. )
WHERE R= 1
)r
ON r.bus_rule =i.bus_rule
--AND i,call_answer_dt BETWEEN r.eff_dt AND r. exp_ dt /*** James: No need as we SELÉCT the newest rule ***/
*******Rejoin, the Meta AG table to find the ag_desc by the adjusted AGENT_GROUP_ID, caLLcenterid_agent, and
catLcenterid *******
LEFT JOIN (
SELECT* except (R) From(
SELECT row_number() OVER (partition by Ag_Branch order by cast (Ag Rcd Id as numeric) desc) as R, *
FROM (inp_db_cja}.(tbname_meta_agg)
WHERE R= 1
a
ON CAST(i.AGENT_ GROUP_ID AS STRING) = a.AGENT_GROUP_ID
--AND i.call_answer_dt BETWEEN a. eff_dt AND a.exp_dt /s** James: No need as we SELECT the newest ag ***/
AND
CAST(COALESCE (i.callcenterid_ agent, i.CALLCENTERID) AS STRING) = a.call_center_id -- the coalesce is important!
OR i.ECCR_CALL_CTR_CD = a.ECCR_CALL_CTR_CD
OR (i.callcenterid_agent IS NULL AND a.AG_CENTER_ BYPASS='Y')
where rule type in ('BGCO Core Call', 'Federal Call','GCO Core Call', 'Tech Adv Call','Tech Exp Call')
group by call_month, call_ansiver_dt,rule_type, eid
),
Agent AS
SELECT*
From
SELECT
DATE_TRUNC (cast(a.Stat_Date as date),MONTH) AS RptMth,
cast(a.Stat Date as date) AS Stat_Date,
trim(b.emp_id) as emp_id,
trim(b.enterprise_id) as eid,
trim(b.level_id) as level_id,
trim(b.function _desc) as function_desc,
trim(b.geographic_loc_desc) as geographic_loc_desc,
trim(b.peoplesoft_title_id) as peoplesoft_title_id,
--trim(c. ag_grouping) as ag_grouping, /*** Grouping by rule_type for Phase II ***/
CASE WHEN cast(b.level_id as numeric) < 4 THEN SUM(CAST (IFNULL (cast (a . Handled_Calls as numeric), 0) AS BIGINT)) E1se 0
End AS CallsHandled,
CASE WHEN cast(b.level_id as numeric) < 4 THEN SUM(CAST (IFNULL(cast (a.Available_Time as numeric), 0) AS BIGINT)) Else 0
End AS Availablé,
CASE WHEN cast (b.level id as numeric) < 4 THEN SUM(CAST (IFNULL (cast (a. Unavailable_Time as numeric), 0) AS BIGINT)) Else
e End AS Unavailable,
CASE WHEN cast (b.level1 id as numeric)<4 THEN SUM(CAST(IFNULL(cast (a.SignedOn_Time as numeric), 0) AS BIGINT)) Else 0
End AS Signon,
CASE WHEN cast (b.level_id as numeric) < 4 THEN SUM(CAST (IFNULL(cast (a.Talk_time as numeric), 0) AS BIGINT)) Else 0 End
CASE WHEN cast(b.level_id as numeric) <4 THEN SUM(CAST(IFNULL(cast (a.Call_Hold_Time as numeric), 0) AS BIGINT)) Else 0
AS Falk,
CASE WHEN cast (b.level_id as numeric) <4 THEN SUM(CAST (IFNULL (cast (a.Cáll_Work_Time as numeric), 0) AS BIGINT)) Else 0
End AS Hold,
End AS Work,
CASE WHEN b.enterprise_id in
('4495815766', '1294824864', '2596489907', '4568927104','2987509857', '1791755608', '7912985533', '8553547467',
'2763224556','6348668747') Then 10*3600
Else 8 3600 End as WorkHoursInSeconds
/** Jomes : essume 8 hours shift for all reps except the InternaL Tech Experts night shift hich witL be 10 hours ****/
FROM {inp_db_cja}.{tbname_agent} a
LEFT JOIN {inp_db_cja}.{tbname_hierarchy} b
On a.employee_id = b.emp_id And a.stat_date between date (b.start_date) and date(b.end_date)
--LEFT JOIN (Select Distinct eid, ag_grouping from IH) c On c. eid = b. enterprise_id
LEFT 3OIN DT On -
HERE a.stat Date BETWEEN DT.RecordStart AND DT . RecordEnd
GROUP BY
RptMth,
a.Stat Date,
b.emp_id,
b.enterprise_id,
b.level_id,
b.function_desc,
b.geographic_loc_desc,
b.peoplesoft_title_id
--C.ag grouping /*** Grouping by rule_type for Phase II ***/
** excLuding the Partner reps that had -2 hrs signon time ***/
--lHERE ( /*** Grouping by rule_type for Phase II ***/
--(aggrouping NOT In ('Internal Advanced Solutions', 'Internal BGCO Dept Team', Internal GCO Phone Suppot, 'Internal TechAdvocates', 'Internat Tech Experts - Day', 'Internal Tech Experts - Night', 'Other')

SELECT
ICM.cal1_answer_dt,
ICM.rule type,
--ICM. interval_start,
Count (Distinct ICM.eid) as Total_Agent,
Sum(ICM.call_volume) as Volume,
Sum(ICH.answered_volume) as Answered_Volume,
Sum(ICM.answered within_30_volume) as calls_answered_ within_30,
SAFE_DIVIDE (SUM(Agent.Talk + Agent.Hold + Agent. Work),SUM(Agent.CallsHandled) ) AS AHT,
SAFE_DIVIDE (SUM(Agent .WorkHours InSeconds)- SUN(Agent.Signon) + SUM(Agent.Unavailable), SUM(Agent.lorkHoursInSeconds))
AS SHR,
J*** James: shrinkage with Lost hours are not correct in the ods_verint table, so we use the calculation of total hrs -
signon time + unavailab Le time as the shrinkage_with_Lost_hrs ***/
SAFE_DIVIDE (SUM(Agent. Talk + Agent.Hold + Agent.Work),SUM(Agent.Talk + Agent.Hold + Agent.Work + Agent.Available)) AS
OCC,
--SAFE_DIVIDE (SUM(Agent. Signon - Agent. Talk - Agent.HoLd - Agent. Work - Agent.Available), SUM(Agent. Signon) ) AS LOS,
--SAFE_DIVIDE (SUM(Agent. WorkHoursInSeconds) /3600, Count (Agent. eid)) As WorkHours
FROM ICM
LEFT J0IN Agent On ICM.call_answer_dt = Agent.Stat_Date And ICM.eid = Agent.eid
LEFT JOIN DT On -0
VWHERE ICM.call_answer_dt BETWEEN DT. RecordStart AND DT.RecordEnd
--AND ICH. ag._grouping Not in ('Abondon ', '0ther ') /*** Grouping by rule_type for Phase II ***/
AND ICM.rule_type in ('BGCO Core Call','Federál Cal1', 'Gco Core Call', 'G1obal Call', 'Tech Adv Call', 'Tech Exp Call')

GROUP BY 
ICM.call_answer_dt,
ICM.rule_type
ORDER BY
ICM.call_answer_dt,
ICM.rule_type
