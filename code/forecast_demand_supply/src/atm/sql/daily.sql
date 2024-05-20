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











