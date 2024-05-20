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
When i. answer_half_hr = 27 Then '13:00:00'
When i. answer_half_hr = 29 Then '14:00:00'
When i, answer_ half_hr = 21 Then '10:00:00'
When i.answer half hr =8 Then '03:30:00'
When i.answer half hr = 2 Then '0:30:00'
When i. answer_half_hr = 25 Then'12:00:00'
When i. answer half_ hr = 6 Then '02:30:00'
When i.answer half_ hr = 4 Then '01:30:00'
When i, answer half_ hr = 44 Then'21:30:00'
When i.answer half_ hr= 46 Then '22:30:00'
When i.answer half hr 12 Then '05:30:00'
When i.answer half_ hr = 48 Then'23:30:00'
When i.answer half hr = 31 Then '15:00:00'
When i.answer half_ hr = 41 Then '20:00:00'
When i. answer half_ hr =33 Then '16:00:00'
When i. answer half_ hr = 10 Then '04:30:00'
When i.answer half hr = 18 Then '08:30:00'
When i.answer_half_ hr 22 Then '10:30:00'
When i. an swer_half_hr = 39 Then '19:00:00' 
bihen i. answer half_hr = 24 Then '11:30:00'
When i. answer_half_hr = 16 Then '07:30:80'
ihen i. answer_half_hr = 30 Then '14:30:08'
ihen i.answer_half_hr = 11 Then '05:00:00'
lhen i.answer half hr = 9 Then '04:00:00'
lhen i.answer _half_hr = 23 Then '11:00:00'
When i.answer half_hr = 32 Then '15:30:00'
When i. answer_half_hr = 26 Then '12:30:00'
When i. answer half_hr = 28 Then '13:30:00'
When i.answer half_hr = 5 Then '02:00:00'
When i.answer half_ hr 45 Then '22:00:00'
When i.answer half hr = 47 Then '23:00:00'
When i. answer half hr = 43 Then '21:00:00'
Wheri i.answer half_ hr = 15 Then '07:00:00'
When i.answer half hr = 40 Then '19:30:00'
When i,answer half hr = 7 Then '03:00:00'
When i.answer half hr = 1 Then '00:00:00'
When i. answer half_ hr =3 Then '01:00:00'
When i.answer half_ hr = 37 Then '18:00:00'
When i.answer half hr = 20 Then '09:30:00'
When i.answer half hr = 17 Then '08:00:00'
When i.answer half hr = 34 Then '16:30:00'
When i.answer hàlf_hr = 38 Then '18:30:00'
When i. answer_ half_hr = 36 Then '17:30:00'
When i. answer_half_hr = 35 Then '17:00:00'
ihen i.answer half hr 13 Then '06:00:00'
ihen i.answer_half hr = 19 Then '09:00:00'
When i.answer half hr = 14 Then '06:30:00'
When i.answer half hr = 42 Then '20:30:00'
End) AS TIME) AS interval_start,
CAST(TRIM(CASE
When i.answer_half_hr = 27 Then '13:30:00'
When i.answer half hr = 29 Then '14:30:00'
When i.answer half hr = 21 Then '10:30:00'
When i.answer half hr = 8 Then '04:00:00'
When i.answer_half_hr = 2 Then '01:00:00' 
When i.answer half hr = 25 Then '12:30:00'
When i.answer half hr =6 Then '03:00:00'
When i.answer half hr = 4 Then '02:00:00'
When i,answer half hr = 44 Then '22:00:00'
When i. answer half hr 46 Then '23:00:00'
When i, answer_ half hr 12 Then '06:00:00'
When i.answer half hr 48 Then '00:00:00'
When i.answer_ half_ hr 31 Then '15:30:00'
When i.answer half hr = 41 Then '20:30:00'
when i.answer half hr = 33 Then '16:30:00'
When i.answer half hr = 10 Then '05:00:00'
When i.answer half hr = 18 Then '09:00:00'
When i.answer half_hr = 22 Then '11:00:00'
When i.answer half hr =39 Then '19:30:00'
When i.answer half hr=24 Then '12:00:00'
When i. answer half hr=16 Then '08:00:00
When i. answer half hr30 Then '15:00:00'
When i.answer_half_hr, = 11 Then '05:30:00'
When i. answerhalf_hr = 9 Then '94:30:00'
ihen i.answerhalf hr 23 Then '11:30:00'
When i.answer half_ hr = 32 Then '16:00:08'
When i. answerhalf_hr = 26 Then '13:00:00'
When i.answer half hr = 28 Then '14:00:00'
When i. answerhalf_hr = 5 Then '02:30:00'
When i. answer_half_hr = 45 Then '22:30:90'
When i.answer half hr = 47 Then '23:30:00'
When i.answer half hr = 43 Then '21:30:00'
When i.answe half hr = 15 Then '07:30:00'
When i.answer half hr = 40 Then '20:00:00'
When i.answer half hr =7 Then '03:30:00'
When i.answer half hr 1 Then '00:30:00'
When i.answer half hr = 3 Then '01:30:00'
ihen i.answer half hr = 37 Then '18:30:00'
When i.answer half hr = 20 Then '10:00:00'
When i.answer half hr =17 Then '08:30:00'
When i.answer_ half hr = 34 Then '17:00:00'






