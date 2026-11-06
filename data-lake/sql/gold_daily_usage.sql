-- Daily energy split by sub-meter. Sub-meters are in Wh per minute, active power in kW.
SELECT CAST(reading_ts AS DATE) AS usage_date,
       COUNT(*) AS readings,
       ROUND(COUNT(*) / 1440, 4) AS coverage,
       SUM(active_power_kw) / 60 AS energy_kwh,
       SUM(kitchen_wh) / 1000 AS kitchen_kwh,
       SUM(laundry_wh) / 1000 AS laundry_kwh,
       SUM(water_heater_ac_wh) / 1000 AS water_heater_ac_kwh,
       SUM(active_power_kw) / 60
           - SUM(kitchen_wh + laundry_wh + water_heater_ac_wh) / 1000 AS other_kwh,
       MAX(active_power_kw) AS peak_kw,
       year(CAST(reading_ts AS DATE)) AS year
FROM silver_readings
GROUP BY CAST(reading_ts AS DATE)
