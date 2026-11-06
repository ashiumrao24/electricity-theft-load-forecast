-- Month-level KPIs. Average daily energy is used because months differ in length and coverage.
WITH monthly AS (
    SELECT date_trunc('MONTH', usage_date) AS month_start,
           SUM(energy_kwh) AS energy_kwh,
           SUM(energy_kwh) / (SUM(coverage)) AS avg_daily_kwh,
           AVG(coverage) AS avg_coverage,
           MAX(peak_kw) AS peak_kw,
           SUM(water_heater_ac_kwh) / SUM(energy_kwh) AS water_heater_ac_share
    FROM gold_daily_usage
    GROUP BY date_trunc('MONTH', usage_date)
)
SELECT CAST(month_start AS DATE) AS month_start,
       ROUND(energy_kwh, 2) AS energy_kwh,
       ROUND(avg_daily_kwh, 2) AS avg_daily_kwh,
       ROUND(avg_coverage, 4) AS avg_coverage,
       ROUND(peak_kw, 3) AS peak_kw,
       ROUND(100 * water_heater_ac_share, 2) AS water_heater_ac_pct,
       ROUND(AVG(avg_daily_kwh) OVER (ORDER BY month_start
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS rolling_3_month_avg_daily_kwh,
       ROUND(100 * (avg_daily_kwh / LAG(avg_daily_kwh, 12) OVER (ORDER BY month_start) - 1), 2)
           AS yoy_change_pct,
       year(month_start) AS year
FROM monthly
