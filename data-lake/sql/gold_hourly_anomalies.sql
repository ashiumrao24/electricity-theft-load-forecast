-- Flags hours far from their own baseline: the same weekday and hour over the 8 prior weeks.
-- The frame ends at 1 PRECEDING, so an hour is never compared with itself or the future.
WITH complete_hours AS (
    SELECT hour_ts, energy_kwh_observed AS energy_kwh
    FROM gold_hourly_energy
    WHERE is_complete
),
baseline AS (
    SELECT hour_ts,
           energy_kwh,
           AVG(energy_kwh) OVER w AS baseline_kwh,
           STDDEV_SAMP(energy_kwh) OVER w AS baseline_std,
           COUNT(energy_kwh) OVER w AS baseline_hours
    FROM complete_hours
    WINDOW w AS (
        PARTITION BY dayofweek(hour_ts), hour(hour_ts)
        ORDER BY hour_ts
        ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
    )
)
SELECT hour_ts,
       ROUND(energy_kwh, 4) AS energy_kwh,
       ROUND(baseline_kwh, 4) AS baseline_kwh,
       ROUND((energy_kwh - baseline_kwh) / baseline_std, 2) AS z_score,
       CASE WHEN energy_kwh > baseline_kwh THEN 'spike' ELSE 'drop' END AS direction,
       year(hour_ts) AS year
FROM baseline
WHERE baseline_hours = 8
  AND baseline_std > 0
  AND ABS(energy_kwh - baseline_kwh) / baseline_std >= 3
