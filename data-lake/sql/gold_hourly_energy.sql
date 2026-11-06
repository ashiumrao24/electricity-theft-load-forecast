-- One row per hour across the whole date range. The spine keeps missing hours visible:
-- an hour with no readings has NULL energy, never 0, so outages cannot look like zero use.
WITH bounds AS (
    SELECT date_trunc('HOUR', MIN(reading_ts)) AS first_hour,
           date_trunc('HOUR', MAX(reading_ts)) AS last_hour
    FROM silver_readings
),
spine AS (
    SELECT explode(sequence(first_hour, last_hour, INTERVAL 1 HOUR)) AS hour_ts
    FROM bounds
),
hourly AS (
    SELECT date_trunc('HOUR', reading_ts) AS hour_ts,
           COUNT(*) AS readings,
           SUM(active_power_kw) / 60 AS energy_kwh_observed,
           AVG(active_power_kw) AS energy_kwh_estimated,
           MAX(active_power_kw) AS peak_kw,
           AVG(voltage_v) AS avg_voltage_v
    FROM silver_readings
    GROUP BY date_trunc('HOUR', reading_ts)
)
SELECT s.hour_ts,
       COALESCE(h.readings, 0) AS readings,
       ROUND(COALESCE(h.readings, 0) / 60, 4) AS coverage,
       COALESCE(h.readings, 0) = 60 AS is_complete,
       h.energy_kwh_observed,
       h.energy_kwh_estimated,
       h.peak_kw,
       h.avg_voltage_v,
       year(s.hour_ts) AS year
FROM spine s
LEFT JOIN hourly h ON s.hour_ts = h.hour_ts
