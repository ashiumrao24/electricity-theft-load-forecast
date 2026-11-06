-- Gaps and islands: every stretch of missing minutes, longest first in the report.
-- The window has no PARTITION BY because a gap can cross any boundary. Reading the
-- timestamp column alone keeps that single-partition step small at this data size.
WITH ordered AS (
    SELECT reading_ts,
           LAG(reading_ts) OVER (ORDER BY reading_ts) AS previous_ts
    FROM silver_readings
)
SELECT previous_ts + INTERVAL 1 MINUTE AS gap_start,
       reading_ts - INTERVAL 1 MINUTE AS gap_end,
       CAST((unix_timestamp(reading_ts) - unix_timestamp(previous_ts)) / 60 AS INT) - 1
           AS missing_minutes
FROM ordered
WHERE unix_timestamp(reading_ts) - unix_timestamp(previous_ts) > 60
