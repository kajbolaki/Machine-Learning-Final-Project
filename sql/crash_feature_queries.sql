-- SELECT example: raw volume by dataset
SELECT COUNT(*) AS crashes_rows FROM crashes_raw;
SELECT COUNT(*) AS vehicles_rows FROM vehicles_raw;
SELECT COUNT(*) AS people_rows FROM people_raw;

-- INSERT + JOIN example: build crash-level modeling table
DROP TABLE IF EXISTS crash_features;
CREATE TABLE crash_features (
    crash_record_id TEXT PRIMARY KEY,
    crash_date TIMESTAMP,
    posted_speed_limit DOUBLE PRECISION,
    traffic_control_device TEXT,
    intersection_related_i TEXT,
    hit_and_run_i TEXT,
    work_zone_i TEXT,
    weather_condition TEXT,
    lighting_condition TEXT,
    first_crash_type TEXT,
    trafficway_type TEXT,
    alignment TEXT,
    roadway_surface_cond TEXT,
    road_defect TEXT,
    prim_contributory_cause TEXT,
    sec_contributory_cause TEXT,
    crash_hour DOUBLE PRECISION,
    crash_day_of_week DOUBLE PRECISION,
    crash_month DOUBLE PRECISION,
    num_units DOUBLE PRECISION,
    injuries_fatal DOUBLE PRECISION,
    injuries_incapacitating DOUBLE PRECISION,
    injuries_non_incapacitating DOUBLE PRECISION,
    injuries_total DOUBLE PRECISION,
    vehicle_count DOUBLE PRECISION,
    towed_vehicle_count DOUBLE PRECISION,
    people_count DOUBLE PRECISION,
    driver_count DOUBLE PRECISION
);

INSERT INTO crash_features (
    crash_record_id, crash_date, posted_speed_limit, traffic_control_device, intersection_related_i,
    hit_and_run_i, work_zone_i, weather_condition,
    lighting_condition, first_crash_type, trafficway_type, alignment, roadway_surface_cond,
    road_defect, prim_contributory_cause, sec_contributory_cause, crash_hour, crash_day_of_week,
    crash_month, num_units, injuries_fatal, injuries_incapacitating, injuries_non_incapacitating,
    injuries_total, vehicle_count, towed_vehicle_count, people_count, driver_count
)
SELECT
    c.crash_record_id,
    NULLIF(c.crash_date::TEXT, '')::TIMESTAMP AS crash_date,
    NULLIF(c.posted_speed_limit::TEXT, '')::DOUBLE PRECISION AS posted_speed_limit,
    c.traffic_control_device,
    c.intersection_related_i,
    c.hit_and_run_i,
    c.work_zone_i,
    c.weather_condition,
    c.lighting_condition,
    c.first_crash_type,
    c.trafficway_type,
    c.alignment,
    c.roadway_surface_cond,
    c.road_defect,
    c.prim_contributory_cause,
    c.sec_contributory_cause,
    NULLIF(c.crash_hour::TEXT, '')::DOUBLE PRECISION AS crash_hour,
    NULLIF(c.crash_day_of_week::TEXT, '')::DOUBLE PRECISION AS crash_day_of_week,
    NULLIF(c.crash_month::TEXT, '')::DOUBLE PRECISION AS crash_month,
    NULLIF(c.num_units::TEXT, '')::DOUBLE PRECISION AS num_units,
    NULLIF(c.injuries_fatal::TEXT, '')::DOUBLE PRECISION AS injuries_fatal,
    NULLIF(c.injuries_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_incapacitating,
    NULLIF(c.injuries_non_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_non_incapacitating,
    NULLIF(c.injuries_total::TEXT, '')::DOUBLE PRECISION AS injuries_total,
    COALESCE(v.vehicle_count, 0) AS vehicle_count,
    COALESCE(v.towed_vehicle_count, 0) AS towed_vehicle_count,
    COALESCE(p.people_count, 0) AS people_count,
    COALESCE(p.driver_count, 0) AS driver_count
FROM crashes_raw c
LEFT JOIN (
    SELECT
        crash_record_id,
        COUNT(*)::DOUBLE PRECISION AS vehicle_count,
        SUM(CASE WHEN towed_i = 'Y' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS towed_vehicle_count
    FROM vehicles_raw
    GROUP BY crash_record_id
) v ON c.crash_record_id = v.crash_record_id
LEFT JOIN (
    SELECT
        crash_record_id,
        COUNT(*)::DOUBLE PRECISION AS people_count,
        SUM(CASE WHEN person_type = 'DRIVER' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS driver_count
    FROM people_raw
    GROUP BY crash_record_id
) p ON c.crash_record_id = p.crash_record_id
WHERE NULLIF(c.crash_date::TEXT, '')::TIMESTAMP >= :start_date;
