# Data sources

Sources are listed separately from scenario assumptions. Third-party files, personal applications, browser sessions and large simulation outputs are not included in this code release. Obtain data under each provider's terms.

## PLATEAU CityGML

- Version: 2025 releases; Chiyoda, Chuo and Minato.
- Study role: Buildings, terrain gap filling, surface roads and water polygons.
- Access: Provider download and source-specific terms.
- Code: `src/preprocessing/download_plateau_models.py`.

- https://www.geospatial.jp/ckan/dataset/plateau-13101-chiyoda-ku-2025
- https://www.geospatial.jp/ckan/dataset/plateau-13102-chuo-ku-2025
- https://www.geospatial.jp/ckan/dataset/plateau-13103-minato-ku-2025

## Tokyo wards airborne laser survey

- Version: 2024 catalog; native 0.50 m ground DEM.
- Study role: Primary bare-earth terrain; point-cloud and orthophoto context checks.
- Access: Provider download; catalog release year is not a certification of every survey epoch.
- Code: `src/preprocessing/acquire_three_ward_dem.py`.

- https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024
- https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024/resource/ff8305fd-a993-4535-80b4-fd6dcb339775
- https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024/resource/7807d6d1-29f3-4b36-b0c8-f7aa0ea2cff3
- https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024/resource/5f96771a-9929-499f-89de-1efa0b2be034
- https://catalog.data.metro.tokyo.lg.jp/dataset/t000029d0000000024

## JMA composite radar archived by Kyoto University RISH

- Version: Parameter 203; original GRIB2; 2024 and 2025 event windows.
- Study role: Historical rainfall forcing, including the separate geometry-reference parent.
- Access: RISH archive use conditions; JMA producer attribution.
- Code: `src/preprocessing/acquire_tokyo_historical_radar_v1.py`, `src/preprocessing/acquire_population_additional_radar_v1.py`.

- http://database.rish.kyoto-u.ac.jp/arch/jmadata/
- http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/
- https://www.data.jma.go.jp/suishin/jyouhou/pdf/568.pdf

## 2020 Population Census, e-Stat

- Version: 2020-10-01 population; JGD2011 boundaries.
- Study role: Official basic-unit nighttime population baseline; 125/250 m mesh allocation checks.
- Access: Public statistical aggregates; population stocks are not observed event-time travelers.
- Code: `src/preprocessing/download_estat_three_ward_population_v1.py`, `src/preprocessing/build_three_ward_population_block_stability_v1.py`.

- https://www.e-stat.go.jp/gis/statmap-search?page=1&type=1&toukeiCode=00200521
- https://www.stat.go.jp/data/mesh/pdf/gaiyo2.pdf
- https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf

## OpenStreetMap via Geofabrik

- Version: Kanto extract, 2026-10-15.
- Study role: Directed surface road network and tagged structures/access restrictions.
- Access: ODbL; retain OpenStreetMap contributor attribution; dated extracts may expire.
- Code: `src/preprocessing/build_normal_surface_network_v1.py`.

- https://download.geofabrik.de/asia/japan/kanto.html
- https://download.geofabrik.de/asia/japan/kanto-261015.osm.pbf
- https://www.openstreetmap.org/copyright

## MLIT National Land Numerical Information

- Version: N03 2024 administrative boundaries; W05 2008 Tokyo river lines.
- Study role: Regional context and river/waterbody matching; not surveyed river bathymetry.
- Access: Individual product conditions apply.
- Code: `src/exp/audit_ksj_tokyo.py`.

- https://nlftp.mlit.go.jp/ksj/
- https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html
- https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html
- https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-20240101/N03-20240101_13_GML.zip
- https://nlftp.mlit.go.jp/ksj/gml/data/W05/W05-08/W05-08_13_GML.zip

## MLIT National Road and Street Traffic Conditions Survey

- Version: 2021, Tokyo section and hourly tables.
- Study role: Normal-road traffic observations and section/geometry constraints.
- Access: Public survey tables; observation/estimation flags retained.
- Code: `src/preprocessing/inspect_tokyo_traffic_data_sources_v1.py`, `src/preprocessing/acquire_census_geometries_v1.py`.

- https://www.mlit.go.jp/road/census/r3/index.html
- https://www.mlit.go.jp/road/census/r3/data/csv/kasyo13.csv
- https://www.mlit.go.jp/road/census/r3/data/csv/zkntrf13.csv
- https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/webmap.html
- https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000028577

## Tokyo Metropolitan Area Person Trip Survey

- Version: 2018 (H30), tables b3/b4/c1/c2/c3/d1/e2 and 5_3.
- Study role: OD and temporal constraints for scenario vehicle demand.
- Access: Download/application and provider conditions apply; personal sessions are not distributed.
- Code: `src/preprocessing/download_personal_pt_tables_v1.cjs`, `src/preprocessing/decode_full_pt_tables_v1.py`.

- https://www.tokyo-pt.jp/data/H30
- https://www.tokyo-pt.jp/data/01download
- https://www.tokyo-pt.jp/data/01download/other
- https://www.tokyo-pt.jp/terms
- https://www.tokyo-pt.jp/static/hp/file/data/H30_zonecode.xlsx
- https://www.tokyo-pt.jp/static/hp/file/data/H30_gis.zip
- https://www.tokyo-pt.jp/info_provision/traffic

## Geospatial Information Authority of Japan tiles

- Version: Retained tiles at acquisition; not a replacement for the native Tokyo DEM.
- Study role: Regional contributing-domain screening and context.
- Access: GSI source conditions; templates use slippy-map tile indices.
- Code: `src/preprocessing/acquire_three_ward_dem.py`.

- https://maps.gsi.go.jp/development/ichiran.html
- https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png
- https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png

## Exact census downloads

Basic-unit boundaries include the official JINKO resident field. Use `B002005212020`; town boundaries use `A002005212020`. Ward codes are 13101, 13102 and 13103. These basic-unit totals are the final nighttime baseline. The mesh allocations are a separate exploratory check.

- https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=B002005212020&code=13101&coordSys=1&format=gml&downloadType=5&datum=2011
- https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=B002005212020&code=13102&coordSys=1&format=gml&downloadType=5&datum=2011
- https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=B002005212020&code=13103&coordSys=1&format=gml&downloadType=5&datum=2011

- 125 m mesh, T001231: https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001231&code=5339&downloadType=2
- 250 m mesh, T001142: https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001142&code=5339&downloadType=2

## Rainfall windows

| Parent | Window in Japan Standard Time | Role |
|---|---|---|
| 2024-06-18 | 07:45–16:45 | Historical baseline |
| 2024-07-31 | 11:45–20:45 | Baseline and interventions |
| 2024-08-30 | 02:00–11:00 | Baseline and interventions |
| 2025-07-10 | 12:50–21:50 | Historical baseline |
| 2025-09-11 | 13:15–22:15 | Baseline, interventions and traffic comparison |

The 2024-08-21 geometry-reference family is separate. Retrieve GRIB2 files from `.../YYYY/MM/DD/`; the scripts retain UTC source clocks and conservative interval mapping. A stage or intervention variant of one parent is not another observed storm.

## Supporting and exploratory resources

- Railway context (not part of the final vehicle-access endpoint): https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-2025.html
- MLIT pedestrian-network catalog: https://www.geospatial.jp/ckan/dataset/0401
- JMA station observations used for context: https://www.data.jma.go.jp/stats/etrn/index.php

River stages, intake locations, receiving allowances, closure thresholds, unobserved demand and minor-road capacity branches are scenario assumptions. They are not additional observed datasets. Source buildings and terrain describe their own survey epochs, not a common survey at every storm date.
