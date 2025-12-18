# Data Directory

This directory contains the data used in the project.

## Raw Data

The `raw/` directory contains the original dataset used in the analysis:
- **Seoul Bike Sharing Demand** dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- Period: 1 December 2017 – 30 November 2018
- Frequency: hourly observations (8,760 rows)
- Variables: 14 variables in total, including the target variable (hourly rented bike count),
  weather-related variables (e.g. temperature, humidity, visibility, wind speed, rainfall,
  snowfall, solar radiation, and dew point temperature), temporal variables (e.g. date, hour,
  season), and binary indicators (e.g. holiday, functioning day).

The raw data contains no missing values. 

## Processed Data

Cleaned and processed datasets are generated automatically after EDA and model training
and saved locally as `.parquet` files during execution.

To avoid tracking large derived files and to ensure reproducibility, processed data are
not committed to version control and are excluded via `.gitignore`.

All processed data can be fully reproduced by running the preprocessing and modelling
pipelines provided in the repository.

## Notes

This separation between raw and processed data follows best practices for reproducible
data science workflows, ensuring a clear distinction between original inputs and
derived artefacts.
