"""
data_loader.py
Data Ingestion, Verification, and Integrity Validation for European Weather Prediction Dataset.
"""

import os
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class WeatherDataLoader:
    """Loads and validates the European Weather Prediction Dataset."""
    
    STATIONS = [
        "BASEL", "BUDAPEST", "DE_BILT", "DRESDEN", "DUSSELDORF",
        "HEATHROW", "KASSEL", "LJUBLJANA", "MAASTRICHT", "MALMO",
        "MONTELIMAR", "MUENCHEN", "OSLO", "PERPIGNAN", "ROMA",
        "SONNBLICK", "STOCKHOLM", "TOURS"
    ]
    
    VARIABLES = [
        "cloud_cover", "humidity", "pressure", "global_radiation",
        "precipitation", "sunshine", "temp_mean", "temp_min",
        "temp_max", "wind_gust", "wind_speed"
    ]
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.main_path = os.path.join(data_dir, "weather_prediction_dataset.csv")
        self.light_path = os.path.join(data_dir, "weather_prediction_dataset_light.csv")
        self.picnic_path = os.path.join(data_dir, "weather_prediction_picnic_labels.csv")
        self.metadata_path = os.path.join(data_dir, "metadata.txt")

    def load_raw_data(self) -> pd.DataFrame:
        """Loads the main tabular weather dataset with parsed datetime index."""
        if not os.path.exists(self.main_path):
            raise FileNotFoundError(f"Main dataset not found at {self.main_path}")
        
        df = pd.read_csv(self.main_path)
        # Parse DATE string format YYYYMMDD to datetime
        df["DATE_DT"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d")
        df = df.sort_values("DATE_DT").reset_index(drop=True)
        return df

    def load_picnic_labels(self) -> pd.DataFrame:
        """Loads picnic weather boolean labels."""
        if not os.path.exists(self.picnic_path):
            raise FileNotFoundError(f"Picnic dataset not found at {self.picnic_path}")
        
        df_picnic = pd.read_csv(self.picnic_path)
        df_picnic["DATE_DT"] = pd.to_datetime(df_picnic["DATE"].astype(str), format="%Y%m%d")
        df_picnic = df_picnic.sort_values("DATE_DT").reset_index(drop=True)
        return df_picnic

    def load_combined_data(self) -> pd.DataFrame:
        """Merges main weather observations with picnic labels."""
        df_weather = self.load_raw_data()
        df_picnic = self.load_picnic_labels()
        
        # Merge on DATE / DATE_DT
        merged = pd.merge(df_weather, df_picnic, on=["DATE", "DATE_DT"], suffixes=("", "_picnic"))
        return merged

    def verify_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive verification of frequency, continuity, missingness, and corrupt tokens."""
        # 1. Date Continuity & Frequency Check
        date_series = pd.Series(df["DATE_DT"].values)
        min_date = df["DATE_DT"].min()
        max_date = df["DATE_DT"].max()
        total_days = (max_date - min_date).days + 1
        expected_range = pd.date_range(start=min_date, end=max_date, freq="D")
        missing_dates = expected_range.difference(df["DATE_DT"])
        
        # 2. Duplicate Rows Check
        duplicate_count = df.duplicated(subset=["DATE"]).sum()
        
        # 3. Missing (NaN / null) values
        null_count = df.isna().sum().sum()
        
        # 4. Corrupt entries / placeholder check (e.g. -9999 or -999.9)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corrupt_neg9999 = (df[numeric_cols] == -9999).sum().sum()
        corrupt_neg999 = (df[numeric_cols] == -999).sum().sum()
        
        # 5. Summary Report
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "start_date": str(min_date.date()),
            "end_date": str(max_date.date()),
            "expected_days_count": total_days,
            "actual_days_count": len(df),
            "date_gaps_count": len(missing_dates),
            "is_regular_daily": len(missing_dates) == 0,
            "duplicate_rows": int(duplicate_count),
            "null_values": int(null_count),
            "corrupt_neg9999_count": int(corrupt_neg9999),
            "corrupt_neg999_count": int(corrupt_neg999),
            "numeric_columns_count": len(numeric_cols),
        }
        return report

    def get_station_columns(self, station: str, df: pd.DataFrame) -> List[str]:
        """Returns all feature columns belonging to a specific European station."""
        prefix = f"{station.upper()}_"
        return [col for col in df.columns if col.startswith(prefix)]

    def get_dataset_card_info(self) -> Dict[str, str]:
        """Returns structured metadata card info for the dataset."""
        return {
            "Dataset Name": "European Weather Prediction Dataset",
            "Primary Source": "European Climate Assessment & Dataset (ECA&D), Klein Tank et al., 2002",
            "Zenodo DOI": "https://doi.org/10.5281/zenodo.7053722",
            "License": "Open Access / Academic Research Use (CC-BY 4.0 compliant with citation)",
            "Temporal Coverage": "2000-01-01 to 2010-01-01 (10 full years + 1 day = 3,654 consecutive days)",
            "Sampling Frequency": "Daily (regular 24h aggregated observations)",
            "Spatial Coverage": "18 European meteorological stations across 9 countries",
            "Variables": "11 physical atmospheric measurements (Mean/Min/Max Temp, Radiation, Sunshine, Cloud Cover, Humidity, Pressure, Precipitation, Wind)",
            "Total Features": "165 tabular feature columns + 18 binary picnic labels",
            "Disk Size": "Main CSV: ~2.77 MB; Light CSV: ~1.54 MB; Picnic Labels: ~394 KB; Total: ~4.7 MB",
            "Data Quality": "Cleaned & preprocessed; columns with >5% missing dropped, <=5% imputed with feature means; transformed into intuitive metric units.",
            "Known Limitations": "Daily temporal aggregation (diurnal micro-cycles masked); slight spatial sparsity (not all 18 stations record all 11 variables); mean imputation smooths extreme missing tail values."
        }
