"""
feature_engineering.py
Feature Engineering, Normalization, Data Leakage Prevention, and Sequence Creation for Neural Networks.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class WeatherFeatureEngineer:
    """Performs end-to-end feature engineering and sequence extraction for Neural Networks."""

    def __init__(self, target_station: str = "BASEL"):
        self.target_station = target_station
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_calendar_and_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes date into year, month, day, dayofweek, dayofyear,
        and applies continuous cyclical sine/cosine transformations.
        """
        df_out = df.copy()
        dt_col = df_out["DATE_DT"]
        
        df_out["year"] = dt_col.dt.year
        df_out["month"] = dt_col.dt.month
        df_out["day"] = dt_col.dt.day
        df_out["dayofweek"] = dt_col.dt.dayofweek
        df_out["dayofyear"] = dt_col.dt.dayofyear
        
        # Continuous cyclical transforms (prevents discontinuity at boundary e.g. Dec 31 -> Jan 1)
        df_out["sin_dayofyear"] = np.sin(2 * np.pi * df_out["dayofyear"] / 365.25)
        df_out["cos_dayofyear"] = np.cos(2 * np.pi * df_out["dayofyear"] / 365.25)
        df_out["sin_month"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
        df_out["cos_month"] = np.cos(2 * np.pi * df_out["month"] / 12.0)
        df_out["sin_dayofweek"] = np.sin(2 * np.pi * df_out["dayofweek"] / 7.0)
        df_out["cos_dayofweek"] = np.cos(2 * np.pi * df_out["dayofweek"] / 7.0)
        
        return df_out

    def create_lag_features(
        self,
        df: pd.DataFrame,
        variables: List[str] = None,
        lags: List[int] = [1, 2, 3, 7]
    ) -> pd.DataFrame:
        """Generates historical lag features for autoregressive neural modeling."""
        df_out = df.copy()
        if variables is None:
            # Key atmospheric variables for the target station and upstream stations
            variables = [
                f"{self.target_station}_temp_mean",
                f"{self.target_station}_pressure",
                f"{self.target_station}_humidity",
                f"{self.target_station}_global_radiation",
                "HEATHROW_temp_mean", "HEATHROW_pressure",
                "TOURS_temp_mean", "TOURS_pressure",
                "DE_BILT_temp_mean", "DE_BILT_pressure"
            ]
        
        for col in variables:
            if col in df_out.columns:
                for lag in lags:
                    df_out[f"{col}_lag_{lag}"] = df_out[col].shift(lag)
                    
        return df_out

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        variables: List[str] = None,
        windows: List[int] = [3, 7, 14, 30]
    ) -> pd.DataFrame:
        """Calculates multi-scale rolling mean, std (volatility), min, and max."""
        df_out = df.copy()
        if variables is None:
            variables = [
                f"{self.target_station}_temp_mean",
                f"{self.target_station}_pressure",
                f"{self.target_station}_humidity"
            ]
            
        for col in variables:
            if col in df_out.columns:
                for w in windows:
                    # Use shift(1) to avoid including current/future values in rolling window
                    rolled = df_out[col].shift(1).rolling(window=w)
                    df_out[f"{col}_roll_mean_{w}d"] = rolled.mean()
                    df_out[f"{col}_roll_std_{w}d"] = rolled.std()
                    df_out[f"{col}_roll_min_{w}d"] = rolled.min()
                    df_out[f"{col}_roll_max_{w}d"] = rolled.max()
                    
        return df_out

    def create_spatial_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes spatial pressure differentials and temperature gradients across Europe.
        Models moving atmospheric fronts (e.g. Atlantic storms traveling Heathrow -> Tours -> Basel).
        """
        df_out = df.copy()
        
        # Pressure differentials relative to upstream stations
        if f"{self.target_station}_pressure" in df_out.columns:
            target_p = df_out[f"{self.target_station}_pressure"]
            for upstream in ["HEATHROW", "TOURS", "DE_BILT", "MONTELIMAR"]:
                up_col = f"{upstream}_pressure"
                if up_col in df_out.columns:
                    df_out[f"delta_pressure_{self.target_station}_vs_{upstream}"] = target_p - df_out[up_col]
                    
        # Temperature gradients relative to upstream stations
        if f"{self.target_station}_temp_mean" in df_out.columns:
            target_t = df_out[f"{self.target_station}_temp_mean"]
            for upstream in ["HEATHROW", "TOURS", "DE_BILT", "MONTELIMAR"]:
                up_col = f"{upstream}_temp_mean"
                if up_col in df_out.columns:
                    df_out[f"delta_temp_{self.target_station}_vs_{upstream}"] = target_t - df_out[up_col]
                    
        return df_out

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs prediction targets for both regression and classification.
        - Continuous Target: Next-day mean temperature (t+1)
        - Continuous Target: Next-day max temperature (t+1)
        - Binary Target: Next-day picnic weather suitability (t+1)
        - Binary Target: Next-day precipitation event (rain > 0.1mm) (t+1)
        """
        df_out = df.copy()
        
        # Regression targets (t+1)
        target_temp_col = f"{self.target_station}_temp_mean"
        if target_temp_col in df_out.columns:
            df_out["target_temp_mean_next_day"] = df_out[target_temp_col].shift(-1)
            
        target_temp_max_col = f"{self.target_station}_temp_max"
        if target_temp_max_col in df_out.columns:
            df_out["target_temp_max_next_day"] = df_out[target_temp_max_col].shift(-1)
            
        # Binary Classification targets (t+1)
        picnic_col = f"{self.target_station}_picnic_weather"
        if picnic_col in df_out.columns:
            df_out["target_picnic_next_day"] = df_out[picnic_col].shift(-1).astype(float)
            
        precip_col = f"{self.target_station}_precipitation"
        if precip_col in df_out.columns:
            df_out["target_rain_next_day"] = (df_out[precip_col].shift(-1) > 0.01).astype(float)
            
        return df_out

    def build_full_feature_pipeline(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Executes full feature engineering transformation chain."""
        df_feat = self.create_calendar_and_cyclical_features(df_merged)
        df_feat = self.create_lag_features(df_feat)
        df_feat = self.create_rolling_features(df_feat)
        df_feat = self.create_spatial_differential_features(df_feat)
        df_feat = self.create_target_variables(df_feat)
        
        # Drop rows with NaN resulting from shifts/lags and target horizon (first 30 days and last 1 day)
        df_clean = df_feat.dropna().reset_index(drop=True)
        return df_clean

    def temporal_train_val_test_split(
        self,
        df_engineered: pd.DataFrame,
        train_end_year: int = 2006,
        val_end_year: int = 2008
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs strict chronological time-series splitting to prevent temporal data leakage.
        - Train Set: 2000 to train_end_year (inclusive, 2000-2006, ~70%)
        - Validation Set: (train_end_year + 1) to val_end_year (inclusive, 2007-2008, ~20%)
        - Test Set: (val_end_year + 1) onwards (2009-2010, ~10%)
        """
        train_df = df_engineered[df_engineered["year"] <= train_end_year].copy()
        val_df = df_engineered[(df_engineered["year"] > train_end_year) & (df_engineered["year"] <= val_end_year)].copy()
        test_df = df_engineered[df_engineered["year"] > val_end_year].copy()
        
        return train_df, val_df, test_df

    def fit_transform_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Fits StandardScaler strictly on the training partition and transforms
        validation and test sets without lookahead leakage.
        """
        if exclude_cols is None:
            exclude_cols = [
                "DATE", "DATE_DT", "year", "month", "day", "dayofweek", "dayofyear",
                "target_temp_mean_next_day", "target_temp_max_next_day",
                "target_picnic_next_day", "target_rain_next_day"
            ]
            # Also exclude picnic boolean columns from feature matrix if present
            exclude_cols += [c for c in train_df.columns if c.endswith("_picnic_weather")]

        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        self.feature_names = feature_cols

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_val = scaler.transform(val_df[feature_cols])
        X_test = scaler.transform(test_df[feature_cols])

        self.scaler = scaler
        return X_train, X_val, X_test, feature_cols

    @staticmethod
    def create_lstm_sequences(
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs sliding window 3D tensors (N, window_size, num_features)
        and target vectors (N,) for recurrent neural network (LSTM/GRU) training.
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i : i + window_size])
            y_seq.append(y[i + window_size])
        return np.array(X_seq), np.array(y_seq)
