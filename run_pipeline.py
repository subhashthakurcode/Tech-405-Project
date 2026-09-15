"""
run_pipeline.py
End-to-End Orchestrator for Weather Prediction EDA, Feature Engineering, and Figure Generation.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from src.data_loader import WeatherDataLoader
from src.eda_analysis import TimeSeriesEDA
from src.feature_engineering import WeatherFeatureEngineer
from src.visualization import WeatherVisualizer


def main():
    print("=" * 80)
    print("EUROPEAN WEATHER PREDICTION: EDA & FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # 1. Dataset Ingestion & Dataset Card
    loader = WeatherDataLoader(data_dir="data")
    card_info = loader.get_dataset_card_info()
    
    print("\n--- [1] DATASET CARD & METADATA ---")
    for k, v in card_info.items():
        print(f"  * {k:22s}: {v}")
        
    df_raw = loader.load_raw_data()
    df_picnic = loader.load_picnic_labels()
    df_merged = loader.load_combined_data()
    
    print(f"\nLoaded Main Weather Dataset: Shape = {df_raw.shape}")
    print(f"Loaded Picnic Labels:        Shape = {df_picnic.shape}")
    print(f"Merged Dataset:              Shape = {df_merged.shape}")
    
    print("\n--- [2] FILE STRUCTURE & FIVE RANDOM SAMPLES ---")
    print("Data Types Summary (First 10 columns):")
    for col in df_merged.columns[:10]:
        print(f"  - {col:25s}: {df_merged[col].dtype}")
        
    print("\nFive Random Samples from Merged Dataset:")
    sample_cols = ["DATE_DT", "BASEL_temp_mean", "BASEL_pressure", "BASEL_humidity", "BASEL_precipitation", "BASEL_picnic_weather", "HEATHROW_temp_mean", "ROMA_temp_mean"]
    sample_cols_clean = [c for c in sample_cols if c in df_merged.columns]
    print(df_merged[sample_cols_clean].sample(5, random_state=42).to_string(index=False))

    # 2. Data Integrity Checks
    print("\n--- [3] MISSING, DUPLICATE, AND CORRUPT ENTRIES VERIFICATION ---")
    integrity = loader.verify_data_integrity(df_merged)
    for k, v in integrity.items():
        print(f"  * {k:25s}: {v}")

    # 3. Statistical Analysis & Outliers
    print("\n--- [4] STATISTICAL PROPERTIES & OUTLIER DETECTION ---")
    eda = TimeSeriesEDA()
    basel_temp = df_merged["BASEL_temp_mean"]
    outliers = eda.detect_outliers_iqr(basel_temp)
    print(f"Basel Mean Temp Outlier Profiling (IQR method):")
    print(f"  * Q25: {outliers['q25']:.2f}°C, Q75: {outliers['q75']:.2f}°C, IQR: {outliers['iqr']:.2f}°C")
    print(f"  * Normal Bounds: [{outliers['lower_bound']:.2f}°C, {outliers['upper_bound']:.2f}°C]")
    print(f"  * Detected Outliers: {outliers['outlier_count']} samples ({outliers['outlier_percentage']:.2f}%)")

    # 4. Seasonal Decomposition, ACF, PACF, & ADF Test
    print("\n--- [5] TIME-SERIES DECOMPOSITION & STATIONARITY (ADF TEST) ---")
    adf_raw = eda.perform_adf_test(basel_temp)
    print(f"Raw Series ADF Test:")
    print(f"  * ADF Statistic:      {adf_raw['adf_statistic']:.4f}")
    print(f"  * p-value:            {adf_raw['p_value']:.4f}")
    print(f"  * Lags Used:          {adf_raw['lags_used']}")
    print(f"  * Critical Values:    {adf_raw['critical_values']}")
    print(f"  * Conclusion:         {adf_raw['interpretation']}")

    diff_series = basel_temp.diff().dropna()
    adf_diff = eda.perform_adf_test(diff_series)
    print(f"\nFirst-Differenced Series (ΔT) ADF Test:")
    print(f"  * ADF Statistic:      {adf_diff['adf_statistic']:.4f}")
    print(f"  * p-value:            {adf_diff['p_value']:.2e}")
    print(f"  * Conclusion:         {adf_diff['interpretation']}")

    # 5. ACF & PACF and LSTM Window Decision
    print("\n--- [6] ACF / PACF & LSTM WINDOW DECISION ---")
    acf_vals, pacf_vals = eda.analyze_autocorrelation(basel_temp, nlags=60)
    window_decision = eda.decide_lstm_window_length(acf_vals, pacf_vals)
    print(f"  * Recommended Window Size W: {window_decision['recommended_window_days']} days (2 synoptic weeks)")
    print(f"  * Alternate Window Size:     {window_decision['alternate_window_days']} days")
    print(f"  * Justification:            {window_decision['justification']}")

    # 6. Feature Engineering
    print("\n--- [7] FEATURE ENGINEERING & TARGET CREATION ---")
    engineer = WeatherFeatureEngineer(target_station="BASEL")
    df_engineered = engineer.build_full_feature_pipeline(df_merged)
    print(f"Engineered Dataset Shape: {df_engineered.shape} (after temporal expansion & null dropping)")

    # 7. Chronological Split & Leakage-Free Normalization
    print("\n--- [8] CHRONOLOGICAL SPLIT & NORMALIZATION PIPELINE ---")
    train_df, val_df, test_df = engineer.temporal_train_val_test_split(df_engineered)
    print(f"  * Train Set (2000–2006):      {len(train_df)} days ({len(train_df)/len(df_engineered)*100:.1f}%)")
    print(f"  * Validation Set (2007–2008): {len(val_df)} days ({len(val_df)/len(df_engineered)*100:.1f}%)")
    print(f"  * Test Set (2009–2010):       {len(test_df)} days ({len(test_df)/len(df_engineered)*100:.1f}%)")

    X_train, X_val, X_test, feat_cols = engineer.fit_transform_features(train_df, val_df, test_df)
    y_train = train_df["target_temp_mean_next_day"].values
    y_val = val_df["target_temp_mean_next_day"].values
    y_test = test_df["target_temp_mean_next_day"].values

    print(f"  * Feature Count (scaled):     {len(feat_cols)}")
    print(f"  * X_train Shape:              {X_train.shape}")
    print(f"  * X_val Shape:                {X_val.shape}")
    print(f"  * X_test Shape:               {X_test.shape}")

    # 8. PyTorch 3D Sequence Tensor Construction
    print("\n--- [9] PYTORCH 3D SEQUENCE TENSOR CREATION ---")
    W = window_decision["recommended_window_days"]
    X_train_seq, y_train_seq = engineer.create_lstm_sequences(X_train, y_train, window_size=W)
    X_val_seq, y_val_seq = engineer.create_lstm_sequences(X_val, y_val, window_size=W)
    X_test_seq, y_test_seq = engineer.create_lstm_sequences(X_test, y_test, window_size=W)

    print(f"  * Sequence Lookback Window W: {W} days")
    print(f"  * X_train_seq Tensor Shape:   {X_train_seq.shape} (N_samples, Window, Features)")
    print(f"  * X_val_seq Tensor Shape:     {X_val_seq.shape}")
    print(f"  * X_test_seq Tensor Shape:    {X_test_seq.shape}")

    # Verify PyTorch Tensor Conversion
    t_X_train = torch.tensor(X_train_seq, dtype=torch.float32)
    t_y_train = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)
    print(f"  * PyTorch Tensor Ready:       X={t_X_train.shape}, y={t_y_train.shape}")

    # 9. Visualization & Snapshot Generation
    print("\n--- [10] GENERATING HIGH-RESOLUTION PUBLICATION FIGURES ---")
    viz = WeatherVisualizer(output_dir="figures")
    
    fig1 = viz.plot_01_temperature_distributions(df_merged)
    print(f"  [+] Generated: {fig1}")
    
    fig2 = viz.plot_02_full_series_and_outliers(df_merged)
    print(f"  [+] Generated: {fig2}")
    
    fig3 = viz.plot_03_seasonal_decomposition(df_merged)
    print(f"  [+] Generated: {fig3}")
    
    fig4 = viz.plot_04_autocorrelation_acf_pacf(df_merged)
    print(f"  [+] Generated: {fig4}")
    
    fig5 = viz.plot_05_stationarity_and_transformation(df_merged, adf_raw, adf_diff)
    print(f"  [+] Generated: {fig5}")
    
    fig6 = viz.plot_06_cross_station_correlation_heatmap(df_merged)
    print(f"  [+] Generated: {fig6}")
    
    fig7 = viz.plot_07_inter_variable_relationships(df_merged)
    print(f"  [+] Generated: {fig7}")
    
    fig8 = viz.plot_08_cyclical_and_lag_features(df_engineered)
    print(f"  [+] Generated: {fig8}")
    
    fig9 = viz.plot_09_target_distributions_and_imbalance(df_engineered)
    print(f"  [+] Generated: {fig9}")
    
    fig10 = viz.plot_10_chronological_splits_and_nn_scaling(train_df, val_df, test_df, X_train)
    print(f"  [+] Generated: {fig10}")

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES AND FIGURE GENERATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
