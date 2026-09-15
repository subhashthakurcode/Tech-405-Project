"""
visualization.py
High-Resolution Publication-Quality Visualization Suite for Weather Prediction EDA and Feature Engineering.
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Set publication style aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14


class WeatherVisualizer:
    """Generates and persists high-resolution analytical figures for the Weather Prediction dataset."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_01_temperature_distributions(self, df: pd.DataFrame) -> str:
        """Plot 1: KDE & Histograms of temperatures across diverse European climates."""
        stations = ["BASEL", "HEATHROW", "ROMA", "OSLO", "SONNBLICK", "BUDAPEST"]
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#e6ab02"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
        axes = axes.flatten()
        
        for idx, (st, col) in enumerate(zip(stations, colors)):
            col_name = f"{st}_temp_mean"
            if col_name in df.columns:
                data = df[col_name].dropna()
                ax = axes[idx]
                sns.histplot(data, kde=True, ax=ax, color=col, stat="density", bins=30, alpha=0.45)
                ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {data.mean():.1f}°C")
                ax.axvline(data.median(), color="red", linestyle=":", linewidth=1.5, label=f"Median: {data.median():.1f}°C")
                ax.set_title(f"{st.title()} (Daily Mean Temp)", fontweight="bold")
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Density" if idx % 3 == 0 else "")
                ax.legend(loc="upper right", frameon=True)
                ax.grid(True, alpha=0.3)
                
        plt.suptitle("Figure 1: Multi-Station Temperature Distributions Across European Climate Zones", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "01_temperature_distributions.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_02_full_series_and_outliers(self, df: pd.DataFrame) -> str:
        """Plot 2: 10-Year Continuous Time Series, frequency verification, and Outlier Boxplots."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        
        # 1. Top Panel: Full 10-Year Time Series
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.plot(df["DATE_DT"], df["BASEL_temp_mean"], color="#1f77b4", alpha=0.7, label="Basel Daily Mean Temp (°C)", linewidth=0.8)
        # 30-day rolling mean
        roll30 = df["BASEL_temp_mean"].rolling(30, center=True).mean()
        ax_top.plot(df["DATE_DT"], roll30, color="#d62728", linewidth=2.0, label="30-Day Moving Average")
        ax_top.set_title("Full 10-Year Continuous Time Series (2000–2010): Basel Daily Mean Temperature", fontweight="bold")
        ax_top.set_ylabel("Temperature (°C)")
        ax_top.set_xlabel("Date (Regular Daily Frequency, N = 3,654 Steps, 0 Missing Gaps)")
        ax_top.legend(loc="upper right", frameon=True)
        ax_top.grid(True, alpha=0.3)
        
        # 2. Bottom Left: Multi-Variable Boxplots for Outlier Detection
        ax_bl = fig.add_subplot(gs[1, 0])
        box_vars = ["BASEL_temp_mean", "BASEL_temp_max", "BASEL_temp_min", "HEATHROW_temp_mean", "ROMA_temp_mean", "SONNBLICK_temp_mean"]
        labels = ["Basel Mean", "Basel Max", "Basel Min", "Heathrow Mean", "Roma Mean", "Sonnblick (Alpine)"]
        sns.boxplot(data=df[box_vars], ax=ax_bl, palette="Set2", fliersize=3)
        ax_bl.set_xticks(range(len(labels)))
        ax_bl.set_xticklabels(labels, rotation=25, ha="right")
        ax_bl.set_title("Temperature Variations & Extreme Outliers Across Stations", fontweight="bold")
        ax_bl.set_ylabel("Temperature (°C)")
        ax_bl.grid(True, alpha=0.3)
        
        # 3. Bottom Right: Precipitation & Radiation Extreme Spikes
        ax_br = fig.add_subplot(gs[1, 1])
        precip_vars = ["BASEL_precipitation", "HEATHROW_precipitation", "DE_BILT_precipitation", "TOURS_precipitation"]
        precip_labels = ["Basel Precip", "Heathrow Precip", "De Bilt Precip", "Tours Precip"]
        sns.boxplot(data=df[precip_vars], ax=ax_br, palette="Blues_r", fliersize=3)
        ax_br.set_xticks(range(len(precip_labels)))
        ax_br.set_xticklabels(precip_labels, rotation=20, ha="right")
        ax_br.set_title("Precipitation Positive Skewness & Heavy-Tail Spikes (in 10 mm)", fontweight="bold")
        ax_br.set_ylabel("Precipitation (10 mm)")
        ax_br.grid(True, alpha=0.3)
        
        plt.suptitle("Figure 2: Full Time-Series Continuity, Regularity, and Outlier Profiling", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "02_full_series_and_outliers.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_03_seasonal_decomposition(self, df: pd.DataFrame) -> str:
        """Plot 3: Additive Seasonal Decomposition (Observed, Trend, Seasonal, Residuals)."""
        series = df.set_index("DATE_DT")["BASEL_temp_mean"].dropna()
        decomp = seasonal_decompose(series, model="additive", period=365)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
        
        # Observed
        axes[0].plot(decomp.observed.index, decomp.observed.values, color="#1f77b4", linewidth=1.0)
        axes[0].set_ylabel("Observed (°C)", fontweight="bold")
        axes[0].set_title("Additive Time-Series Decomposition of Basel Mean Temperature (Period = 365 Days)", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomp.trend.index, decomp.trend.values, color="#d62728", linewidth=1.8)
        axes[1].set_ylabel("Trend (°C)", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="#2ca02c", linewidth=1.0)
        axes[2].set_ylabel("Seasonal (°C)", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        
        # Residuals
        axes[3].scatter(decomp.resid.index, decomp.resid.values, color="#9467bd", alpha=0.5, s=8)
        axes[3].axhline(0, color="black", linestyle="--", linewidth=1.0)
        axes[3].set_ylabel("Residuals (°C)", fontweight="bold")
        axes[3].set_xlabel("Date (Years 2000–2010)")
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "03_seasonal_decomposition.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_04_autocorrelation_acf_pacf(self, df: pd.DataFrame) -> str:
        """Plot 4: Autocorrelation (ACF) and Partial Autocorrelation (PACF) Analysis."""
        series = df["BASEL_temp_mean"].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # ACF (Long-range 400 lags to capture full annual 365-day harmonic)
        plot_acf(series, lags=400, ax=axes[0], color="#1f77b4", title="Autocorrelation Function (ACF) - Lags 0 to 400")
        axes[0].axvline(365, color="red", linestyle="--", linewidth=1.5, label="Annual Cycle (Lag 365)")
        axes[0].set_xlabel("Lag (Days)")
        axes[0].set_ylabel("ACF Correlation")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        
        # PACF (Short-range 40 lags to identify direct autoregressive cutoff)
        plot_pacf(series, lags=40, ax=axes[1], color="#d62728", method="yw", title="Partial Autocorrelation (PACF) - Lags 0 to 40")
        axes[1].axvline(3, color="green", linestyle=":", linewidth=1.5, label="Synoptic Decay (Lag 3)")
        axes[1].axvline(14, color="purple", linestyle="--", linewidth=1.5, label="Recommended LSTM Window (W=14)")
        axes[1].set_xlabel("Lag (Days)")
        axes[1].set_ylabel("PACF Correlation")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 4: Autocorrelation & Partial Autocorrelation Diagnostics for LSTM Window Sizing", y=1.02, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "04_autocorrelation_acf_pacf.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_05_stationarity_and_transformation(self, df: pd.DataFrame, adf_raw: Dict, adf_diff: Dict) -> str:
        """Plot 5: Stationarity Assessment & Transformation (Raw Series vs Differenced Series)."""
        series = df["BASEL_temp_mean"].dropna()
        diff_series = series.diff().dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        
        # Raw Series
        axes[0, 0].plot(df["DATE_DT"], series, color="#1f77b4", linewidth=0.9)
        axes[0, 0].set_title(f"Raw Series: Basel Mean Temp (ADF Stat = {adf_raw['adf_statistic']:.3f}, p = {adf_raw['p_value']:.4f})", fontweight="bold")
        axes[0, 0].set_ylabel("Temperature (°C)")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Differenced Series (Delta y_t)
        axes[0, 1].plot(df["DATE_DT"].iloc[1:], diff_series, color="#2ca02c", linewidth=0.8)
        axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1.0)
        axes[0, 1].set_title(f"First-Differenced Series: ΔT (ADF Stat = {adf_diff['adf_statistic']:.3f}, p = {adf_diff['p_value']:.2e} - Stationary)", fontweight="bold")
        axes[0, 1].set_ylabel("Daily Change ΔT (°C)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Raw Series Histogram
        sns.histplot(series, kde=True, ax=axes[1, 0], color="#1f77b4", stat="density", bins=30)
        axes[1, 0].set_title("Raw Distribution (Bimodal Seasonality)", fontweight="bold")
        axes[1, 0].set_xlabel("Temperature (°C)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Differenced Histogram (Zero-Centered Gaussian)
        sns.histplot(diff_series, kde=True, ax=axes[1, 1], color="#2ca02c", stat="density", bins=30)
        axes[1, 1].set_title("Differenced Distribution (Stationary Gaussian Noise)", fontweight="bold")
        axes[1, 1].set_xlabel("Daily Temperature Shift ΔT (°C)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 5: Stationarity Check (ADF Testing) and Transformations for Neural Network Stability", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "05_stationarity_and_transformation.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_06_cross_station_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Plot 6: Cross-Station Spatial Correlation Heatmap across European cities."""
        stations = [
            "BASEL", "DE_BILT", "DUSSELDORF", "MAASTRICHT", "HEATHROW",
            "TOURS", "MONTELIMAR", "PERPIGNAN", "MUENCHEN", "DRESDEN",
            "KASSEL", "BUDAPEST", "LJUBLJANA", "ROMA", "SONNBLICK",
            "MALMO", "STOCKHOLM", "OSLO"
        ]
        
        temp_cols = [f"{s}_temp_mean" for s in stations if f"{s}_temp_mean" in df.columns]
        corr_matrix = df[temp_cols].corr()
        corr_matrix.columns = [c.replace("_temp_mean", "") for c in corr_matrix.columns]
        corr_matrix.index = [c.replace("_temp_mean", "") for c in corr_matrix.index]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=0.5,
            vmax=1.0,
            square=True,
            cbar_kws={"label": "Pearson Correlation Coefficient"},
            ax=ax,
            linewidths=0.5
        )
        ax.set_title("Figure 6: Cross-Station Spatial Correlation Matrix of Daily Mean Temperature Across Europe", fontweight="bold", pad=15)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "06_cross_station_correlation_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_07_inter_variable_relationships(self, df: pd.DataFrame) -> str:
        """Plot 7: Multivariate correlation and scatter relationships for a single station (Basel)."""
        vars_base = [
            "BASEL_temp_mean", "BASEL_global_radiation", "BASEL_sunshine",
            "BASEL_cloud_cover", "BASEL_humidity", "BASEL_pressure", "BASEL_precipitation"
        ]
        clean_vars = [v for v in vars_base if v in df.columns]
        corr_subset = df[clean_vars].corr()
        clean_labels = [v.replace("BASEL_", "").replace("_", " ").title() for v in clean_vars]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Correlation Heatmap
        sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="vlag", vmin=-1.0, vmax=1.0,
                    xticklabels=clean_labels, yticklabels=clean_labels, ax=axes[0], cbar_kws={"label": "Correlation"})
        axes[0].set_title("Inter-Variable Correlation Matrix (Basel Station)", fontweight="bold")
        axes[0].set_xticklabels(clean_labels, rotation=35, ha="right")
        
        # 2. Scatter Plot: Radiation vs Sunshine colored by Cloud Cover
        scatter = axes[1].scatter(
            df["BASEL_sunshine"],
            df["BASEL_global_radiation"],
            c=df["BASEL_cloud_cover"],
            cmap="viridis_r",
            alpha=0.6,
            s=20
        )
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label("Cloud Cover (Oktas: 0=Clear, 8=Overcast)")
        axes[1].set_xlabel("Sunshine Duration (0.1 Hours)")
        axes[1].set_ylabel("Global Radiation (100 W/m²)")
        axes[1].set_title("Solar Radiation vs Sunshine Hours vs Cloud Cover", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 7: Physical Inter-Variable Meteorological Relationships", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "07_inter_variable_relationships.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_08_cyclical_and_lag_features(self, df_feat: pd.DataFrame) -> str:
        """Plot 8: Visualizing Continuous Cyclical Feature Encodings (Polar & Cartesian) and Lag Correlations."""
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        
        # 1. Polar Plot of Day of Year (Cyclical Continuity)
        theta = 2 * np.pi * df_feat["dayofyear"] / 365.25
        axes[0].plot(df_feat["sin_dayofyear"][:365], df_feat["cos_dayofyear"][:365], color="#e7298a", linewidth=2.5)
        # Annotate months
        month_midpoints = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for m_day, m_name in zip(month_midpoints, month_names):
            x = np.sin(2 * np.pi * m_day / 365.25)
            y = np.cos(2 * np.pi * m_day / 365.25)
            axes[0].scatter([x], [y], color="black", s=30)
            axes[0].text(x * 1.15, y * 1.15, m_name, fontsize=9, ha="center", va="center", fontweight="bold")
        axes[0].set_title(r"Cyclical Day-of-Year Encoding ($\sin, \cos$ Space)", fontweight="bold")
        axes[0].set_xlabel(r"$\sin(2\pi \cdot d / 365.25)$")
        axes[0].set_ylabel(r"$\cos(2\pi \cdot d / 365.25)$")
        axes[0].set_xlim(-1.35, 1.35)
        axes[0].set_ylim(-1.35, 1.35)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Time-series view of Sin/Cos features
        sample_year = df_feat[df_feat["year"] == 2005]
        axes[1].plot(sample_year["DATE_DT"], sample_year["sin_dayofyear"], label=r"$\sin(2\pi d / 365.25)$", color="#1f77b4", linewidth=2)
        axes[1].plot(sample_year["DATE_DT"], sample_year["cos_dayofyear"], label=r"$\cos(2\pi d / 365.25)$", color="#ff7f0e", linewidth=2)
        axes[1].set_title("Continuous Annual Cyclical Waves (Year 2005)", fontweight="bold")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Encoded Feature Value")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # 3. Lag-1 vs Current Scatter (Autoregressive Inertia)
        target = "BASEL_temp_mean"
        lag1 = f"{target}_lag_1"
        if lag1 in df_feat.columns:
            sns.regplot(data=df_feat.sample(1000, random_state=42), x=lag1, y=target, ax=axes[2],
                        scatter_kws={"alpha": 0.4, "color": "#2ca02c", "s": 15}, line_kws={"color": "red", "linewidth": 2})
            corr_val = df_feat[lag1].corr(df_feat[target])
            axes[2].set_title(f"Autoregressive Memory: $T_t$ vs $T_{{t-1}}$ (r = {corr_val:.3f})", fontweight="bold")
            axes[2].set_xlabel("Lag-1 Temperature $T_{t-1}$ (°C)")
            axes[2].set_ylabel("Current Temperature $T_t$ (°C)")
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 8: Feature Engineering - Continuous Temporal Cyclical Embeddings and Lag Dynamics", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "08_cyclical_and_lag_features.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_09_target_distributions_and_imbalance(self, df_feat: pd.DataFrame) -> str:
        """Plot 9: Continuous Target & Binary Classification Imbalance Profiling."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Continuous Regression Target (Next-Day Temperature)
        sns.histplot(df_feat["target_temp_mean_next_day"], kde=True, ax=axes[0], color="#1f77b4", stat="density", bins=35)
        mean_v = df_feat["target_temp_mean_next_day"].mean()
        std_v = df_feat["target_temp_mean_next_day"].std()
        axes[0].axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.1f}°C")
        axes[0].axvline(mean_v - std_v, color="orange", linestyle=":", label=f"±1σ: [{mean_v-std_v:.1f}, {mean_v+std_v:.1f}]")
        axes[0].axvline(mean_v + std_v, color="orange", linestyle=":")
        axes[0].set_title("Continuous Target: Next-Day Mean Temperature ($T_{t+1}$)", fontweight="bold")
        axes[0].set_xlabel("Next-Day Temperature (°C)")
        axes[0].set_ylabel("Density")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        
        # 2. Binary Target Imbalance (Picnic Weather Suitability)
        picnic_counts = df_feat["target_picnic_next_day"].value_counts(normalize=True) * 100
        bars = axes[1].bar(["Unsuitable (0: False)", "Suitable (1: True)"],
                           [picnic_counts.get(0.0, 0), picnic_counts.get(1.0, 0)],
                           color=["#d95f02", "#1b9e77"], width=0.5, edgecolor="black")
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1.5, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Binary Target Imbalance: Next-Day Picnic Weather ($Y_{t+1}$)", fontweight="bold")
        axes[1].set_ylabel("Percentage of Total Days (%)")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 9: Prediction Target Distributions and Class Imbalance Assessment", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "09_target_distributions_and_imbalance.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_10_chronological_splits_and_nn_scaling(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: np.ndarray
    ) -> str:
        """Plot 10: Visualizing Chronological Split Protocol and StandardScaler Normalization for Neural Networks."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        total_samples = len(train_df) + len(val_df) + len(test_df)
        axes[0].plot(train_df["DATE_DT"], train_df["BASEL_temp_mean"], label=f"Train (2000–2006: {len(train_df)} samples, {len(train_df)/total_samples*100:.1f}%)", color="#1f77b4", linewidth=0.8)
        axes[0].plot(val_df["DATE_DT"], val_df["BASEL_temp_mean"], label=f"Validation (2007–2008: {len(val_df)} samples, {len(val_df)/total_samples*100:.1f}%)", color="#ff7f0e", linewidth=0.8)
        axes[0].plot(test_df["DATE_DT"], test_df["BASEL_temp_mean"], label=f"Test (2009–2010: {len(test_df)} samples, {len(test_df)/total_samples*100:.1f}%)", color="#2ca02c", linewidth=0.8)
        axes[0].axvline(val_df["DATE_DT"].min(), color="black", linestyle="--", linewidth=1.5)
        axes[0].axvline(test_df["DATE_DT"].min(), color="black", linestyle="--", linewidth=1.5)
        axes[0].set_title("Chronological Train / Validation / Test Splitting (Zero Lookahead Leakage)", fontweight="bold")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].legend(loc="upper right", frameon=True)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Scaled Feature Distributions (Mean = 0, Std = 1 for Neural Gradient Stability)
        sample_scaled = pd.DataFrame(X_train[:, :4], columns=["Basel Temp", "Basel Pressure", "Basel Humidity", "Basel Rad"])
        sns.kdeplot(data=sample_scaled, ax=axes[1], palette="Set1", linewidth=2.0)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1.0)
        axes[1].set_title("StandardScaler Output Distribution for Neural Network Gradient Stability", fontweight="bold")
        axes[1].set_xlabel("Standardized Z-Score (Mean = 0, Std = 1)")
        axes[1].set_ylabel("Density")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Figure 10: Chronological Dataset Partitioning and Neural Network Standardization Pipeline", y=0.98, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, "10_chronological_splits_and_nn_scaling.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path
