# Machine Learning Models

## Overview

Database Capacity Planner uses an **ensemble approach** that combines 7 different prediction models to generate accurate capacity forecasts. This document explains each model, when to use it, and how the ensemble works.

## Ensemble Architecture

```
Historical Data
    │
    ├──→ Prophet (30% weight) ────┐
    ├──→ LSTM (25% weight) ────────┤
    ├──→ XGBoost (20% weight) ─────┤
    ├──→ LightGBM (15% weight) ────┤──→ Weighted Average ──→ Final Prediction
    ├──→ Linear Trend (10% weight) ┤      + Confidence
    ├──→ Seasonal Decomposition ───┤        Intervals
    └──→ Moving Average ──────────┘
```

**Why Ensemble?**
- No single model excels at all patterns
- Combining models reduces prediction variance
- Weighted averaging leverages each model's strengths
- More robust to outliers and anomalies

---

## Model 1: Prophet

**Library**: Facebook Prophet
**Weight**: 30% (highest)
**Installation**: `pip install prophet==1.1.5`
**Optional**: Yes (graceful degradation if not installed)

### When to Use

Prophet excels at:
- **Seasonal patterns**: Daily, weekly, monthly cycles
- **Holiday effects**: Incorporates special events
- **Multiple seasonalities**: Handles nested patterns (hourly + daily + weekly)
- **Missing data**: Robust to gaps in historical data
- **Trend changes**: Detects changepoints automatically

### Best For

- E-commerce databases (holiday spikes, weekend patterns)
- SaaS applications (business hours patterns)
- Any workload with strong time-of-day or day-of-week patterns

### How It Works

1. **Decomposition**: Separates data into trend + seasonality + holidays + noise
2. **Trend Modeling**: Piecewise linear or logistic growth
3. **Seasonality**: Fourier series to model cycles
4. **Holidays**: Explicit holiday effects
5. **Uncertainty**: Monte Carlo sampling for confidence intervals

### Data Requirements

- Minimum: 2 weeks of data (preferably 3+ months)
- Frequency: Hourly or daily recommended
- Missing data: Handles gaps automatically

### Accuracy Characteristics

- **Short-term (7 days)**: Excellent (MAPE typically 3-8%)
- **Medium-term (30 days)**: Good (MAPE typically 8-15%)
- **Long-term (90 days)**: Moderate (MAPE typically 15-25%)

### Configuration

```python
# Prophet parameters (internal, auto-configured)
prophet_model = Prophet(
    growth='linear',           # or 'logistic' for bounded growth
    seasonality_mode='additive',  # or 'multiplicative'
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05  # Flexibility of trend changes
)
```

---

## Model 2: LSTM (Long Short-Term Memory)

**Library**: TensorFlow
**Weight**: 25%
**Installation**: `pip install tensorflow==2.15.0`
**Optional**: Yes

### When to Use

LSTM excels at:
- **Complex non-linear patterns**: Learns intricate relationships
- **Long-term dependencies**: Remembers patterns from distant past
- **Sequential patterns**: Naturally suited for time series
- **High-frequency data**: Works well with minute/second-level data

### Best For

- IoT sensor databases (high-frequency writes)
- Game backends (complex player behavior patterns)
- Financial trading databases (volatility modeling)

### How It Works

1. **Sequence Creation**: Sliding window over historical data (e.g., use past 24 hours to predict next hour)
2. **Neural Network**: LSTM cells maintain memory of past patterns
3. **Training**: Backpropagation through time with early stopping
4. **Prediction**: Generate future sequence autoregressively

**Architecture**:
```
Input (sequence length × features)
    ↓
LSTM Layer (64 units)
    ↓
Dropout (20%)
    ↓
LSTM Layer (32 units)
    ↓
Dropout (20%)
    ↓
Dense Layer (16 units)
    ↓
Output (prediction)
```

### Data Requirements

- **Minimum**: 100 data points (preferably 500+)
- **Frequency**: Any (minute, hourly, daily)
- **Preprocessing**: Data normalized to [0, 1] range

### Accuracy Characteristics

- **Short-term (7 days)**: Excellent with sufficient data (MAPE 4-10%)
- **Medium-term (30 days)**: Good (MAPE 10-18%)
- **Long-term (90 days)**: Degrades (MAPE >20%)
- **Note**: Requires more data than other models

### Training Time

- **Small dataset (<1000 points)**: ~10-30 seconds
- **Medium dataset (1000-10000 points)**: ~1-3 minutes
- **Large dataset (>10000 points)**: ~5-10 minutes

---

## Model 3: XGBoost

**Library**: XGBoost
**Weight**: 20%
**Installation**: `pip install xgboost==2.0.3` (included in requirements.txt)
**Optional**: No (always available)

### When to Use

XGBoost excels at:
- **General-purpose forecasting**: Works well across most patterns
- **Feature interactions**: Captures complex relationships
- **Robustness**: Handles outliers well
- **Speed**: Fast training and prediction

### Best For

- Analytics warehouse databases (batch query patterns)
- SaaS applications (mixed workload patterns)
- General-purpose forecasting when pattern is unclear

### How It Works

1. **Feature Engineering**: Create time-based features (hour, day, month, lag values)
2. **Gradient Boosting**: Sequential tree building, each correcting previous errors
3. **Regularization**: L1/L2 penalties prevent overfitting
4. **Tree Ensemble**: Combine many decision trees

**Features Used**:
- Hour of day (0-23)
- Day of week (0-6)
- Day of month (1-31)
- Month of year (1-12)
- Lag features (previous 1, 7, 30 days)
- Rolling statistics (7-day, 30-day averages)

### Data Requirements

- Minimum: 30 data points (preferably 90+)
- Frequency: Any
- Missing data: Handled via imputation

### Accuracy Characteristics

- **Short-term (7 days)**: Good (MAPE 5-12%)
- **Medium-term (30 days)**: Good (MAPE 10-18%)
- **Long-term (90 days)**: Moderate (MAPE 18-28%)

### Configuration

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## Model 4: LightGBM

**Library**: LightGBM
**Weight**: 15%
**Installation**: `pip install lightgbm==4.2.0` (included in requirements.txt)
**Optional**: No (always available)

### When to Use

LightGBM excels at:
- **Large datasets**: Faster than XGBoost on >10,000 points
- **Memory efficiency**: Lower memory footprint
- **Categorical features**: Handles categorical data natively
- **Similar to XGBoost**: Good general-purpose model

### Best For

- Large historical datasets (1+ years of hourly data)
- IoT databases (high-volume sensor data)
- When training time is a concern

### How It Works

Similar to XGBoost but with optimizations:
- **Leaf-wise growth**: Grows trees by best leaf, not level-wise
- **Histogram binning**: Faster than exact splits
- **GOSS**: Gradient-based One-Side Sampling reduces data
- **EFB**: Exclusive Feature Bundling reduces features

### Data Requirements

- Minimum: 50 data points (optimal: 1000+)
- Large datasets: Shines with 10,000+ points

### Accuracy Characteristics

- Similar to XGBoost
- **Short-term (7 days)**: Good (MAPE 5-12%)
- **Medium-term (30 days)**: Good (MAPE 11-19%)
- **Long-term (90 days)**: Moderate (MAPE 19-29%)

---

## Model 5: Linear Trend

**Library**: NumPy (polyfit)
**Weight**: 10%
**Installation**: N/A (always available)
**Optional**: No

### When to Use

Linear Trend excels at:
- **Simple linear growth**: Steady, predictable growth patterns
- **Baseline predictions**: Conservative estimates
- **Interpretability**: Easy to explain to stakeholders

### Best For

- Storage capacity (often grows linearly)
- Steady-state databases with consistent growth
- When simplicity is preferred over accuracy

### How It Works

1. Fit linear regression: `y = mx + b`
2. Extrapolate into future
3. Calculate confidence intervals from residual variance

### Accuracy Characteristics

- **Good for**: Actual linear trends (MAPE 3-10%)
- **Poor for**: Non-linear, seasonal, or volatile patterns (MAPE >30%)

---

## Model 6: Seasonal Decomposition

**Library**: statsmodels
**Weight**: Included in ensemble (dynamic weight)
**Installation**: Included with scipy
**Optional**: No

### When to Use

Seasonal Decomposition excels at:
- **Detecting cycles**: Identifies seasonal periods
- **Detrending**: Separates growth from cycles
- **Pattern visualization**: Helps understand data structure

### How It Works

1. **Decompose**: Separate data into trend + seasonal + residual
2. **Forecast**: Extrapolate trend and repeat seasonal pattern
3. **Combine**: Add trend and seasonal forecasts

### Best For

- Exploratory analysis
- Supplementing other models
- Simple seasonal patterns without complex interactions

---

## Model 7: Moving Average

**Library**: NumPy
**Weight**: Included in ensemble (dynamic weight)
**Installation**: N/A (always available)
**Optional**: No

### When to Use

Moving Average excels at:
- **Short-term smoothing**: Reduce noise
- **Simple forecasts**: Assume recent past continues
- **Baseline comparison**: Simple benchmark model

### How It Works

1. Calculate rolling average (e.g., 7-day MA)
2. Use recent average as forecast
3. Flat forecast (no trend)

### Accuracy Characteristics

- **Very short-term (1-3 days)**: Decent for stable metrics
- **Longer term**: Poor (no trend, no seasonality)

---

## Ensemble Weighting Strategy

### Default Weights

| Model | Weight | Rationale |
|-------|--------|-----------|
| Prophet | 30% | Best at seasonality, most versatile |
| LSTM | 25% | Excellent but requires more data |
| XGBoost | 20% | Robust general-purpose |
| LightGBM | 15% | Fast, efficient, similar to XGBoost |
| Linear | 10% | Simple baseline, interpretable |

### Adaptive Weighting

Weights adjust based on:
1. **Historical accuracy**: Models with lower error get higher weight
2. **Data availability**: LSTM weight reduced if <100 points
3. **Pattern detection**: Increase Prophet weight if strong seasonality detected
4. **Forecast horizon**: Linear weight increases for storage (linear growth expected)

### Confidence Intervals

Calculated from ensemble variance:
```
Lower Bound = Mean Prediction - (z-score × standard deviation)
Upper Bound = Mean Prediction + (z-score × standard deviation)
```

For 95% confidence: z-score = 1.96

---

## Model Selection Guide

### By Workload Type

| Workload | Recommended Primary Model | Why |
|----------|-------------------------|-----|
| E-commerce | Prophet | Strong seasonality (holidays, weekends) |
| SaaS | Prophet / XGBoost | Business hours patterns |
| IoT | LSTM / LightGBM | High-frequency, large volumes |
| Analytics | XGBoost | Batch patterns, moderate complexity |
| Game | LSTM / Prophet | Complex player behavior + time patterns |
| Financial | XGBoost / LSTM | Volatility modeling |

### By Metric Type

| Metric | Recommended Model | Why |
|--------|------------------|-----|
| Storage | Linear / XGBoost | Often linear growth |
| CPU | Prophet / LSTM | Strong time-of-day patterns |
| Memory | Prophet / XGBoost | Correlates with workload |
| IOPS | LSTM / Prophet | Can be volatile and seasonal |

### By Data Availability

| Historical Data | Recommended Models |
|----------------|-------------------|
| <30 days | Linear, Moving Average (limited options) |
| 30-90 days | XGBoost, LightGBM, Linear |
| 90-365 days | Prophet, XGBoost, LightGBM |
| >365 days | All models, especially Prophet, LSTM |

---

## Installing Optional Models

### Prophet

```bash
pip install prophet==1.1.5
```

**Dependencies**:
- pystan (compiled models, can be slow to install)
- fbprophet renamed to prophet in v1.0+

**Installation time**: ~5-10 minutes (compiles C++ extensions)

### TensorFlow (for LSTM)

```bash
# CPU-only version (recommended for most users)
pip install tensorflow==2.15.0

# GPU version (if CUDA available)
pip install tensorflow[and-cuda]==2.15.0
```

**Installation time**: ~2-5 minutes
**Size**: ~400-500 MB

### PyTorch (optional, for autoencoders)

```bash
pip install torch==2.1.2
```

---

## Model Performance Tuning

### Hyperparameter Tuning

For production use, consider tuning model parameters:

**XGBoost/LightGBM**:
- `n_estimators`: Number of trees (default: 100, try: 50-200)
- `max_depth`: Tree depth (default: 5, try: 3-10)
- `learning_rate`: Step size (default: 0.1, try: 0.01-0.3)

**LSTM**:
- `sequence_length`: Lookback window (default: 24, try: 12-72 for hourly)
- `units`: LSTM cells (default: 64, try: 32-128)
- `epochs`: Training iterations (default: 50, try: 30-100)

**Prophet**:
- `changepoint_prior_scale`: Trend flexibility (default: 0.05, try: 0.01-0.5)
- `seasonality_prior_scale`: Seasonality strength (default: 10, try: 1-20)

---

## Troubleshooting

### Prophet Not Available

**Error**: `PROPHET_AVAILABLE = False`, predictions use other models

**Solution**:
```bash
pip install prophet==1.1.5
```
If installation fails due to pystan issues, use conda:
```bash
conda install -c conda-forge prophet
```

### TensorFlow/LSTM Errors

**Error**: LSTM predictions skipped, "TensorFlow not available"

**Solution**: Install TensorFlow as shown above

**Error**: "Failed to compile TensorFlow" on ARM Mac

**Solution**: Use Apple Silicon-specific wheel:
```bash
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal  # GPU acceleration
```

### Prediction Quality Issues

**Symptom**: High MAPE (>30%) or unrealistic predictions

**Diagnosis**:
1. Check data quality: Missing values? Outliers?
2. Verify sufficient historical data (90+ days recommended)
3. Review metric patterns: Is data actually predictable?

**Solutions**:
- Increase historical data collection period
- Clean outliers before prediction
- Try different forecast horizons (shorter = more accurate)
- Check if metric has actual patterns (some are inherently random)

---

For accuracy expectations and validation, see [ACCURACY.md](ACCURACY.md).
For usage examples, see [EXAMPLES.md](EXAMPLES.md).
