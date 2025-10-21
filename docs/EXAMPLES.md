# Usage Examples

## Table of Contents

1. [Quick Start: Demo Mode](#example-1-quick-start-demo-mode)
2. [Connecting to PostgreSQL Database](#example-2-connecting-to-postgresql-database)
3. [Forecasting Storage Growth](#example-3-forecasting-storage-growth-for-next-30-days)
4. [Setting Up Anomaly Detection](#example-4-setting-up-anomaly-detection)
5. [Generating Executive Reports](#example-5-generating-executive-capacity-report)
6. [Importing Historical Metrics](#example-6-importing-historical-metrics-from-csv)
7. [Comparing Multiple Databases](#example-7-comparing-multiple-databases)
8. [Using External Factors](#example-8-using-external-factors-in-predictions)
9. [Backtesting Model Accuracy](#example-9-backtesting-model-accuracy)
10. [Troubleshooting Common Issues](#example-10-troubleshooting-common-issues)

---

## Example 1: Quick Start - Demo Mode

**Scenario**: First time using the tool, want to explore features without connecting to database.

### Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch application
streamlit run app.py
```

### What Happens

- Browser opens to `http://localhost:8501`
- Tool initializes with sample data (E-commerce scenario by default)
- Interactive dashboard displays:
  - Current metrics (CPU, memory, storage, I/O)
  - Historical trends (last 90 days)
  - Metric threshold indicators

### Exploring Features

1. **Navigate Scenarios**:
   - Sidebar → "Workload Scenario"
   - Select "SaaS Application"
   - Click "Reload with New Scenario"
   - Observe different traffic patterns

2. **Generate Forecast**:
   - Main panel → "Forecasting" section
   - Metric: "CPU Usage"
   - Forecast Horizon: 14 days
   - Click "Generate Forecast"
   - Review prediction chart with confidence intervals

3. **View Recommendations**:
   - Scroll to "Recommendations" panel
   - Review auto-scaling suggestions
   - Check estimated dates for capacity actions

**Expected Time**: 5 minutes

---

## Example 2: Connecting to PostgreSQL Database

**Scenario**: Monitor production PostgreSQL database for capacity planning.

### Prerequisites

- PostgreSQL database accessible over network
- Monitoring user with read permissions (see [DATABASES.md](DATABASES.md))

### Steps

#### 1. Create Monitoring User (on PostgreSQL server)

```sql
-- As superuser
CREATE USER capacity_monitor WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE production_db TO capacity_monitor;
GRANT pg_read_all_stats TO capacity_monitor;
GRANT SELECT ON pg_stat_statements TO capacity_monitor;
```

#### 2. Configure Connection in UI

In Streamlit sidebar:

- **Database Type**: PostgreSQL
- **Host**: `prod-db.example.com`
- **Port**: `5432`
- **Database**: `production_db`
- **Username**: `capacity_monitor`
- **Password**: `secure_password`
- **Enable SSL**: ✅ (recommended for remote)

Click **"Test Connection"** → Should see "✅ Connection successful"

#### 3. Start Collecting Metrics

- Click **"Start Collecting Metrics"**
- Collection runs every 5 minutes automatically
- View real-time data in dashboard

#### 4. Wait for Historical Data

- **Minimum**: 24 hours for initial forecasts
- **Recommended**: 7+ days for better accuracy
- **Optimal**: 90+ days for seasonal pattern detection

### Verification

Check "Metrics Overview" panel:
- Current CPU, memory, disk usage displayed
- "Last Updated" timestamp shows recent data
- Charts populate with actual database metrics

**Expected Time**: 15 minutes setup + 24 hours data collection

---

## Example 3: Forecasting Storage Growth for Next 30 Days

**Scenario**: Database growing rapidly, need to know when to expand capacity.

### Prerequisites

- Historical data: 30+ days (90+ days recommended)
- Current disk usage: 72%

### Steps

#### 1. Navigate to Storage Analysis

Main panel → "Storage Forecasting" section

#### 2. Configure Forecast

- **Metric**: Disk Usage (%)
- **Forecast Horizon**: 30 days
- **Confidence Interval**: 95%
- **Models**: All (ensemble)

#### 3. Generate Forecast

Click **"Generate Forecast"**

Processing takes 10-30 seconds.

#### 4. Review Results

**Prediction Chart**:
```
Day 0:  72% (current)
Day 7:  75% ± 2%
Day 14: 78% ± 3%
Day 21: 81% ± 4%
Day 30: 84% ± 5%
```

**Threshold Visualization**:
- Green zone: 0-80% (safe)
- Yellow zone: 80-95% (warning)
- Red zone: >95% (critical)

Forecast line crosses yellow threshold at Day 21.

#### 5. Review Recommendations

**Recommendations Panel** shows:

```
⚠️  Action Required: Scale Up Storage
├─ Urgency: Medium
├─ Current Usage: 72%
├─ Predicted Usage (30 days): 84%
├─ Threshold: 80% (warning), 95% (critical)
├─ Recommended Action: Expand disk capacity by 30% within 21 days
└─ Justification: Current growth rate (0.4%/day) will reach warning threshold by Day 21
```

#### 6. Export Forecast

- Click **"Export Forecast"** → CSV
- Save for documentation: `storage_forecast_2024-01.csv`

### Interpretation

- **Safe until Day 20**: No immediate action needed
- **Plan expansion by Day 21**: Initiate procurement/approval process
- **Target completion**: Day 25 (5-day buffer before 80% threshold)
- **Capacity to add**: 30% expansion (enough for 3 months at current growth rate)

**Expected Time**: 5 minutes

---

## Example 4: Setting Up Anomaly Detection

**Scenario**: Detect unusual CPU spikes that might indicate issues.

### Prerequisites

- Historical data: 90+ days (more data = better baseline)

### Steps

#### 1. Navigate to Anomaly Detection

Main panel → "Anomaly Detection" tab

#### 2. Configure Detection

- **Metric**: CPU Usage (%)
- **Detection Method**: Isolation Forest (fast) or LSTM (accurate)
- **Sensitivity**: Medium (2% false positive rate)
- **Lookback Period**: 90 days

#### 3. Run Detection

Click **"Detect Anomalies"**

Processing takes 30-60 seconds.

#### 4. Review Flagged Anomalies

**Results Table**:

| Timestamp | Value | Expected Range | Severity | Type |
|-----------|-------|----------------|----------|------|
| 2024-01-15 03:15 | 95% | 30-50% | High | Spike |
| 2024-01-18 14:30 | 8% | 40-60% | Medium | Drop |
| 2024-01-22 09:00 | 92% | 35-55% | High | Spike |

**Anomaly Chart**:
- Blue line: Actual CPU usage
- Gray band: Expected range (normal)
- Red markers: Detected anomalies

#### 5. Investigate Anomalies

Click on anomaly timestamp → Drill-down view:

```
Anomaly Details: 2024-01-15 03:15
├─ Actual Value: 95% CPU
├─ Expected: 40% ± 10%
├─ Deviation: +55 percentage points
├─ Severity: High
├─ Duration: 45 minutes
├─ Context:
│   ├─ Memory: 88% (also elevated)
│   ├─ IOPS: 2500 (normal: 800-1200)
│   └─ Connections: 450 (normal: 100-200)
└─ Possible Causes:
    ├─ Batch job ran unexpectedly
    ├─ Deployment/migration activity
    └─ Query storm from application
```

#### 6. Set Up Alerts (Future Implementation)

Export anomalies → Integrate with monitoring:
- CSV export: `anomalies_jan_2024.csv`
- Send to Slack/PagerDuty for real-time alerting

### Use Cases

- **Incident Correlation**: Match anomalies with incident reports
- **Capacity Events**: Identify when near-capacity events occurred
- **Pattern Discovery**: Find recurring anomalies (e.g., every Monday 3am = backup job)

**Expected Time**: 10 minutes

---

## Example 5: Generating Executive Capacity Report

**Scenario**: Create quarterly capacity report for management review.

### Prerequisites

- Historical data: 90 days
- Forecasts generated for all key metrics

### Steps

#### 1. Navigate to Reports

Main panel → "Reports" tab

#### 2. Configure Report

- **Report Type**: Executive Summary
- **Time Period**: Q1 2024 (Jan 1 - Mar 31)
- **Forecast Horizon**: 90 days
- **Include**:
  - ✅ Current health score
  - ✅ All key metrics (CPU, memory, storage, I/O)
  - ✅ Forecasts and predictions
  - ✅ Scaling recommendations
  - ✅ Cost projections (if configured)

#### 3. Generate Report

Click **"Generate PDF Report"**

Processing takes 1-2 minutes (includes chart rendering).

#### 4. Review Generated Report

**Report Structure**:

```
Executive Capacity Report: Q1 2024

1. Executive Summary
   - Overall Health Score: 82/100 (Good)
   - Critical Issues: 0
   - Warnings: 2
   - Projected Actions Required: 1 (Storage expansion in 45 days)

2. Database Overview
   - Database: production_db (PostgreSQL 14.5)
   - Size: 450 GB
   - Daily Growth: 5.2 GB/day
   - Connection Count: Avg 250, Peak 800

3. Current Capacity Status
   - CPU: 45% (Good) - 40% headroom
   - Memory: 65% (Good) - 25% headroom
   - Storage: 72% (Warning) - 18% headroom
   - IOPS: 1200/5000 (Excellent) - 76% headroom

4. 90-Day Forecast
   [Charts for each metric showing historical + predicted]

   - CPU: Expected to remain stable (45% → 48%)
   - Memory: Slight increase (65% → 72%)
   - Storage: Approaching threshold (72% → 93%)
   - IOPS: Stable (1200 → 1400)

5. Recommendations
   ⚠️  Priority 1: Expand Storage Capacity
   - Action: Add 200 GB disk capacity
   - Timeline: Within 45 days
   - Justification: Current 5.2 GB/day growth will reach 95% capacity in 50 days
   - Estimated Cost: $500/month (AWS gp3 SSD)

   ℹ️  Priority 2: Monitor Memory Usage
   - Action: Review for optimization if exceeds 80%
   - Timeline: 60 days (approaching warning threshold)

6. External Factors
   - Q1 Holiday Lull: -15% traffic (Jan 1-15)
   - End-of-Quarter Spike: +20% expected (Mar 25-31)

7. Historical Accuracy
   - Previous forecast (Q4 2023) vs Actual: MAPE 8.2%
   - Model confidence: High (95% CI)

8. Appendix
   - Methodology: Ensemble ML models (Prophet, XGBoost, LSTM)
   - Data Sources: PostgreSQL system views, 90-day history
   - Assumptions: Current growth rate continues, no major architecture changes
```

#### 5. Export and Share

- **PDF**: `capacity_report_Q1_2024.pdf` (15 pages)
- **Excel**: `capacity_report_Q1_2024.xlsx` (includes raw data)
- **PowerPoint** (optional): Extract charts for presentation

### Distribution

- Email to: CTO, VP Engineering, Database Team Lead
- Archive in: Confluence / SharePoint
- Schedule: Quarterly reviews

**Expected Time**: 15 minutes

---

## Example 6: Importing Historical Metrics from CSV

**Scenario**: Import 6 months of historical metrics exported from CloudWatch or Datadog.

### Prerequisites

- CSV file with metrics
- Required format:
  - Column 1: `timestamp` (ISO 8601: `2024-01-15 10:30:00`)
  - Columns 2+: Metric values

### CSV Format Example

```csv
timestamp,cpu_usage,memory_usage,disk_usage,iops,connection_count
2024-01-01 00:00:00,35.2,55.8,68.5,950,120
2024-01-01 01:00:00,32.1,54.2,68.5,850,95
2024-01-01 02:00:00,40.5,58.9,68.6,1200,150
...
```

### Steps

#### 1. Prepare CSV File

Ensure CSV meets requirements:
- Timestamp column named `timestamp` or `date` or `datetime`
- Metric columns match expected names:
  - `cpu_usage`, `memory_usage`, `disk_usage`
  - `iops`, `read_throughput`, `write_throughput`
  - `connection_count`

#### 2. Navigate to Data Import

Sidebar → "Data Import" section

#### 3. Upload File

- Click **"Upload File"**
- Select `cloudwatch_metrics_jan_jun_2024.csv`
- File size: 25 MB (acceptable, chunked reading)

#### 4. Map Columns

Tool auto-detects columns. Verify mappings:

```
CSV Column           →  Mapped To
──────────────────────────────────────────
timestamp            →  timestamp ✓
CPUUtilization       →  cpu_usage ✓
MemoryUtilization    →  memory_usage ✓
DiskSpaceUtilization →  disk_usage ✓
ReadIOPS             →  iops (read component) ✓
ConnectionCount      →  connection_count ✓
```

If mapping incorrect:
- Click "Edit Mapping"
- Select correct target metric
- Save

#### 5. Configure Date Format

- **Format Detected**: `YYYY-MM-DD HH:MM:SS` ✓
- If incorrect, specify manually: `%Y-%m-%d %H:%M:%S`

#### 6. Import

Click **"Import Data"**

**Progress**:
```
Parsing CSV...        [████████████] 100%
Validating data...    [████████████] 100%
Loading to system...  [████████████] 100%

✓ Import complete
  - 4,320 records imported
  - Date range: 2024-01-01 to 2024-06-30 (181 days)
  - Hourly frequency detected
  - 0 errors, 12 warnings (minor gaps filled with interpolation)
```

#### 7. Verify Import

Navigate to "Metrics Overview":
- Charts populate with imported data
- Check data range: Should show Jan-Jun 2024
- Verify metric values look reasonable

**Expected Time**: 10 minutes

---

## Example 7: Comparing Multiple Databases

**Scenario**: Compare capacity usage across production, staging, and development databases.

### Steps

#### 1. Connect to Multiple Databases

Repeat connection process for each database:
- Production: `prod-db.example.com:5432/production_db`
- Staging: `staging-db.example.com:5432/staging_db`
- Development: `dev-db.example.com:5432/dev_db`

#### 2. Enable Multi-Database Mode

Sidebar → "Database Selection"
- Select all 3 databases
- Click "Enable Comparison Mode"

#### 3. View Comparison Dashboard

**Comparison Table**:

| Database | CPU | Memory | Storage | Growth Rate | Connections |
|----------|-----|--------|---------|-------------|-------------|
| Production | 55% | 70% | 85% | 5 GB/day | 450 |
| Staging | 25% | 40% | 40% | 0.5 GB/day | 50 |
| Development | 15% | 30% | 25% | 0.2 GB/day | 10 |

**Side-by-Side Charts**:
- Overlay CPU usage for all 3 databases
- Compare storage growth trajectories
- Connection count patterns

#### 4. Identify Insights

**Observations**:
- Staging using 45% of production capacity (appropriate for load testing)
- Development underutilized (consider downsizing instance)
- Storage growth ratio matches traffic patterns (100:10:4)

#### 5. Generate Comparative Report

Export comparison → Excel with 3 sheets (one per database)

**Expected Time**: 20 minutes

---

## Example 8: Using External Factors in Predictions

**Scenario**: Incorporate economic events (e.g., Fed rate decisions) into e-commerce capacity forecasts.

### Prerequisites

- Alpha Vantage API key (free): https://www.alphavantage.co/support/#api-key
- E-commerce workload (benefits from external factors)

### Steps

#### 1. Configure API Key

```bash
export ALPHA_VANTAGE_API_KEY=your_api_key_here
```

Or in Streamlit:
- Settings → "External Factors"
- Enter API key
- Save

#### 2. Enable External Factors

Forecasting section:
- ✅ "Include External Factors"
- Select events:
  - ✅ Economic events (Fed decisions, jobs reports)
  - ✅ Holidays (Thanksgiving, Christmas)
  - ⬜ Natural disasters (optional)

#### 3. Generate Forecast with Factors

Click "Generate Forecast"

**Processing**:
```
Fetching external events... ✓ (25 events found)
Correlating with database load... ✓
Adjusting predictions... ✓
```

#### 4. Review Impact

**External Events Timeline**:

| Date | Event | Predicted Impact |
|------|-------|------------------|
| Nov 24 | Thanksgiving | -40% (people not shopping online) |
| Nov 25 | Black Friday | +400% (major shopping day) |
| Dec 15 | Fed Rate Decision | +15% (market activity) |
| Dec 25 | Christmas | -60% (holiday) |

**Adjusted Forecast**:
- Standard forecast: Linear 2% growth/week
- With external factors: Accounts for Black Friday spike, holiday lulls

#### 5. Compare Forecasts

Chart shows:
- **Blue line**: Standard forecast (no external factors)
- **Green line**: Adjusted forecast (with external factors)
- **Shaded**: Events marked on timeline

**Accuracy Improvement**: MAPE reduced from 15% → 9% with external factors for e-commerce workload.

**Expected Time**: 15 minutes

---

## Example 9: Backtesting Model Accuracy

**Scenario**: Validate forecast accuracy before trusting predictions for capacity decisions.

### Prerequisites

- Historical data: 120+ days (split 80/20 for train/test)

### Steps

#### 1. Navigate to Model Validation

Main panel → "Model Validation" tab

#### 2. Enable Backtesting

- ✅ "Enable Backtesting Mode"

#### 3. Configure Backtest

- **Train/Test Split**: 80/20 (96 days train, 24 days test)
- **Forecast Horizons**: 7, 14, 30 days
- **Metric**: Disk Usage

#### 4. Run Backtest

Click **"Run Backtest"**

Processing takes 2-5 minutes.

#### 5. Review Results

**Accuracy Metrics**:

| Horizon | MAE | MAPE | RMSE | R² | Within CI |
|---------|-----|------|------|----|-----------|
| 7 days | 2.1% | 3.2% | 2.8% | 0.96 | 94% |
| 14 days | 3.5% | 5.1% | 4.2% | 0.92 | 92% |
| 30 days | 6.2% | 8.7% | 7.5% | 0.85 | 89% |

**Interpretation**:
- **7-day forecast**: Excellent (MAPE 3.2%)
- **14-day forecast**: Good (MAPE 5.1%)
- **30-day forecast**: Acceptable (MAPE 8.7%)
- **Confidence Intervals**: 89-94% coverage (close to 95% target)

**Predicted vs Actual Chart**:
- Blue line: Actual disk usage (test period)
- Red line: Predicted disk usage
- Gray band: 95% confidence interval
- Most actual values fall within gray band ✓

#### 6. Per-Model Breakdown

**Model Performance**:

| Model | MAPE (7-day) | MAPE (30-day) |
|-------|--------------|---------------|
| Prophet | 2.8% | 7.5% |
| LSTM | 3.5% | 9.2% |
| XGBoost | 3.9% | 8.1% |
| LightGBM | 4.1% | 8.9% |
| Linear | 4.5% | 9.8% |
| **Ensemble** | **3.2%** | **8.7%** |

**Insight**: Ensemble outperforms individual models.

#### 7. Export Results

Save validation report: `backtest_results_disk_usage.json`

### Decision

With MAPE <10% for 30-day forecasts, predictions are reliable enough for capacity planning decisions.

**Expected Time**: 10 minutes

---

## Example 10: Troubleshooting Common Issues

### Issue 1: Connection to PostgreSQL Fails

**Error**: `psycopg2.OperationalError: FATAL: password authentication failed`

**Solutions**:
1. Verify username/password correct
2. Check `pg_hba.conf` allows connection from your IP:
   ```
   host    production_db    capacity_monitor    192.168.1.0/24    md5
   ```
3. Test connection manually:
   ```bash
   psql -h prod-db.example.com -U capacity_monitor -d production_db
   ```

---

### Issue 2: Forecast Shows Unrealistic Values

**Symptom**: Disk usage forecast predicts 150% (impossible)

**Solutions**:
1. Check data quality:
   - Outliers in historical data? (Clean or remove)
   - Data gaps? (Interpolate missing values)
2. Ensure sufficient historical data (90+ days)
3. Try shorter forecast horizon (7 days instead of 90)
4. Review metric scaling (is disk_usage in % or GB?)

---

### Issue 3: LSTM Model Not Available

**Error**: `LSTM predictions skipped: TensorFlow not available`

**Solution**:
```bash
pip install tensorflow==2.15.0
```

Restart Streamlit application.

---

### Issue 4: CSV Import Fails

**Error**: `Unable to parse date column`

**Solutions**:
1. Verify date format matches expected:
   - Expected: `2024-01-15 10:30:00`
   - If different, specify format: `%Y-%m-%d %H:%M:%S`
2. Check for missing timestamps (fill gaps)
3. Ensure column named `timestamp`, `date`, or `datetime`

---

### Issue 5: Forecast Takes Too Long

**Symptom**: "Generating forecast..." runs for >5 minutes

**Solutions**:
1. Reduce data points:
   - Aggregate hourly → daily for long time ranges
   - Use smaller date range for backtesting
2. Disable expensive models:
   - Uncheck LSTM if >10,000 data points
   - Disable Prophet if not needed
3. Increase server resources (more RAM/CPU)

---

## Next Steps

- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [ML_MODELS.md](ML_MODELS.md) for model selection guidance
- See [ACCURACY.md](ACCURACY.md) for expected forecast accuracy
- Consult [DATABASES.md](DATABASES.md) for database-specific setup
