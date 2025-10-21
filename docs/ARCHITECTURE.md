# System Architecture

## Overview

Database Capacity Planner is built as a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│              (Interactive dashboards, charts)                │
└────────────────┬────────────────────────────┬────────────────┘
                 │                            │
    ┌────────────▼───────────────┐  ┌────────▼──────────────┐
    │  EnhancedCapacityPlanner   │  │   Data Import/Export  │
    │   (Main Orchestrator)      │  │       (Reports)       │
    └────────────┬───────────────┘  └───────────────────────┘
                 │
    ┌────────────┴────────────┬────────────┬────────────────┐
    │                         │            │                │
┌───▼──────────────┐  ┌──────▼─────┐  ┌──▼──────┐  ┌──────▼─────────┐
│AdvancedPrediction│  │ Workload   │  │AutoScal.│  │   Anomaly      │
│     Engine       │  │ Forecaster │  │ Manager │  │   Detection    │
└───┬──────────────┘  └────────────┘  └─────────┘  └────────────────┘
    │
    │ (ML Ensemble)
    │
    ├── Prophet
    ├── LSTM
    ├── XGBoost
    ├── LightGBM
    ├── Linear Trend
    ├── Seasonal Decomposition
    └── Moving Average

┌────────────────────────────────────────────────────────────┐
│                  Data Collection Layer                      │
├──────────────────────┬─────────────────────────────────────┤
│ RealMetricsCollector │  OptimizedDatabaseConnector         │
│ (Live DB metrics)    │  (Multi-DB connection pooling)      │
└──────────────────────┴─────────────────────────────────────┘
```

## Core Components

### 1. EnhancedCapacityPlanner (app.py:6150)

**Purpose**: Main orchestrator coordinating all system components

**Responsibilities**:
- Initialize system with sample data or live database connection
- Coordinate data collection, analysis, and prediction workflows
- Manage session state and user configuration
- Generate comprehensive capacity reports

**Key Methods**:
- `initialize(use_sample_data: bool)`: System initialization
- `collect_metrics()`: Trigger metrics collection from connected databases
- `generate_capacity_report()`: Create comprehensive capacity analysis
- `export_report(format: str)`: Export reports as PDF, Excel, or JSON

**Dependencies**: All other components

### 2. AdvancedPredictionEngine (app.py:3152-3251)

**Purpose**: ML ensemble for time series forecasting

**Architecture**:
- **Ensemble Approach**: Combines 7 different prediction models
- **Weighted Averaging**: Each model contributes based on historical accuracy
- **Confidence Intervals**: Statistical bounds calculated from model variance
- **Feature Importance**: Identifies key factors driving predictions

**Prediction Models**:

| Model | Weight | Requirements | Best For |
|-------|--------|-------------|----------|
| Prophet | 0.30 | prophet library | Seasonal patterns, holidays |
| LSTM | 0.25 | tensorflow, >100 data points | Complex non-linear trends |
| XGBoost | 0.20 | xgboost (always available) | General purpose, fast |
| LightGBM | 0.15 | lightgbm (always available) | Large datasets, speed |
| Linear | 0.10 | numpy (always available) | Simple linear trends |
| Seasonal | - | scipy (always available) | Cyclical patterns |
| Moving Avg | - | numpy (always available) | Smoothing, short-term |

**Key Methods**:
- `predict_future_metrics(df, metric, days, confidence)`: Generate forecast
  - **Input**: Historical data DataFrame, metric name, forecast horizon
  - **Output**: Predictions with confidence intervals, model performance
- `calculate_confidence_intervals()`: Statistical CI calculation
- `get_feature_importance()`: Model interpretability

**Data Flow**:
```
Historical Data → Data Validation → Model Training → Ensemble Prediction → Confidence Intervals → Output
```

### 3. WorkloadForecaster (app.py:2866)

**Purpose**: Analyze workload patterns and predict future load

**Capabilities**:
- Pattern detection (daily, weekly, monthly cycles)
- Peak time identification
- Growth trend analysis
- Correlation analysis between metrics

**Key Methods**:
- `forecast_workload()`: Generate workload predictions
- `detect_patterns()`: Identify recurring patterns
- `analyze_growth()`: Calculate growth rates

### 4. AutoScalingManager (app.py:3878)

**Purpose**: Generate intelligent resource scaling recommendations

**Decision Logic**:
```
Current Usage + Predicted Growth → Threshold Comparison → Recommendation
                                                           ↓
                                               ┌───────────┴─────────────┐
                                               │                         │
                                         Scale Up                  Scale Down
                                    (approaching limits)      (underutilized)
```

**Recommendation Types**:
- **Scale Up**: Increase resources before hitting limits
- **Scale Down**: Reduce overprovisioned resources
- **No Action**: Current capacity sufficient
- **Urgent Action**: Critical threshold imminent

**Key Methods**:
- `generate_recommendations(predictions)`: Create scaling suggestions
- `calculate_optimal_capacity()`: Determine right-sized resources
- `estimate_cost_impact()`: Project cost changes from scaling

### 5. RealMetricsCollector (app.py:1124)

**Purpose**: Collect live metrics from connected databases

**Supported Metrics**:
- **Performance**: CPU, memory, query times, buffer hit ratio
- **Storage**: Disk usage, data size, index size, temp usage
- **I/O**: IOPS, read/write throughput
- **Network**: Network I/O, replication lag
- **Locks**: Lock waits, deadlocks

**Collection Strategy**:
- Configurable collection interval (default: 5 minutes)
- Batch collection to minimize database impact
- Error handling and retry logic
- Metric validation and outlier detection

**Key Methods**:
- `collect_from_database(conn)`: Gather metrics from database
- `validate_metrics(data)`: Ensure data quality
- `store_metrics(data)`: Persist collected data

### 6. OptimizedDatabaseConnector (app.py:857)

**Purpose**: Manage connections to multiple database types

**Features**:
- **Connection Pooling**: Reuse connections for efficiency
- **Multi-Database Support**: PostgreSQL, MySQL, Oracle, MariaDB, SQLite, MongoDB
- **Retry Logic**: Exponential backoff on connection failures
- **Security**: Secure credential management

**Connection Parameters** (DatabaseConnection dataclass, app.py:102):
```python
@dataclass
class DatabaseConnection:
    db_type: str          # Database type (postgresql, mysql, etc.)
    host: str             # Server address
    port: int             # Database port
    database: str         # Database name
    username: str         # Username
    password: str         # Password (encrypted in memory)
    ssl_enabled: bool     # SSL/TLS configuration
```

**Key Methods**:
- `connect(db_config)`: Establish database connection
- `get_connection(db_type)`: Retrieve pooled connection
- `close_all()`: Clean shutdown of all connections

### 7. DataLoader (app.py:519-683)

**Purpose**: Import historical metrics from various formats

**Supported Formats**:
| Format | Use Case | Large File Support |
|--------|----------|-------------------|
| CSV | Most common export format | Yes (chunked reading >100MB) |
| JSON | API exports, structured data | Yes |
| BSON | MongoDB exports | Yes |
| Parquet | Big data ecosystems | Yes |
| Excel | Manual data entry, reports | Yes (multiple sheets) |
| SQL Dump | Database backups | Yes (temporary SQLite) |

**Data Validation**:
- Timestamp format validation
- Metric value range checking
- Missing data handling
- Duplicate detection

**Key Methods**:
- `load_file(filepath, format)`: Load data from file
- `parse_format(data, format)`: Format-specific parsing
- `validate_data(df)`: Data quality checks

## Data Flow Architecture

### Typical Usage Flow

```
1. User Input
   ↓
2. Data Source Selection
   ├─→ Demo Mode: Generate sample data
   └─→ Live Database: Connect via OptimizedDatabaseConnector
   ↓
3. Historical Data Collection
   ├─→ RealMetricsCollector (live database)
   └─→ DataLoader (imported files)
   ↓
4. Data Storage (session state / cache)
   ↓
5. Analysis & Prediction
   ├─→ AdvancedPredictionEngine: Generate forecasts
   ├─→ WorkloadForecaster: Analyze patterns
   ├─→ Anomaly Detection: Flag outliers
   └─→ AutoScalingManager: Create recommendations
   ↓
6. Visualization (Streamlit UI)
   ├─→ Charts: Plotly interactive graphs
   ├─→ Tables: Metric summaries
   └─→ Recommendations: Actionable insights
   ↓
7. Report Export
   └─→ PDF / Excel / JSON
```

### ML Prediction Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                   Historical Metrics Data                   │
│              (min 10 points, recommended 90+ days)          │
└─────────────┬──────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│               Data Preprocessing & Validation                │
│  • Handle missing values  • Detect outliers                 │
│  • Normalize timestamps   • Validate metric ranges          │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                Feature Engineering                           │
│  • Time-based features (hour, day, month)                   │
│  • Lag features (past values)                               │
│  • Rolling statistics (moving averages)                     │
│  • External factors (economic events, holidays)             │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Training (Parallel)                   │
│  ┌───────┐  ┌───────┐  ┌──────────┐  ┌──────────┐         │
│  │Prophet│  │ LSTM  │  │ XGBoost  │  │ LightGBM │ ...     │
│  └───┬───┘  └───┬───┘  └────┬─────┘  └────┬─────┘         │
└──────┼──────────┼───────────┼─────────────┼────────────────┘
       │          │            │             │
       └──────────┴────────────┴─────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │     Ensemble Aggregation      │
       │   (Weighted average based     │
       │    on historical accuracy)    │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Confidence Interval Calc    │
       │  (Statistical bounds from    │
       │   model variance)            │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │     Final Predictions        │
       │  • Point estimates           │
       │  • Upper/lower bounds        │
       │  • Feature importance        │
       │  • Model performance metrics │
       └──────────────────────────────┘
```

## Technology Stack

### Core Framework
- **Streamlit 1.31+**: Web UI framework
- **Python 3.9+**: Programming language

### Data Processing
- **Pandas 2.1+**: Data manipulation
- **NumPy 1.26+**: Numerical computing
- **SciPy 1.11+**: Scientific computing

### Machine Learning
- **scikit-learn 1.4+**: Classical ML algorithms
- **XGBoost 2.0+**: Gradient boosting
- **LightGBM 4.2+**: Fast gradient boosting
- **Prophet 1.1+** (optional): Time series forecasting
- **TensorFlow 2.15+** (optional): Deep learning (LSTM)
- **PyTorch 2.1+** (optional): Deep learning (autoencoders)

### Visualization
- **Plotly 5.18+**: Interactive charts
- **Matplotlib 3.8+**: Static visualizations
- **Seaborn 0.13+**: Statistical visualizations

### Database Connectivity
- **SQLAlchemy 2.0+**: SQL database abstraction
- **psycopg2-binary 2.9+**: PostgreSQL driver
- **PyMySQL 1.1+**: MySQL/MariaDB driver
- **PyMongo 4.6+**: MongoDB driver
- **cx_Oracle 8.3+** (optional): Oracle driver

### Reporting
- **ReportLab 4.0+**: PDF generation
- **openpyxl 3.1+**: Excel file creation

## Performance Considerations

### Caching Strategy
- **@lru_cache**: Function-level caching for expensive computations
- **Streamlit @st.cache_data**: Data caching across sessions
- **Database connection pooling**: Reuse connections

### Scalability Limits
- **Maximum data points**: ~1 million rows (recommend aggregation beyond this)
- **Concurrent users**: 10-50 users (Streamlit limitation)
- **Forecast horizon**: 90 days (accuracy degrades beyond 30 days)

### Optimization Techniques
- **Chunked file reading**: CSV files >100MB read in chunks
- **Parallel model training**: ML models trained concurrently
- **Lazy loading**: Data loaded on-demand
- **Metric downsampling**: Aggregate high-frequency data for long time ranges

## Security Considerations

### Credential Management
- Passwords stored in session state (not persisted)
- Optional SSL/TLS for database connections
- No credential logging

### Data Privacy
- All data remains local (no external transmission except optional API calls)
- Exported reports contain only aggregated metrics
- No personally identifiable information collected

### External API Calls
- **Alpha Vantage API**: Economic events (optional, requires user-provided API key)
- **Graceful degradation**: System works without external APIs

## Deployment Architectures

### Local Development
```
User's Machine
├── Python 3.9+
├── Streamlit server (localhost:8501)
├── Local database connections
└── File-based data import
```

### Containerized Deployment
```
Docker Container
├── Python 3.9 slim image
├── All dependencies installed
├── Streamlit exposed on port 8501
└── Volume mounts for data persistence
```

### Cloud Deployment (Example: AWS)
```
AWS EC2 Instance
├── Ubuntu 22.04 LTS
├── Streamlit behind nginx reverse proxy
├── SSL/TLS certificate (Let's Encrypt)
├── Database connections via VPC peering
└── S3 for report storage
```

## Extension Points

### Adding New Database Support
1. Add database type to `SUPPORTED_DATABASES` (app.py:172)
2. Implement connection logic in `OptimizedDatabaseConnector`
3. Add metric collection queries in `RealMetricsCollector`
4. Update `docs/DATABASES.md` with configuration guide

### Adding New ML Models
1. Implement model in `AdvancedPredictionEngine`
2. Add to ensemble with appropriate weight
3. Handle optional dependencies gracefully
4. Document in `docs/ML_MODELS.md`

### Adding New Export Formats
1. Implement export function in report generation section
2. Add format option to UI selector
3. Update `DataLoader` if format should also be importable

## Monitoring and Observability

### Logging
- **File logging**: `db_capacity_planner.log` (configurable level)
- **Console logging**: Real-time feedback during execution
- **Log levels**: INFO for normal operations, WARNING for issues, ERROR for failures

### Performance Metrics
- **Prediction time**: Time taken to generate forecasts
- **Model accuracy**: Tracked per model in ensemble
- **Database query time**: Connection and metric collection latency

### Error Handling
- **Try-catch blocks**: All external interactions wrapped
- **Graceful degradation**: System continues with reduced functionality on errors
- **User-friendly error messages**: Clear guidance on resolution

---

For implementation details of specific components, see:
- [API.md](API.md) - Detailed method signatures and usage
- [DATABASES.md](DATABASES.md) - Database-specific configuration
- [ML_MODELS.md](ML_MODELS.md) - ML model implementation details
