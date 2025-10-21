# API Reference

## Core Classes

### EnhancedCapacityPlanner

Main orchestrator coordinating all system components.

**Location**: `app.py:6150`

#### Methods

##### `initialize(use_sample_data: bool = True) -> None`

Initialize the capacity planner with sample data or prepare for live database connection.

**Parameters**:
- `use_sample_data` (bool): If True, generates synthetic sample data. If False, prepares for live database connection.

**Example**:
```python
planner = EnhancedCapacityPlanner()
planner.initialize(use_sample_data=True)  # Demo mode
```

##### `collect_metrics(db_connection: DatabaseConnection) -> pd.DataFrame`

Collect current metrics from connected database.

**Parameters**:
- `db_connection` (DatabaseConnection): Database connection configuration

**Returns**:
- DataFrame with collected metrics

**Example**:
```python
connection = DatabaseConnection(
    db_type='postgresql',
    host='localhost',
    port=5432,
    database='prod_db',
    username='user',
    password='pass'
)
metrics = planner.collect_metrics(connection)
```

##### `generate_capacity_report(metrics_df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Any]`

Generate comprehensive capacity analysis report.

**Parameters**:
- `metrics_df` (pd.DataFrame): Historical metrics data
- `forecast_days` (int): Number of days to forecast (default: 30)

**Returns**:
- Dictionary containing:
  - `current_status`: Current capacity utilization
  - `predictions`: Future capacity forecasts
  - `recommendations`: Scaling suggestions
  - `health_score`: Overall database health (0-100)

---

### AdvancedPredictionEngine

ML ensemble for time series forecasting.

**Location**: `app.py:3152-3251`

#### Methods

##### `predict_future_metrics(df: pd.DataFrame, metric: str, prediction_days: int = 7, confidence: float = 0.95) -> Dict[str, Any]`

Generate forecasts using ensemble ML models.

**Parameters**:
- `df` (pd.DataFrame): Historical data with timestamp and metric columns
- `metric` (str): Name of metric column to predict (e.g., 'cpu_usage', 'disk_usage')
- `prediction_days` (int): Forecast horizon in days (default: 7, max: 90)
- `confidence` (float): Confidence interval level (default: 0.95, range: 0.8-0.99)

**Returns**:
- Dictionary containing:
  - `predictions`: Array of predicted values
  - `timestamps`: Future timestamps
  - `lower_bound`: Lower confidence interval
  - `upper_bound`: Upper confidence interval
  - `model_performance`: Accuracy metrics per model
  - `feature_importance`: Key factors driving predictions

**Requirements**:
- Minimum 10 data points (recommended: 90+ days)
- LSTM requires 100+ data points

**Example**:
```python
engine = AdvancedPredictionEngine()
result = engine.predict_future_metrics(
    df=historical_data,
    metric='disk_usage',
    prediction_days=30,
    confidence=0.95
)
print(f"Predicted usage in 30 days: {result['predictions'][-1]:.1f}%")
```

##### `cross_validate(data: pd.DataFrame, metric: str, n_folds: int = 5, forecast_days: int = 7) -> Dict[str, float]`

Perform time series cross-validation.

**Parameters**:
- `data` (pd.DataFrame): Historical data
- `metric` (str): Metric to validate
- `n_folds` (int): Number of validation folds (default: 5)
- `forecast_days` (int): Forecast horizon for each fold (default: 7)

**Returns**:
- Dictionary with accuracy metrics:
  - `mae_mean`: Mean Absolute Error (average across folds)
  - `mape_mean`: Mean Absolute Percentage Error
  - `rmse_mean`: Root Mean Square Error
  - `r2_mean`: R² score
  - `*_std`: Standard deviation for each metric

**Example**:
```python
validation_results = engine.cross_validate(
    data=historical_data,
    metric='cpu_usage',
    n_folds=5
)
print(f"Average MAPE: {validation_results['mape_mean']:.2f}%")
```

---

### WorkloadForecaster

Analyze workload patterns and predict future load.

**Location**: `app.py:2866`

#### Methods

##### `forecast_workload(metrics_df: pd.DataFrame, days: int = 7) -> Dict[str, Any]`

Generate workload forecast based on historical patterns.

**Parameters**:
- `metrics_df` (pd.DataFrame): Historical metrics
- `days` (int): Forecast horizon

**Returns**:
- Dictionary with workload predictions

##### `detect_patterns(metrics_df: pd.DataFrame) -> Dict[str, Any]`

Identify recurring patterns in workload.

**Returns**:
- Dictionary containing:
  - `daily_pattern`: Hour-by-hour pattern
  - `weekly_pattern`: Day-by-day pattern
  - `peak_times`: List of peak usage periods
  - `growth_rate`: Trend analysis

---

### AutoScalingManager

Generate resource scaling recommendations.

**Location**: `app.py:3878`

#### Methods

##### `generate_recommendations(predictions: Dict[str, Any], thresholds: Dict[str, MetricThreshold]) -> List[Dict[str, Any]]`

Create scaling recommendations based on predictions.

**Parameters**:
- `predictions`: Forecast results from PredictionEngine
- `thresholds`: Metric threshold configuration

**Returns**:
- List of recommendations, each containing:
  - `action`: 'scale_up', 'scale_down', or 'no_action'
  - `metric`: Affected metric
  - `current_value`: Current utilization
  - `predicted_value`: Forecasted utilization
  - `urgency`: 'low', 'medium', 'high', 'critical'
  - `estimated_date`: When action should be taken
  - `justification`: Explanation for recommendation

**Example**:
```python
manager = AutoScalingManager()
recommendations = manager.generate_recommendations(
    predictions=forecast_results,
    thresholds=default_thresholds
)
for rec in recommendations:
    print(f"{rec['action']}: {rec['metric']} - {rec['justification']}")
```

---

### RealMetricsCollector

Collect live metrics from connected databases.

**Location**: `app.py:1124`

#### Methods

##### `collect_from_database(connection: DatabaseConnection) -> pd.DataFrame`

Gather metrics from live database.

**Parameters**:
- `connection`: Database connection configuration

**Returns**:
- DataFrame with current metrics

**Collected Metrics**:
- Performance: cpu_usage, memory_usage, query_time, buffer_hit_ratio
- Storage: disk_usage, data_size, index_size, temp_usage
- I/O: iops, read_throughput, write_throughput
- Network: network_in, network_out, replication_lag
- Locks: lock_waits, deadlocks

##### `validate_metrics(data: pd.DataFrame) -> pd.DataFrame`

Validate collected metrics for data quality.

**Returns**:
- Validated DataFrame with outliers flagged

---

### OptimizedDatabaseConnector

Manage connections to multiple database types.

**Location**: `app.py:857`

#### Methods

##### `connect(db_config: DatabaseConnection) -> Any`

Establish database connection with pooling.

**Parameters**:
- `db_config`: Database connection configuration

**Returns**:
- Database connection object (SQLAlchemy engine or MongoDB client)

**Example**:
```python
connector = OptimizedDatabaseConnector()
conn = connector.connect(DatabaseConnection(
    db_type='postgresql',
    host='localhost',
    port=5432,
    database='mydb',
    username='user',
    password='pass'
))
```

##### `get_connection(db_type: str) -> Any`

Retrieve existing pooled connection.

##### `close_all() -> None`

Close all open connections.

---

### DataLoader

Import historical metrics from various formats.

**Location**: `app.py:519-683`

#### Methods

##### `load_file(filepath: str, format: str = 'auto') -> pd.DataFrame`

Load data from file with format auto-detection.

**Parameters**:
- `filepath`: Path to data file
- `format`: File format ('csv', 'json', 'excel', 'parquet', 'bson', or 'auto')

**Returns**:
- DataFrame with loaded metrics

**Supported Formats**:
- CSV: Comma-separated values (chunked reading for >100MB)
- JSON: JavaScript Object Notation
- Excel: .xls, .xlsx workbooks (multiple sheets)
- Parquet: Columnar storage format
- BSON: MongoDB binary JSON
- SQL Dump: .sql files (creates temporary SQLite database)

**Example**:
```python
loader = DataLoader()
data = loader.load_file('metrics_2024.csv', format='csv')
```

##### `parse_csv(filepath: str, chunk_size: int = 100000) -> pd.DataFrame`

Parse CSV file with optional chunking.

**Parameters**:
- `filepath`: Path to CSV file
- `chunk_size`: Rows per chunk for large files (default: 100000)

---

## Data Classes

### DatabaseConnection

**Location**: `app.py:102`

```python
@dataclass
class DatabaseConnection:
    db_type: str           # 'postgresql', 'mysql', 'mongodb', 'sqlite', 'oracle', 'mariadb'
    host: str              # Server hostname or IP
    port: int              # Database port
    database: str          # Database name
    username: str          # Username
    password: str          # Password
    ssl_enabled: bool = False  # Enable SSL/TLS
```

### MetricThreshold

**Location**: `app.py:92`

```python
@dataclass
class MetricThreshold:
    warning: float         # Warning threshold value
    critical: float        # Critical threshold value
    unit: str              # Metric unit ('%', 'MB/s', 'ops/sec', etc.)
    description: str       # Human-readable description
    recovery_action: str   # Suggested action when exceeded
    impact_score: float    # Severity multiplier (0.0-1.0)
```

---

## Constants

### Supported Databases

**Location**: `app.py:172-178`

```python
SUPPORTED_DATABASES = {
    'postgresql': {'port': 5432, 'driver': 'postgresql+psycopg2'},
    'mysql': {'port': 3306, 'driver': 'mysql+pymysql'},
    'mariadb': {'port': 3306, 'driver': 'mysql+pymysql'},
    'oracle': {'port': 1521, 'driver': 'oracle+cx_oracle'},
    'mongodb': {'port': 27017, 'driver': 'mongodb'},
    'sqlite': {'port': None, 'driver': 'sqlite'}
}
```

### Default Metric Thresholds

**Location**: `app.py:185-199`

```python
DEFAULT_THRESHOLDS = {
    'cpu_usage': MetricThreshold(70.0, 85.0, '%', 'CPU utilization'),
    'memory_usage': MetricThreshold(75.0, 90.0, '%', 'Memory utilization'),
    'disk_usage': MetricThreshold(80.0, 95.0, '%', 'Disk space usage'),
    'iops': MetricThreshold(1000, 2000, 'ops/sec', 'I/O operations per second'),
    'read_throughput': MetricThreshold(100, 200, 'MB/s', 'Read throughput'),
    'write_throughput': MetricThreshold(100, 200, 'MB/s', 'Write throughput'),
    'connection_count': MetricThreshold(80, 95, 'connections', 'Active connections')
}
```

### ML Model Configuration

**Location**: `app.py:298-302`

```python
MODEL_WEIGHTS = {
    'prophet': 0.30,      # Requires prophet library
    'lstm': 0.25,         # Requires tensorflow, >100 data points
    'xgboost': 0.20,      # Always available
    'lightgbm': 0.15,     # Always available
    'linear': 0.10        # Always available
}
```

---

## Usage Patterns

### Complete Workflow Example

```python
# 1. Initialize system
planner = EnhancedCapacityPlanner()
planner.initialize(use_sample_data=False)

# 2. Connect to database
connection = DatabaseConnection(
    db_type='postgresql',
    host='prod-db.example.com',
    port=5432,
    database='main_db',
    username='monitoring_user',
    password='secure_password'
)

# 3. Collect historical metrics
collector = RealMetricsCollector()
metrics = collector.collect_from_database(connection)

# 4. Generate forecasts
engine = AdvancedPredictionEngine()
cpu_forecast = engine.predict_future_metrics(
    df=metrics,
    metric='cpu_usage',
    prediction_days=30,
    confidence=0.95
)

# 5. Get scaling recommendations
manager = AutoScalingManager()
recommendations = manager.generate_recommendations(
    predictions=cpu_forecast,
    thresholds=DEFAULT_THRESHOLDS
)

# 6. Generate report
report = planner.generate_capacity_report(
    metrics_df=metrics,
    forecast_days=30
)

# 7. Export results
planner.export_report(report, format='pdf', output_path='capacity_report.pdf')
```

---

## Error Handling

All methods follow consistent error handling:

```python
try:
    result = method_call()
except ValueError as e:
    # Invalid input parameters
    logger.error(f"Invalid parameters: {e}")
except ConnectionError as e:
    # Database connection failed
    logger.error(f"Connection failed: {e}")
except Exception as e:
    # Unexpected errors
    logger.error(f"Unexpected error: {e}")
    # System continues with graceful degradation
```

---

For conceptual architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).
For database-specific usage, see [DATABASES.md](DATABASES.md).
