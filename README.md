# Database Capacity Planner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive machine learning-based database capacity planning and forecasting tool built with Streamlit. Predict future resource needs, optimize costs, and prevent capacity-related outages with advanced ML ensemble models.

## Overview

Database Capacity Planner helps database administrators and DevOps teams proactively manage database resources through intelligent forecasting and monitoring. The tool combines multiple ML models to provide accurate capacity predictions and actionable recommendations.

### Key Features

- **Multi-Database Support**: Works with PostgreSQL, MySQL, Oracle, MariaDB, SQLite, and MongoDB
- **Comprehensive Metrics Tracking**:
  - CPU utilization with configurable thresholds
  - Memory usage monitoring
  - Storage/disk capacity tracking
  - I/O operations (IOPS) and throughput
  - Connection count tracking
- **Advanced ML Forecasting**: Ensemble approach combining 7 prediction models:
  - Facebook Prophet for time series forecasting
  - LSTM neural networks (TensorFlow)
  - XGBoost gradient boosting
  - LightGBM fast gradient boosting
  - Linear trend analysis
  - Seasonal decomposition
  - Moving average smoothing
- **Realistic Demo Mode**: 6 workload scenarios for testing (e-commerce, SaaS, IoT, analytics, gaming, financial)
- **Flexible Data Import**: Support for JSON, BSON, CSV, SQL dumps, Parquet, and Excel formats
- **Professional Reporting**: Export capacity reports as PDF, Excel workbooks, or JSON
- **Anomaly Detection**: Automated outlier detection with IsolationForest and LSTM models
- **Auto-Scaling Recommendations**: Intelligent resource scaling suggestions based on predicted load
- **External Factors Integration**: Incorporate economic events and seasonal patterns into forecasts

### Use Cases

- **Capacity Planning**: Forecast storage, CPU, and memory needs for next 30-90 days
- **Cost Optimization**: Right-size database instances based on actual usage patterns
- **Performance Monitoring**: Track key metrics and receive alerts on threshold violations
- **Budget Planning**: Predict infrastructure costs with accurate growth forecasts
- **Proactive Scaling**: Prevent outages by scaling before capacity limits are reached

## Installation

### Requirements

- Python 3.9 or higher
- pip package manager
- (Optional) Docker for test database environments

### Core Installation

Install core dependencies for basic functionality:

```bash
# Clone the repository
git clone https://github.com/your-org/Database-Capacity-Planner.git
cd Database-Capacity-Planner

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Full Installation (with Optional ML Libraries)

For improved forecasting accuracy, install optional ML dependencies:

```bash
# Install all dependencies including Prophet, TensorFlow, PyTorch
pip install prophet==1.1.5
pip install tensorflow==2.15.0
pip install torch==2.1.2
```

### Database Driver Installation

Install drivers only for databases you plan to connect:

```bash
# PostgreSQL (already included in requirements.txt)
pip install psycopg2-binary==2.9.9

# MySQL/MariaDB (already included in requirements.txt)
pip install pymysql==1.1.0

# MongoDB (already included in requirements.txt)
pip install pymongo==4.6.1

# Oracle (optional, requires Oracle Instant Client)
pip install cx_oracle==8.3.0
```

### Verification

Verify installation:

```bash
streamlit --version
python -c "import streamlit, pandas, numpy, plotly; print('Installation successful!')"
```

## Quick Start

### Demo Mode (No Database Required)

Launch the application with synthetic sample data:

```bash
streamlit run app.py
```

This starts the web interface at `http://localhost:8501` with pre-loaded sample data demonstrating all features.

### Basic Workflow

1. **Choose a Scenario**: Select from 6 realistic workload scenarios (E-commerce, SaaS, IoT, Analytics, Game, Financial)
2. **Review Current Metrics**: Examine CPU, memory, storage, I/O trends in interactive charts
3. **Generate Forecast**: Predict resource needs for next 7, 14, 30, or 90 days
4. **Analyze Results**: Review predictions with confidence intervals and model insights
5. **Export Report**: Download PDF or Excel report for stakeholders

### Connecting to a Real Database

To monitor a live database:

1. Navigate to the "Database Connection" section in the sidebar
2. Select your database type (PostgreSQL, MySQL, etc.)
3. Enter connection details:
   - **Host**: Database server address
   - **Port**: Database port (defaults: PostgreSQL=5432, MySQL=3306, MongoDB=27017)
   - **Database**: Database name
   - **Username**: Database user with read permissions
   - **Password**: User password
4. Click "Connect" to start collecting real metrics
5. Historical data will be collected automatically

### Example: PostgreSQL Connection

```python
# Connection parameters in UI:
Database Type: PostgreSQL
Host: localhost
Port: 5432
Database: production_db
Username: readonly_user
Password: ********
```

## Usage Examples

### Example 1: Forecasting Storage Growth

**Scenario**: Predict PostgreSQL storage needs for next 30 days

1. Connect to PostgreSQL database or load historical CSV data
2. Navigate to "Storage Analysis" section
3. Select metric: "Disk Usage (%)"
4. Set forecast horizon: 30 days
5. Set confidence interval: 95%
6. Click "Generate Forecast"
7. Review prediction chart showing expected growth trajectory
8. Check "Recommendations" panel for scaling suggestions

**Expected Output**:
- Current usage: 72%
- Predicted usage (30 days): 84% ± 3%
- Recommendation: Plan capacity expansion in 21 days

### Example 2: Setting Up Anomaly Detection

**Scenario**: Detect unusual CPU spikes in production database

1. Load historical metrics (minimum 90 days recommended)
2. Navigate to "Anomaly Detection" section
3. Select metric: "CPU Usage (%)"
4. Choose detection method: "IsolationForest" (fast) or "LSTM" (accurate)
5. Set sensitivity: Medium (2% false positive rate)
6. Click "Detect Anomalies"
7. Review flagged anomalies with timestamps and severity
8. Export anomaly report for incident correlation

### Example 3: Generating Executive Capacity Report

**Scenario**: Create quarterly capacity report for management

1. Load 90 days of historical data
2. Generate forecasts for all key metrics (CPU, memory, storage, I/O)
3. Navigate to "Reports" section
4. Select report type: "Executive Summary"
5. Choose format: PDF
6. Include sections:
   - Current health score
   - 90-day forecast summary
   - Cost projections
   - Scaling recommendations
7. Click "Generate Report"
8. Download PDF and share with stakeholders

### Example 4: Importing Historical Metrics from CSV

**Scenario**: Analyze metrics exported from monitoring tool

```bash
# CSV format requirements:
# - Column 1: timestamp (ISO 8601 format: 2024-01-15 10:30:00)
# - Columns 2+: metric values (cpu_usage, memory_usage, disk_usage, iops, etc.)
```

1. Prepare CSV file with required format
2. Navigate to "Data Import" section
3. Click "Upload File" and select CSV
4. Map CSV columns to metric names
5. Set date format if non-standard
6. Click "Import" to load data
7. Validate data loaded correctly in "Metrics Overview"

## Configuration

### Metric Thresholds

Default thresholds (configurable in app.py):

- **CPU Usage**: Warning at 70%, Critical at 85%
- **Memory Usage**: Warning at 75%, Critical at 90%
- **Disk Usage**: Warning at 80%, Critical at 95%
- **IOPS**: Warning at 1000 ops/sec, Critical at 2000 ops/sec
- **Throughput**: Warning at 100 MB/s, Critical at 200 MB/s
- **Connections**: Warning at 80 connections, Critical at 95 connections

### Prediction Settings

Adjust forecast parameters in the UI:

- **Forecast Horizon**: 7-90 days (accuracy degrades beyond 30 days)
- **Confidence Interval**: 0.8 to 0.99 (default: 0.95)
- **Models Used**: Select which ML models to include in ensemble
- **Include External Factors**: Toggle economic/seasonal event integration

### External Factors Configuration

To enable real-world external factors (economic events):

1. Obtain free API key from Alpha Vantage: https://www.alphavantage.co/support/#api-key
2. Set environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```
3. Enable "Include External Factors" checkbox in UI
4. External events will be fetched and correlated with capacity predictions

**Note**: Without API key, tool falls back to synthetic external factors for demonstration purposes.

## Demo Mode Scenarios

The tool includes 6 realistic workload scenarios for testing and demonstration:

1. **E-commerce Database**: Daily peaks 8pm-11pm, weekend traffic spikes, Black Friday 10x surge
2. **SaaS Application**: Business hours only (9am-6pm Mon-Fri), consistent patterns
3. **IoT Sensor Network**: Constant 24/7 write-heavy load, aggressive storage growth
4. **Analytics Warehouse**: Batch-heavy queries, ETL windows, end-of-quarter spikes
5. **Game Backend**: Evening peaks 6pm-midnight, weekend 2x load, expansion launch surges
6. **Financial Trading**: Market hours only (9:30am-4pm), volatility-driven spikes

See [docs/SCENARIOS.md](docs/SCENARIOS.md) for detailed scenario characteristics.

## Architecture

High-level component overview:

- **EnhancedCapacityPlanner**: Main orchestrator coordinating all components
- **AdvancedPredictionEngine**: ML ensemble combining 7 forecasting models
- **WorkloadForecaster**: Workload pattern analysis and future load prediction
- **AutoScalingManager**: Resource scaling recommendations based on forecasts
- **RealMetricsCollector**: Live database metrics collection with connection pooling
- **OptimizedDatabaseConnector**: Multi-database connection management
- **DataLoader**: Import historical data from various formats

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

Comprehensive documentation available in the `docs/` directory:

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and component design
- [API.md](docs/API.md) - Code API reference for all classes and methods
- [DATABASES.md](docs/DATABASES.md) - Database-specific configuration guides
- [ML_MODELS.md](docs/ML_MODELS.md) - ML model explanations and accuracy characteristics
- [EXAMPLES.md](docs/EXAMPLES.md) - Practical usage examples and tutorials
- [SCENARIOS.md](docs/SCENARIOS.md) - Detailed workload scenario descriptions
- [ACCURACY.md](docs/ACCURACY.md) - Model accuracy ranges and validation methodology

## Testing

### Running Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run only fast tests (skip integration)
pytest -m "not slow and not integration"

# Run specific test file
pytest tests/test_prediction_engine.py
```

### Test Coverage

Current test coverage: 70%+ on core components

- Core classes (EnhancedCapacityPlanner, AdvancedPredictionEngine): 80%+
- Data loading/export functions: 75%+
- Database connectors: 70%+
- ML model validation: Accuracy metrics tracked in benchmarks/

## Contributing

Contributions welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8, use `black` for formatting
2. **Testing**: Add tests for new features, maintain >70% coverage
3. **Documentation**: Update relevant docs/ files for new features
4. **Pull Requests**: Provide clear description of changes and link to related issues

Run code quality checks before submitting:

```bash
black .
flake8 app.py tests/
mypy app.py
pytest
```

## Support

- **Issues**: Report bugs at [GitHub Issues](https://github.com/your-org/Database-Capacity-Planner/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/your-org/Database-Capacity-Planner/discussions)
- **Email**: support@your-org.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for web UI
- ML models: [Prophet](https://facebook.github.io/prophet/), [TensorFlow](https://www.tensorflow.org/), [XGBoost](https://xgboost.readthedocs.io/)
- Database connectivity: [SQLAlchemy](https://www.sqlalchemy.org/), [PyMongo](https://pymongo.readthedocs.io/)

## Roadmap

Future enhancements:

- Custom scenario builder UI
- Real-time monitoring dashboard mode
- Multi-database comparison reports
- Automated alerting via email/Slack/PagerDuty
- Cloud provider cost estimation (AWS RDS, GCP Cloud SQL, Azure Database)
- Kubernetes integration for auto-scaling

---

**Disclaimer**: Predictions are statistical estimates. Always validate forecasts with your own historical data before making critical capacity decisions.
