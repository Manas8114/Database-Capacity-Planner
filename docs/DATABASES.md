# Database Configuration Guide

## Supported Databases

Database Capacity Planner supports 6 different database engines with varying levels of metric collection capabilities.

| Database | Advanced Metrics | Default Port | Driver |
|----------|-----------------|--------------|--------|
| PostgreSQL | ✅ Yes | 5432 | psycopg2-binary |
| MySQL | ✅ Yes | 3306 | pymysql |
| Oracle | ✅ Yes | 1521 | cx_oracle |
| MariaDB | ✅ Yes | 3306 | pymysql |
| SQLite | ⚠️ Limited | N/A (file-based) | built-in |
| MongoDB | ⚠️ Limited | 27017 | pymongo |

**Advanced Metrics**: CPU usage, memory usage, disk I/O, query times, buffer statistics, replication lag
**Limited Metrics**: Basic storage and connection metrics only

---

## PostgreSQL

### Prerequisites

```bash
pip install psycopg2-binary==2.9.9
```

### Connection Configuration

**UI Configuration**:
- Database Type: `PostgreSQL`
- Host: `localhost` or server IP/hostname
- Port: `5432` (default)
- Database: Your database name
- Username: Database user with monitoring privileges
- Password: User password

**Code Example**:
```python
from dataclasses import dataclass

connection = DatabaseConnection(
    db_type='postgresql',
    host='prod-db.example.com',
    port=5432,
    database='production_db',
    username='monitoring_user',
    password='secure_password',
    ssl_enabled=True  # Recommended for remote connections
)
```

### Required Permissions

The monitoring user needs these privileges:

```sql
-- Create monitoring user
CREATE USER monitoring_user WITH PASSWORD 'secure_password';

-- Grant connection permission
GRANT CONNECT ON DATABASE production_db TO monitoring_user;

-- Grant read access to system views
GRANT pg_read_all_stats TO monitoring_user;

-- For detailed query statistics
GRANT SELECT ON pg_stat_statements TO monitoring_user;

-- For replication monitoring
GRANT pg_monitor TO monitoring_user;  -- PostgreSQL 10+
```

### Collected Metrics

**Performance**:
- `cpu_usage`: CPU utilization percentage
- `memory_usage`: Shared buffers and work memory usage
- `connection_count`: Active client connections
- `query_time`: Average query execution time
- `buffer_hit_ratio`: Cache hit rate (target: >95%)
- `long_transactions`: Transactions running >5 minutes

**Storage**:
- `disk_usage`: Database size and disk usage
- `data_size`: Actual data size (excluding indexes)
- `index_size`: Total index size
- `temp_usage`: Temporary file usage
- `largest_table_size`: Biggest table in database

**I/O**:
- `iops`: Blocks read/written per second
- `read_throughput`: Read MB/s
- `write_throughput`: Write MB/s

**Replication** (if configured):
- `replication_lag`: Replication delay in seconds
- `replication_status`: Replica health

### Troubleshooting

**Issue**: `permission denied for table pg_stat_statements`
**Solution**: Run `CREATE EXTENSION pg_stat_statements;` as superuser and grant SELECT

**Issue**: SSL connection failed
**Solution**: Set `ssl_enabled=True` and ensure PostgreSQL has `ssl = on` in postgresql.conf

**Issue**: Too many connections
**Solution**: Check `max_connections` setting and ensure monitoring user isn't consuming connection pool

---

## MySQL / MariaDB

### Prerequisites

```bash
pip install pymysql==1.1.0
```

### Connection Configuration

**UI Configuration**:
- Database Type: `MySQL` or `MariaDB`
- Host: `localhost` or server IP/hostname
- Port: `3306` (default)
- Database: Your database name
- Username: Database user with monitoring privileges
- Password: User password

**Code Example**:
```python
connection = DatabaseConnection(
    db_type='mysql',  # or 'mariadb'
    host='mysql-server.example.com',
    port=3306,
    database='app_database',
    username='monitor',
    password='password123'
)
```

### Required Permissions

```sql
-- Create monitoring user
CREATE USER 'monitor'@'%' IDENTIFIED BY 'password123';

-- Grant necessary privileges
GRANT PROCESS ON *.* TO 'monitor'@'%';
GRANT SELECT ON performance_schema.* TO 'monitor'@'%';
GRANT SELECT ON information_schema.* TO 'monitor'@'%';
GRANT REPLICATION CLIENT ON *.* TO 'monitor'@'%';

FLUSH PRIVILEGES;
```

### Collected Metrics

**Performance**:
- `cpu_usage`: Derived from thread CPU time
- `memory_usage`: InnoDB buffer pool usage
- `connection_count`: Current connections
- `query_time`: Slow query log analysis
- `buffer_hit_ratio`: InnoDB buffer pool hit rate
- `network_latency`: Connection response time

**Storage**:
- `disk_usage`: Data and index file sizes
- `data_size`: TABLE_ROWS × AVG_ROW_LENGTH
- `index_size`: INDEX_LENGTH from information_schema
- `temp_usage`: Temporary table usage

**I/O**:
- `iops`: InnoDB I/O operations
- `read_throughput`: InnoDB read MB/s
- `write_throughput`: InnoDB write MB/s

### Troubleshooting

**Issue**: `Access denied for user 'monitor'@'host'`
**Solution**: Check user privileges and host access. Use `'monitor'@'%'` for any host or specific IP.

**Issue**: `performance_schema` not available
**Solution**: Enable performance_schema in my.cnf: `performance_schema = ON`, then restart MySQL

**Issue**: No query time metrics
**Solution**: Enable slow query log: `SET GLOBAL slow_query_log = 'ON'; SET GLOBAL long_query_time = 1;`

---

## Oracle

### Prerequisites

**Oracle Instant Client**:
- Download from: https://www.oracle.com/database/technologies/instant-client/downloads.html
- Install appropriate version for your OS
- Set environment variable: `LD_LIBRARY_PATH=/path/to/instantclient`

**Python Driver**:
```bash
pip install cx_oracle==8.3.0
```

### Connection Configuration

**UI Configuration**:
- Database Type: `Oracle`
- Host: Oracle server hostname
- Port: `1521` (default)
- Database: Service name or SID
- Username: Oracle user with monitoring privileges
- Password: User password

**Code Example**:
```python
connection = DatabaseConnection(
    db_type='oracle',
    host='oracle-db.example.com',
    port=1521,
    database='ORCL',  # Service name or SID
    username='SYSTEM',
    password='oracle_password'
)
```

### Required Permissions

```sql
-- Create monitoring user
CREATE USER monitor IDENTIFIED BY password123;

-- Grant necessary privileges
GRANT CREATE SESSION TO monitor;
GRANT SELECT ANY DICTIONARY TO monitor;
GRANT SELECT ON V_$SESSION TO monitor;
GRANT SELECT ON V_$SESSTAT TO monitor;
GRANT SELECT ON V_$STATNAME TO monitor;
GRANT SELECT ON V_$SYSSTAT TO monitor;
GRANT SELECT ON V_$SYSTEM_EVENT TO monitor;
GRANT SELECT ON DBA_DATA_FILES TO monitor;
GRANT SELECT ON DBA_FREE_SPACE TO monitor;
GRANT SELECT ON V_$DATABASE TO monitor;
```

### Collected Metrics

**Performance**:
- `cpu_usage`: CPU usage from V$SYSSTAT
- `memory_usage`: SGA and PGA usage
- `connection_count`: Active sessions from V$SESSION
- `buffer_hit_ratio`: Buffer cache hit ratio
- `library_cache_hit_ratio`: Shared pool efficiency

**Storage**:
- `disk_usage`: Tablespace usage from DBA_DATA_FILES
- `data_size`: Actual data size across tablespaces
- `temp_usage`: Temporary tablespace usage

**I/O**:
- `iops`: Physical reads/writes per second
- `read_throughput`: Read I/O MB/s
- `write_throughput`: Write I/O MB/s

### Troubleshooting

**Issue**: `DPI-1047: Cannot locate a 64-bit Oracle Client library`
**Solution**: Install Oracle Instant Client and set LD_LIBRARY_PATH correctly

**Issue**: `ORA-12541: TNS:no listener`
**Solution**: Verify Oracle listener is running: `lsnrctl status`

**Issue**: `ORA-01017: invalid username/password`
**Solution**: Verify credentials and ensure user has CREATE SESSION privilege

---

## MongoDB

### Prerequisites

```bash
pip install pymongo==4.6.1
```

### Connection Configuration

**UI Configuration**:
- Database Type: `MongoDB`
- Host: `localhost` or server hostname
- Port: `27017` (default)
- Database: Database name
- Username: MongoDB user (optional if no auth)
- Password: User password (optional)

**Code Example**:
```python
# With authentication
connection = DatabaseConnection(
    db_type='mongodb',
    host='mongo-server.example.com',
    port=27017,
    database='app_db',
    username='admin',
    password='mongo_password'
)

# Without authentication (local development)
connection = DatabaseConnection(
    db_type='mongodb',
    host='localhost',
    port=27017,
    database='test_db',
    username='',
    password=''
)
```

### Required Permissions

```javascript
// Create monitoring user
use admin
db.createUser({
  user: "monitor",
  pwd: "password123",
  roles: [
    { role: "read", db: "admin" },
    { role: "read", db: "local" },
    { role: "clusterMonitor", db: "admin" }
  ]
})
```

### Collected Metrics

**Performance** (Limited):
- `connection_count`: Current connections from serverStatus
- `memory_usage`: Resident memory usage
- `operation_count`: Operations per second

**Storage**:
- `disk_usage`: Data + index size from db.stats()
- `data_size`: Actual document size
- `index_size`: All indexes size
- `collection_count`: Number of collections

**I/O** (Basic):
- `operations_per_sec`: Commands executed

**Note**: MongoDB has limited metric collection compared to relational databases. For comprehensive monitoring, consider MongoDB Atlas or Ops Manager.

### Troubleshooting

**Issue**: `Authentication failed`
**Solution**: Verify username/password and ensure user has correct roles

**Issue**: `Connection refused on port 27017`
**Solution**: Check MongoDB is running: `systemctl status mongod`

**Issue**: `ServerSelectionTimeoutError`
**Solution**: Verify host/port and network connectivity. Check firewall rules.

---

## SQLite

### Prerequisites

SQLite support is built into Python (no additional installation required).

### Connection Configuration

**UI Configuration**:
- Database Type: `SQLite`
- Host: Leave empty or use `localhost`
- Port: Leave empty
- Database: Full path to .db file (e.g., `/path/to/database.db`)
- Username: Not required
- Password: Not required

**Code Example**:
```python
connection = DatabaseConnection(
    db_type='sqlite',
    host='',
    port=0,
    database='/var/data/app_database.db',
    username='',
    password=''
)
```

### Collected Metrics

**Storage** (Limited):
- `disk_usage`: Database file size
- `data_size`: Estimated data size (page_count × page_size)
- `index_size`: Rough index size estimate

**Note**: SQLite is a file-based database with very limited runtime metrics. Not recommended for production capacity planning. Use for development/testing only.

### Troubleshooting

**Issue**: `OperationalError: unable to open database file`
**Solution**: Verify file path is correct and application has read permissions

**Issue**: `database is locked`
**Solution**: Close other connections. SQLite has limited concurrency support.

---

## Connection Best Practices

### Security

1. **Use Read-Only Users**: Create dedicated monitoring users with minimal privileges
2. **Enable SSL/TLS**: For remote connections, always use encrypted connections
3. **Rotate Credentials**: Regularly update monitoring user passwords
4. **Restrict Network Access**: Use firewall rules to limit monitoring connections to specific IPs
5. **Avoid Root/Admin Users**: Never use superuser accounts for monitoring

### Performance

1. **Connection Pooling**: Tool automatically reuses connections
2. **Limit Collection Frequency**: Collect metrics every 5-15 minutes (default: 5 min)
3. **Avoid Peak Hours**: Schedule initial historical data collection during off-peak
4. **Use Replica for Metrics**: Collect from read replica if available to reduce primary load

### Monitoring Impact

Typical overhead of metric collection:

| Database | CPU Impact | I/O Impact | Network Impact |
|----------|-----------|------------|----------------|
| PostgreSQL | <1% | Minimal | <100 KB/min |
| MySQL | <1% | Minimal | <100 KB/min |
| Oracle | <2% | Low | <200 KB/min |
| MariaDB | <1% | Minimal | <100 KB/min |
| MongoDB | <0.5% | Minimal | <50 KB/min |
| SQLite | <0.1% | Minimal | N/A (local) |

---

## Testing Database Connections

### Quick Connection Test

Use this script to verify database connectivity before running full tool:

```python
import sys
from sqlalchemy import create_engine, text

def test_connection(db_type, host, port, database, username, password):
    try:
        if db_type == 'postgresql':
            url = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
        elif db_type == 'mysql':
            url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
        # Add other database types as needed

        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✅ Connection successful to {db_type} at {host}:{port}")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

# Test PostgreSQL
test_connection('postgresql', 'localhost', 5432, 'mydb', 'user', 'password')
```

---

## Common Issues Across All Databases

### Network Issues

**Symptoms**: Connection timeout, connection refused
**Solutions**:
- Verify database server is running
- Check firewall rules allow connection on database port
- Test connectivity: `telnet host port` or `nc -zv host port`
- Verify hostname resolution: `nslookup hostname`

### Authentication Issues

**Symptoms**: Access denied, invalid credentials
**Solutions**:
- Verify username and password are correct
- Check user has necessary permissions (see database-specific sections)
- Ensure user can connect from your IP (`host` setting in user grants)

### Permission Issues

**Symptoms**: Access denied to system tables/views
**Solutions**:
- Grant additional permissions as documented above
- For PostgreSQL: Use `pg_read_all_stats` role
- For MySQL: Grant `PROCESS` and access to `performance_schema`
- For Oracle: Grant `SELECT ANY DICTIONARY`

---

For API usage examples, see [API.md](API.md).
For architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).
