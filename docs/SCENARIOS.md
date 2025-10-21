# Workload Scenarios

## Overview

Database Capacity Planner includes 6 realistic workload scenarios for demonstration and testing. Each scenario simulates characteristic patterns of real-world database workloads based on industry observations and typical usage patterns.

**Purpose**:
- Test forecasting accuracy across different workload types
- Demonstrate tool capabilities without requiring live database
- Educate users on typical database capacity patterns
- Provide reproducible test cases for validation

---

## Scenario 1: E-commerce Database

### Overview

Simulates an online retail platform database serving a consumer-facing e-commerce application.

**Typical Organizations**: Shopify stores, Amazon marketplace sellers, retail websites

### Traffic Characteristics

**Daily Pattern**:
- Morning (6am-12pm): Steady increase, 50% → 80% of baseline
- Afternoon (12pm-6pm): Sustained moderate load, 80% → 100%
- **Peak Evening (8pm-11pm)**: Shopping peak, 300% of baseline
- Late Night (11pm-6am): Drop to 20-30% of baseline

**Weekly Pattern**:
- **Weekends**: 50% higher than weekdays (people shop on Sat/Sun)
- Friday evening: Traffic starts climbing
- Monday: Slight morning surge (weekend cart checkouts)

**Seasonal Spikes**:
- **Black Friday**: 10x normal load (sustained 12-24 hours)
- **Cyber Monday**: 8x normal load
- **Holiday Season** (Nov-Dec): Baseline 3x higher
- **Summer Sale** (July): 2x spike for 1 week

### Metric Patterns

#### CPU Usage
- **Baseline**: 30-40% during business hours
- **Evening Peak**: 90-120% (throttled at 100%)
- **Weekend Multiplier**: 1.5x
- **Black Friday**: Sustained 95-100% for 24 hours

#### Memory Usage
- Correlates strongly with CPU (r=0.85)
- **Baseline**: 45-55%
- **Peak**: 85-90%
- Higher correlation due to session caching

#### Storage (Disk Usage)
- **Growth Rate**: 5 GB/day baseline
- **Black Friday Jump**: +50 GB (new orders, images)
- **Linear Trend**: Predictable accumulation of order history
- **Retention**: No cleanup (e-commerce keeps all order data)

#### I/O Patterns
- **Read-heavy**: 70% reads, 30% writes
- **Checkout Spikes**: Write throughput 5x during payment processing
- **IOPS Baseline**: 800-1200 ops/sec
- **Peak IOPS**: 4000-5000 ops/sec

#### Connections
- **Baseline**: 50-100 connections
- **Peak**: 500-800 connections (concurrent shoppers)
- **Bot Traffic**: Occasional spikes to 1000+ (scrapers, bots)

### Anomalies

**Flash Sale Events** (5% probability per week):
- Duration: 2 hours
- Magnitude: 5-10x CPU and connections
- Recovery: Gradual over 1 hour after sale ends

**Product Launch** (2% probability per week):
- Duration: 24 hours
- Magnitude: 4x baseline load
- Pattern: Spike at launch (noon), sustained elevation

**Bot Attack** (1% probability per day):
- Duration: 30 minutes to 2 hours
- Characteristics: CPU spike + abnormally high connection count
- Mitigation: WAF kicks in, load drops suddenly

### Growth Model

**User Base Growth**: Exponential at 20% per quarter
```
Storage(t) = 5GB/day × (1 + 0.20)^(quarter)
```

### When to Use This Scenario

- **Testing seasonal forecasting**: Strong holiday patterns
- **Peak capacity planning**: Handling traffic spikes
- **E-commerce industry**: Most relevant pattern
- **Read-heavy workloads**: Typical catalog browsing

---

## Scenario 2: SaaS Application Database

### Overview

Simulates a Software-as-a-Service business application database (e.g., project management, CRM, accounting software).

**Typical Organizations**: Asana, Salesforce, QuickBooks Online, Jira

### Traffic Characteristics

**Daily Pattern**:
- **Business Hours Only** (9am-6pm Mon-Fri): 90% of all traffic
- Morning Surge (9am-10am): Login spike, 80% → 100%
- Lunch Dip (12pm-1pm): 70% of peak
- Afternoon Plateau (2pm-5pm): Steady 90-100%
- Evening/Night: 5-10% (automated tasks, offshore teams)

**Weekly Pattern**:
- Mon-Thu: Consistent high usage
- **Friday**: 30% lower (half-days, end-of-week slowdown)
- **Weekends**: 90% drop (minimal activity)

**Seasonal Patterns**:
- **Summer Vacation** (July-Aug): 20% lower baseline
- **Christmas Week**: 40% drop (offices closed)
- **End-of-Quarter** (Mar/Jun/Sep/Dec): 15% spike (reporting, closing books)
- **January**: Surge (new year planning)

### Metric Patterns

#### CPU Usage
- **Business Hours**: 45-55%
- **Off-Hours**: 5-10% (background jobs)
- **End-of-Month**: Brief spikes to 75% (report generation)
- Very predictable, consistent pattern

#### Memory Usage
- Moderate correlation with CPU (r=0.60)
- **Business Hours**: 50-60%
- **Off-Hours**: 30-40% (cached data persists)
- Background jobs decouple memory from CPU at night

#### Storage
- **Growth Rate**: 2 GB/day (steady, linear)
- **Monthly Cleanup**: Sudden 10% drop (log rotation, temp file cleanup)
- **Predictable**: Very linear, easy to forecast
- **Retention Policy**: 90-day rolling window for some tables

#### I/O Patterns
- **Balanced**: 55% reads, 45% writes (collaborative editing)
- **Batch Processing**: Midnight-2am, I/O spikes (backups, ETL)
- **IOPS**: 600-1000 ops/sec during business hours
- **Throughput**: Moderate, consistent

#### Connections
- **Business Hours**: 200-300 connections (concurrent users)
- **Off-Hours**: 10-20 connections (maintenance)
- **Steady**: Very predictable connection count

### Anomalies

**Deployment Windows** (every 2 weeks, Sunday 3am):
- Duration: 30 minutes
- Pattern: CPU spike 60% → 95% (migrations, cache warming)
- Recovery: Immediate return to baseline

**Batch Processing** (nightly, midnight-2am):
- Duration: 2 hours
- Pattern: I/O spike, CPU moderate
- Purpose: Report generation, data exports, backups

**Incident** (0.5% probability per day):
- Duration: 1-2 hours
- Pattern: Sudden 10x increase in logs/errors
- Characteristic: High CPU, low throughput (database lock)
- Recovery: Gradual over 1-2 hours after fix

### Growth Model

**User Base Growth**: Linear, steady customer acquisition
```
Storage(t) = 60 GB/month × months + monthly_cleanup
```

### When to Use This Scenario

- **Business hours patterns**: Testing time-of-day forecasting
- **Predictable workloads**: Easiest to forecast accurately
- **SaaS industry**: Representative of B2B applications
- **Balanced workloads**: Mix of reads and writes

---

## Scenario 3: IoT Sensor Network Database

### Overview

Simulates time series database for IoT sensor data (e.g., industrial monitoring, smart home, environmental sensors).

**Typical Organizations**: InfluxData users, Timescale DB, industrial IoT platforms

### Traffic Characteristics

**Daily Pattern**:
- **Constant 24/7**: Sensors transmit continuously
- **No Variation**: Minimal hourly differences (±5%)
- **Extremely Predictable**: Flat baseline

**Weekly Pattern**:
- **No Weekly Cycle**: Sensors don't take weekends off
- Uniform across all days

**Seasonal Patterns**:
- **Summer**: +10% sensor activity (warm weather = more active sensors)
- **Winter**: Baseline (some outdoor sensors offline in cold)

### Metric Patterns

#### CPU Usage
- **Very Stable**: 20% ±3%
- **Write-Optimized**: Low CPU for time series inserts
- **Aggregation Spikes**: Hourly rollups cause brief 40% spikes

#### Memory Usage
- **High**: 70-80% (large write buffers)
- **Stable**: Little variation
- **Purpose**: Buffer sensor data before batch writes

#### Storage
- **Aggressive Growth**: 10 GB/day (time series data accumulates fast)
- **Retention Policy**: 90-day window
- **Sawtooth Pattern**: Growth then sudden drop (data purge)
- **Lifecycle**: Day 1-90 (growth), Day 90 (purge 30 days), repeat

**Storage Pattern Example**:
```
Day 1-30:  300 GB → 600 GB
Day 30:    600 GB → 300 GB (purge)
Day 31-60: 300 GB → 600 GB
Day 60:    600 GB → 300 GB (purge)
```

#### I/O Patterns
- **Write-Heavy**: 90% writes, 10% reads
- **High Throughput**: 200-300 MB/s write
- **IOPS**: 5000-8000 ops/sec (batch inserts)
- **Reads**: Occasional dashboard queries, analytics

#### Connections
- **Low Count**: 10-30 connections (batch ingest processes)
- **Persistent**: Long-lived connections
- **Minimal Variation**: Very stable

### Anomalies

**Sensor Flood** (1% probability per day):
- Duration: 5 minutes
- Cause: Malfunctioning sensor sending data at 100x rate
- Pattern: Write spike 10,000 MB/s, IOPS 50,000+
- Resolution: Automatic rate limiting kicks in

**Network Partition** (0.5% probability per week):
- Duration: 30 minutes to 2 hours
- Pattern: Sudden 30% drop (sensor group disconnected)
- Recovery: Surge 2x normal (backfill missed data)

**Aggregation Jobs** (hourly, :00 minute):
- Duration: 2-3 minutes
- Pattern: Brief CPU spike 20% → 45%
- Purpose: Roll up raw data to hourly summaries

### Growth Model

**Sensor Fleet Growth**: Linear, adding 10,000 sensors/month
```
Data_rate(t) = baseline_rate × (1 + 0.05 × months)
Storage(t) = data_rate × 90 days (retention window)
```

### When to Use This Scenario

- **Time series databases**: InfluxDB, TimescaleDB patterns
- **High-throughput writes**: Testing write-heavy forecasting
- **Retention policies**: Modeling sawtooth storage patterns
- **IoT industry**: Representative sensor data workload

---

## Scenario 4: Analytics Warehouse Database

### Overview

Simulates data warehouse for business intelligence and analytics (e.g., Redshift, BigQuery, Snowflake).

**Typical Organizations**: Enterprise BI teams, data analytics platforms

### Traffic Characteristics

**Daily Pattern**:
- **Batch-Heavy**: Large queries at scheduled times
- Morning Reports (8am): Heavy load (executive dashboards)
- Midday (12pm): Ad-hoc analysis
- Evening (5pm): End-of-day reports
- **ETL Window** (2am-5am): Nightly data loads

**Weekly Pattern**:
- **Mon-Wed**: Heavy (weekly reports, analysis)
- Thu-Fri: Lighter (reporting done)
- Weekends: Minimal (10% of weekday)

**Seasonal Patterns**:
- **End-of-Quarter**: 30% higher (financial close, board reports)
- **End-of-Year**: 50% higher (annual reports, planning)

### Metric Patterns

#### CPU Usage
- **Baseline**: 30% (idle, light queries)
- **Query Spikes**: 30% → 90% during large queries
- **ETL Window**: Sustained 60-70% (data processing)
- **Highly Variable**: Depends on query workload

#### Memory Usage
- **Query-Driven**: 60-75%
- **Purpose**: Cache query results, intermediate data
- **Spikes**: Large joins push to 90%

#### Storage
- **Predictable Growth**: ETL adds 50 GB/day
- **Very Linear**: Daily batch loads
- **Growth Rate**: 1.5 TB/month
- **No Purges**: Data retained indefinitely (cold storage archives)

#### I/O Patterns
- **Read-Heavy During Queries**: 80% reads during business hours
- **Write-Heavy During ETL**: 90% writes 2am-5am
- **Batch Operations**: Large sequential reads/writes
- **IOPS**: Variable, 1000-5000 depending on query

#### Connections
- **Few Concurrent**: 10-20 connections
- **Long-Running**: Queries take minutes to hours
- **Blocking**: Large queries block others

### Anomalies

**Ad-Hoc Analysis** (random during business hours, 3-5 per day):
- Duration: 1-3 hours
- Pattern: CPU 30% → 85%
- Cause: Data scientist running complex query

**ETL Failure and Retry** (1% probability):
- Duration: 2x normal ETL time
- Pattern: Duplicate load attempt
- Effect: 2x I/O for that window

**Data Refresh** (quarterly):
- Duration: 24 hours
- Pattern: Sustained high activity
- Purpose: Rebuild indexes, refresh materialized views

### Growth Model

**Data Accumulation**: Steady append-only
```
Storage(t) = 50 GB/day × days
```

### When to Use This Scenario

- **Data warehouses**: Testing analytical workload forecasting
- **Batch patterns**: Scheduled query workloads
- **BI teams**: Representative analytics usage
- **Read-heavy queries**: Large sequential scans

---

## Scenario 5: Game Backend Database

### Overview

Simulates online multiplayer game backend database (player data, game state, leaderboards).

**Typical Organizations**: Mobile games, MMORPGs, real-time multiplayer games

### Traffic Characteristics

**Daily Pattern**:
- Morning (6am-12pm): Low, 20% of peak
- Afternoon (12pm-6pm): Building, 40% of peak
- **Peak Evening (6pm-midnight)**: Gaming prime time, 100% load
- Late Night (midnight-6am): Sharp drop to 10%

**Weekly Pattern**:
- Weekdays: Moderate
- **Weekends**: 2x weekday load (Sat-Sun people have more time)
- **Friday Night**: Traffic climbs starting 6pm

**Seasonal Spikes**:
- **New Expansion Launch**: 10x baseline for first week
- **Holiday Events** (Christmas, Halloween): 3-4x for event duration
- **Summer**: Lower (people outside, not gaming)

### Metric Patterns

#### CPU Usage
- **Daytime**: 50%
- **Peak Evening**: 150% → throttled to 100%
- **Weekends**: Sustained 90-100%
- **Expansion Launch**: Sustained 95-100% for days

#### Memory Usage
- Strong correlation with player count (r=0.90)
- **Off-Peak**: 40%
- **Peak**: 85%
- **Purpose**: Cache player sessions, game state

#### Storage
- **Player Data + Logs**: 5 GB/day baseline
- **Expansion Jump**: +100 GB (new content, assets)
- **Moderate Growth**: User-generated content accumulates

#### I/O Patterns
- **High IOPS**: Real-time game state updates
- **Write-Heavy**: 60% writes (player actions, state changes)
- **IOPS Baseline**: 2000 ops/sec
- **Peak IOPS**: 10,000 ops/sec

#### Connections
- **Baseline**: 100 connections (off-peak players)
- **Peak**: 1000-1200 connections (concurrent players)
- **Expansion Launch**: 3000+ connections

### Anomalies

**Game Launch Spike** (first week of expansion):
- Duration: 7 days
- Magnitude: 10x baseline
- Pattern: Sustained high load, gradual decay

**Server Events** (in-game boss fight, scheduled 2x per week):
- Duration: 2 hours
- Magnitude: 5x concurrent players
- Pattern: Sharp spike, sudden drop after event

**DDoS Attempt** (0.1% probability per day):
- Duration: 30 minutes
- Pattern: Connection spike to 5000+, CPU maxed
- Mitigation: DDoS protection kicks in

### Growth Model

**User Base**: Burst growth with expansion releases
```
Base_load(t) = steady_state + (expansion_factor × decay_function(days_since_launch))
```

### When to Use This Scenario

- **Gaming industry**: Representative game server patterns
- **Peak concurrency**: Handling evening traffic surges
- **Launch events**: Modeling expansion/release spikes
- **Real-time workloads**: High IOPS, write-heavy

---

## Scenario 6: Financial Trading Database

### Overview

Simulates financial trading platform database (order books, trade history, market data).

**Typical Organizations**: Stock brokerages, cryptocurrency exchanges, trading platforms

### Traffic Characteristics

**Daily Pattern**:
- **Market Hours Only** (9:30am-4pm ET Mon-Fri): 100% of load
- **Off-Hours**: <2% (system maintenance only)
- **Very Binary**: On during market hours, off otherwise

**Weekly Pattern**:
- Mon-Fri: Consistent
- **Weekends/Holidays**: Zero load (markets closed)

**Seasonal Patterns**:
- **August/December**: Lower volume (trader vacations)
- **Volatility Events**: Sudden massive spikes (Fed announcements, earnings)

### Metric Patterns

#### CPU Usage
- **Market Hours**: 60%
- **Market News**: Spike to 95% (2-3 times per day)
- **Off-Hours**: 2% (idle)
- **Extreme Binary**: On/off pattern

#### Memory Usage
- **Market Hours**: 55%
- **Off-Hours**: 10%
- **Purpose**: Order book caching, price data

#### Storage
- **Append-Only**: Trade logs accumulate
- **Growth**: 2 GB/day (trade history)
- **Predictable**: Linear, no purging (regulatory retention)

#### I/O Patterns
- **Read-Heavy**: 75% reads (price lookups, order book queries)
- **Write Bursts**: Trade execution causes write spikes
- **IOPS Market Hours**: 2000 ops/sec
- **IOPS Off-Hours**: 50 ops/sec

#### Connections
- **Market Hours**: 500+ connections (active traders)
- **Off-Hours**: 5-10 connections (monitoring)
- **Very Peaked**: Mirrors market hours exactly

### Anomalies

**Market Volatility Event** (VIX spike, 2-3 times per week):
- Duration: 10 minutes to 2 hours
- Magnitude: 10x query rate
- Trigger: Fed announcement, major news
- Pattern: Sharp spike, gradual decay

**Flash Crash** (0.01% probability per day):
- Duration: 10 minutes
- Pattern: Extreme spike (20x) then immediate recovery
- Cause: Algorithmic trading cascade

**Trading Halt** (rare, 0.1% probability):
- Duration: 15 minutes to 2 hours
- Pattern: Sudden drop to 5% of normal
- Cause: Circuit breaker triggered

### Growth Model

**Trade Volume**: Slowly growing with platform adoption
```
Storage(t) = 2 GB/day × days (append-only)
```

### When to Use This Scenario

- **Financial industry**: Trading platform patterns
- **Market hours**: Binary on/off workload testing
- **Volatility handling**: Sudden spike forecasting
- **Regulatory**: Append-only storage modeling

---

## Scenario Comparison Table

| Feature | E-commerce | SaaS | IoT | Analytics | Game | Financial |
|---------|-----------|------|-----|-----------|------|-----------|
| **Daily Peak** | 8-11pm | 9am-6pm | None | 8am, 12pm, 5pm | 6pm-12am | 9:30am-4pm |
| **Weekend Pattern** | Higher | Much lower | Same | Lower | Much higher | None (closed) |
| **Seasonal Spikes** | Holidays | Quarterly | None | End-of-year | Expansions | Volatility |
| **CPU Pattern** | Variable | Predictable | Stable | Spikey | Variable | Binary |
| **Storage Growth** | Exponential | Linear | Sawtooth | Linear | Burst | Linear |
| **I/O Type** | Read-heavy | Balanced | Write-heavy | Read-heavy | Write-heavy | Read-heavy |
| **Connections** | High variable | Moderate stable | Low stable | Low | High variable | Binary on/off |
| **Forecast Difficulty** | Hard | Easy | Easy | Medium | Hard | Medium |

---

## Using Scenarios

### In the UI

1. Launch application: `streamlit run app.py`
2. Navigate to **"Workload Scenario Selector"** in sidebar
3. Choose scenario from dropdown
4. Optional: Adjust growth rate, anomaly frequency, noise level
5. Click **"Initialize with Scenario"**
6. Explore generated data in dashboards

### Customization Options

- **Growth Rate Multiplier**: 0.5x to 2x (scale default growth)
- **Anomaly Frequency**: None, Low (0.5%), Medium (2%), High (5%)
- **Noise Level**: Low, Medium, High (affects data variability)
- **Random Seed**: Set for reproducibility or randomize for variation

### For Testing

Each scenario is useful for testing specific forecasting challenges:

- **E-commerce**: Seasonal forecasting, holiday spikes
- **SaaS**: Business hours patterns, predictable loads
- **IoT**: Retention policies, write-heavy loads
- **Analytics**: Batch processing, ETL windows
- **Game**: Launch events, peak concurrency
- **Financial**: Market hours, volatility spikes

---

For model selection by workload type, see [ML_MODELS.md](ML_MODELS.md).
For practical usage examples, see [EXAMPLES.md](EXAMPLES.md).
