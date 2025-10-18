# PrefetchLenz Configuration-Based Interface

Welcome to PrefetchLenz! This guide explains how to run prefetching algorithm simulations without modifying any code. Simply use configuration files to select and tune your preferred algorithm.

## Quick Start

### 1. List Available Algorithms

See all 31 available prefetching algorithms:

```bash
python run.py --list-algorithms
```

Output will show all available algorithms including:
- Linear, Strided, BestOffset (simple stride-based)
- GHB, Correlation, MarkovPredictor (correlation-based)
- SMS, Bingo, Triage (spatial prefetchers)
- Neural, LearnCluster (ML-based)
- And 21 more specialized prefetchers

### 2. Run with a Sample Configuration

Run a prefetcher using one of the provided sample configs:

```bash
# Linear prefetcher (simple, fast)
python run.py --config prefetchlenz/config/configs/sample_linear.yml

# BestOffset (learns best offset per PC)
python run.py --config prefetchlenz/config/configs/sample_bestoffset.yml

# SMS - Spatial Memory Streaming
python run.py --config prefetchlenz/config/configs/sample_sms.yml

# Bingo - Advanced spatial prefetcher
python run.py --config prefetchlenz/config/configs/sample_bingo.yml

# GHB - Global History Buffer (correlation-based)
python run.py --config prefetchlenz/config/configs/sample_ghb.yml

# LearnCluster - ML-based with LSTM
python run.py --config prefetchlenz/config/configs/sample_learncluster.yml
```

### 3. View Configuration Template

See all available configuration options:

```bash
python run.py --template
```

---

## Configuration Files

Configuration files are YAML-based and located in `prefetchlenz/config/configs/`.

### Basic Structure

```yaml
# Select algorithm
algorithm: linear

# Algorithm-specific parameters
algorithm_config:
  PARAMETER_NAME: value
  ANOTHER_PARAM: value

# Where to save logs
log_file: "prefetchlenz/config/logs/my_run.log"

# Log verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level: INFO

# Output: console, file, or both
output_format: both
```

### Creating Your Own Config

1. Copy a sample config:
   ```bash
   cp prefetchlenz/config/configs/sample_linear.yml my_config.yml
   ```

2. Edit `my_config.yml`:
   ```yaml
   algorithm: sms
   algorithm_config:
     FILTER_ENTRIES: 64          # Increase filter size
     ACCUM_ENTRIES: 128          # Increase accumulation entries
     PREFETCH_DEGREE: 8          # More aggressive
   log_file: "my_output.log"
   log_level: DEBUG              # Show detailed debug info
   output_format: both
   ```

3. Run with your config:
   ```bash
   python run.py --config my_config.yml
   ```

---

## Algorithm Guide

### Lightweight Algorithms (Good for Prototyping)

**Linear**
- Simple stride-based prefetcher with hardware stream buffers
- Fast, memory-efficient
- Best for: Sequential patterns, simple workloads
- Config: `prefetchlenz/config/configs/sample_linear.yml`

**BestOffset**
- Tests predefined offsets and learns the best one per PC
- Offline learning phase, then fixed offetch
- Best for: Regular stride patterns
- Config: `prefetchlenz/config/configs/sample_bestoffset.yml`

**Strided**
- Classic stride detection and prefetching
- Very lightweight
- Best for: Linear array accesses

### Medium Complexity Algorithms

**SMS (Spatial Memory Streaming)**
- Detects multi-block spatial patterns within memory regions
- Uses AGT (Active Generation Table) and PHT (Pattern History Table)
- Good accuracy with moderate overhead
- Config: `prefetchlenz/config/configs/sample_sms.yml`

**Bingo**
- Advanced spatial prefetcher with 4 pattern history tables
- High accuracy, moderate memory overhead
- Best for: Complex workloads with diverse patterns
- Config: `prefetchlenz/config/configs/sample_bingo.yml`

**GHB (Global History Buffer)**
- Correlation-based using global access history
- Maps address sequences to deltas
- Good for: Recurring patterns, regular sequences
- Config: `prefetchlenz/config/configs/sample_ghb.yml`

### Advanced Algorithms

**LearnCluster (ML-Based)**
- Machine learning with k-means clustering + LSTM
- Highest accuracy for complex patterns
- Requires PyTorch
- Best for: Offline analysis, complex non-linear patterns
- Config: `prefetchlenz/config/configs/sample_learncluster.yml`

**Neural**
- Neural network-based prefetcher

**Perceptron**
- Perceptron-based prefetch filtering

### Specialized Algorithms

**IPCP** - Indirect Pattern-based Cache Prefetcher (pointer chasing)
**EBCP** - Epoch-Based Correlation Prefetcher
**TCP** - Temporal Correlation Prefetcher
**HDS** - Heap Directed Software prefetching
**Graph** - Graph-based prefetching
**Metadata** - Metadata-driven prefetching
**And 12 more...**

---

## Command Line Options

### Basic Usage

```bash
python run.py --config <config_file>
```

### List Available Algorithms

```bash
python run.py --list-algorithms
```

### Show Configuration Template

```bash
python run.py --template
```

### Override Log Level

Show debug information:
```bash
python run.py --config my_config.yml --log-level DEBUG
```

### Override Log File

Save output to a different file:
```bash
python run.py --config my_config.yml --log-file my_output.log
```

### Combine Options

```bash
python run.py --config prefetchlenz/config/configs/sample_sms.yml \
              --log-level DEBUG \
              --log-file detailed_run.log
```

---

## Log File Format

### Log Line Format

```
[YYYY-MM-DD HH:MM:SS] [LEVEL] logger_name - message
```

### Example

```
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Loading configuration from: prefetchlenz/config/configs/sample_linear.yml
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Configuration loaded and validated successfully
[2024-10-18 15:42:32] [INFO] prefetchlenz.prefetchingalgorithm.impl.linear - Initializing Linear prefetcher
[2024-10-18 15:42:32] [DEBUG] prefetchlenz.prefetchingalgorithm.impl.linear - Stream buffer initialized: 32 entries
[2024-10-18 15:42:40] [INFO] prefetchlenz.prefetchingalgorithm.impl.linear - Total accesses processed: 1000
[2024-10-18 15:42:40] [INFO] prefetchlenz.prefetchingalgorithm.impl.linear - Total prefetch hits: 847
[2024-10-18 15:42:40] [INFO] prefetchlenz.prefetchingalgorithm.impl.linear - Prefetch accuracy: 84.7%
```

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed algorithm state, internal decisions | "Processing access: address=0x1000, pc=0x400100" |
| **INFO** | High-level events, statistics, initialization | "Initializing prefetcher", "Accuracy: 84.7%" |
| **WARNING** | Potential issues, unusual patterns | "Configuration deprecated", "Unusual memory pattern" |
| **ERROR** | Errors that don't stop execution | "Failed to load cache policy" |
| **CRITICAL** | Fatal errors | "Algorithm initialization failed" |

### What to Look For in Logs

1. **Initialization Phase**:
   - Algorithm loaded successfully
   - Configuration parameters applied
   - Data structures initialized

2. **Processing Phase**:
   - Number of memory accesses processed
   - Pattern detected messages (algorithm-specific)

3. **Statistics Phase**:
   - Total accesses and prefetch hits/misses
   - Accuracy percentage
   - Algorithm-specific metrics

See `prefetchlenz/config/logs/sample_run.log` for a complete example.

---

## Common Use Cases

### 1. Quick Test of Linear Prefetcher

```bash
python run.py --config prefetchlenz/config/configs/sample_linear.yml
```

Output log will show:
- Linear prefetcher initialized
- Sequential memory access pattern detected
- Final accuracy metrics

### 2. Debug SMS with Detailed Output

```bash
python run.py --config prefetchlenz/config/configs/sample_sms.yml \
              --log-level DEBUG \
              --log-file sms_debug.log
```

### 3. Compare Algorithms

Run each sample config and compare results:
```bash
# Linear
python run.py --config prefetchlenz/config/configs/sample_linear.yml --log-file linear_result.log

# SMS
python run.py --config prefetchlenz/config/configs/sample_sms.yml --log-file sms_result.log

# Bingo
python run.py --config prefetchlenz/config/configs/sample_bingo.yml --log-file bingo_result.log

# Compare log files: tail -20 *_result.log
```

### 4. Tune Algorithm Parameters

Edit the config file to try different parameters:

```yaml
# Original
algorithm_config:
  PREFETCH_DEGREE: 4
  FILTER_ENTRIES: 32

# Modified for more aggressive prefetching
algorithm_config:
  PREFETCH_DEGREE: 8
  FILTER_ENTRIES: 64
```

Then run and compare results.

### 5. Use ML-Based Prefetcher for Offline Analysis

```bash
python run.py --config prefetchlenz/config/configs/sample_learncluster.yml \
              --log-level DEBUG
```

---

## Troubleshooting

### "Config file not found" Error

```
Error: Config file not found: my_config.yml
```

**Solution**: Use absolute path or relative path from project root:
```bash
python run.py --config prefetchlenz/config/configs/sample_linear.yml
```

### "Unknown algorithm" Error

```
ValueError: Unknown algorithm: my_algo
```

**Solution**: Check available algorithms:
```bash
python run.py --list-algorithms
```

### Log File Not Being Created

Ensure the directory exists:
```bash
mkdir -p prefetchlenz/config/logs
```

Or update config to use a valid path:
```yaml
log_file: "my_logs/output.log"
```

### "Could not find Prefetcher class" Error

The algorithm module exists but class name doesn't match. Check the algorithm implementation file or contact maintainers.

---

## Configuration Parameters Reference

### Universal Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | required | Algorithm name (use --list-algorithms to see all) |
| `log_file` | string | null | Path to log file output |
| `log_level` | string | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `output_format` | string | console | console, file, or both |
| `random_seed` | integer | null | Seed for reproducibility (ML algorithms) |

### Algorithm-Specific Parameters

Each algorithm supports custom parameters. Check the sample config files for examples:

- **Linear**: `STREAM_BUFFER_SIZE`, `PREFETCH_DEGREE`
- **SMS**: `FILTER_ENTRIES`, `ACCUM_ENTRIES`, `PHT_NUM_SETS`, `PHT_ASSOC`
- **Bingo**: `PATTERN_LEN`, `REGION_SIZE`, `PHT_SIZE`, `PREFETCH_DEGREE`
- **GHB**: `INDEX_TABLE_SIZE`, `CORRELATION_TABLE_SIZE`, `GHB_SIZE`
- **LearnCluster**: `NUM_CLUSTERS`, `KMEANS_ITERS`, `HIDDEN_DIM`, `TOPK`, `DEVICE`

For a complete list, see the comments in sample config files or the algorithm source code.

---

## Files and Directories

```
prefetchlenz/config/
├── config_loader.py              # Configuration parser (internal)
├── config_template.yml           # Template with all options
├── INTERFACE.md                  # This file
│
├── configs/                      # User configuration files
│   ├── sample_linear.yml         # Sample Linear prefetcher config
│   ├── sample_bestoffset.yml     # Sample BestOffset config
│   ├── sample_sms.yml            # Sample SMS config
│   ├── sample_bingo.yml          # Sample Bingo config
│   ├── sample_ghb.yml            # Sample GHB config
│   └── sample_learncluster.yml   # Sample LearnCluster (ML) config
│
└── logs/                         # Log file output directory
    └── sample_run.log            # Example log file output

run.py                            # Main entry point (root directory)
```

---

## Examples

### Example 1: Run Linear Prefetcher (Default Sample)

```bash
$ python run.py --config prefetchlenz/config/configs/sample_linear.yml
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Loading configuration from: prefetchlenz/config/configs/sample_linear.yml
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Configuration loaded and validated successfully
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Loading algorithm: linear
[2024-10-18 15:42:31] [INFO] prefetchlenz.config.config_loader - Successfully instantiated linear prefetcher
[2024-10-18 15:42:32] [INFO] prefetchlenz.analyzer.analyzer - Analysis complete
[2024-10-18 15:42:32] [INFO] prefetchlenz.analyzer.analyzer -   Correct predictions: 847
[2024-10-18 15:42:32] [INFO] prefetchlenz.analyzer.analyzer -   Incorrect predictions: 153
[2024-10-18 15:42:32] [INFO] prefetchlenz.analyzer.analyzer -   Accuracy: 84.70%
```

### Example 2: Create Custom Config for SMS with Debug Output

Create `my_sms_config.yml`:
```yaml
algorithm: sms

algorithm_config:
  FILTER_ENTRIES: 64              # Increase from default
  ACCUM_ENTRIES: 128
  PREDICTION_THRESHOLD: 2

log_file: "my_sms_run.log"
log_level: DEBUG                  # Show detailed debug info
output_format: both
```

Run it:
```bash
python run.py --config my_sms_config.yml
```

Output will include detailed debug messages showing pattern detection.

### Example 3: List and Compare Algorithms

```bash
# See all algorithms
python run.py --list-algorithms

# Run 3 different prefetchers
python run.py --config prefetchlenz/config/configs/sample_linear.yml --log-file test_linear.log
python run.py --config prefetchlenz/config/configs/sample_sms.yml --log-file test_sms.log
python run.py --config prefetchlenz/config/configs/sample_bingo.yml --log-file test_bingo.log

# Compare final accuracies
grep "Accuracy:" test_*.log
```

---

## Tips and Best Practices

1. **Start Simple**: Begin with Linear or BestOffset to understand the framework
2. **Use Sample Configs**: Each sample config is tuned for typical workloads
3. **Enable Debug for Tuning**: Use `--log-level DEBUG` when experimenting
4. **Check Sample Log**: See `prefetchlenz/config/logs/sample_run.log` for expected output format
5. **Create Custom Configs**: Copy samples and tweak parameters incrementally
6. **Monitor Log Output**: Watch logs to understand what algorithms are doing
7. **Use Reproducible Seeds**: Set `random_seed` for consistent ML algorithm results

---

## Getting Help

- **List algorithms**: `python run.py --list-algorithms`
- **Show template**: `python run.py --template`
- **Check sample configs**: Look in `prefetchlenz/config/configs/`
- **View sample log**: See `prefetchlenz/config/logs/sample_run.log`
- **Report issues**: Use project issue tracker

---

## Next Steps

Now you can:
1. Run the sample configurations as shown above
2. Create your own configurations by copying and modifying samples
3. Tune algorithm parameters to optimize for your workloads
4. Compare different algorithms by running each and analyzing logs
5. Integrate custom memory traces (see input_trace option in config_template.yml)

Happy prefetching! 🚀
