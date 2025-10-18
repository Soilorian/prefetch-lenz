#!/usr/bin/env python3
"""
PrefetchLenz Configuration-Based Runner
Allows users to select and run prefetching algorithms without modifying code.

Usage:
    python run.py --config config/configs/sample_linear.yml
    python run.py --list-algorithms
    python run.py --config config/configs/sample_sms.yml --log-level DEBUG
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from prefetchlenz.config.config_loader import ConfigLoader, setup_logging
from prefetchlenz.analyzer.analyzer import Analyzer
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.access.linearmemoryaccess import LinearMemoryAccess


class SimpleDataLoader:
    """Simple data loader for memory access traces."""

    def __init__(self, data: list):
        self.data = data

    def load(self):
        """Return the array of addresses."""
        return self.data

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def create_sample_trace(num_accesses: int = 100) -> list:
    """
    Create a sample memory access trace for testing.
    This generates a simple linear access pattern with LinearMemoryAccess objects.
    """
    accesses = []
    address = 0x1000
    for i in range(num_accesses):
        # Linear access pattern: sequential addresses
        # Use LinearMemoryAccess for compatibility with Linear algorithm
        access = LinearMemoryAccess(
            address=address + (i * 8),
            pc=0x400100 + (i % 10) * 4,
            loaded_pointer=None  # No pointer chasing in this sample
        )
        accesses.append(access)

    # Add some random pattern switches
    for i in range(20):
        access = LinearMemoryAccess(
            address=0x2000 + (i * 16),
            pc=0x400200 + (i % 5) * 4,
            loaded_pointer=None
        )
        accesses.append(access)

    return accesses


def run_with_config(config_path: str, log_level_override: str = None, log_file_override: str = None):
    """
    Load configuration and run prefetching simulation.

    Args:
        config_path: Path to YAML configuration file
        log_level_override: Override log level from config
        log_file_override: Override log file from config
    """
    try:
        # Load configuration
        loader = ConfigLoader(config_path)
        config = loader.load()

        # Override settings if provided
        if log_level_override:
            config['log_level'] = log_level_override
        if log_file_override:
            config['log_file'] = log_file_override

        # Setup logging
        setup_logging(config)
        logger = logging.getLogger('prefetchlenz')

        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"Algorithm: {config['algorithm']}")

        # Get or create algorithm instance
        algorithm = loader.get_algorithm_instance()

        # Get input trace or generate sample
        input_trace = loader.get_input_trace()
        if input_trace:
            logger.info(f"Loading trace from: {input_trace}")
            # TODO: Implement actual trace loading
            accesses = create_sample_trace(100)
        else:
            logger.info("No input trace specified, generating sample trace")
            accesses = create_sample_trace(100)

        # Run simulation
        logger.info(f"Running analysis with {len(accesses)} memory accesses")
        data_loader = SimpleDataLoader(accesses)
        analyzer = Analyzer(algorithm, data_loader)
        analyzer.run()

        logger.info("Analysis complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_algorithms():
    """List all available algorithms."""
    loader = ConfigLoader("dummy")  # Dummy path, not used for listing
    loader.list_algorithms()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PrefetchLenz Configuration-Based Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available algorithms
  python run.py --list-algorithms

  # Run with a sample configuration
  python run.py --config prefetchlenz/config/configs/sample_linear.yml

  # Run with custom log level (show debug output)
  python run.py --config prefetchlenz/config/configs/sample_sms.yml --log-level DEBUG

  # Run with custom log file output
  python run.py --config prefetchlenz/config/configs/sample_bingo.yml --log-file my_run.log

  # Run sample Linear prefetcher
  python run.py --config prefetchlenz/config/configs/sample_linear.yml

  # Run sample SMS prefetcher
  python run.py --config prefetchlenz/config/configs/sample_sms.yml

  # Run sample GHB prefetcher
  python run.py --config prefetchlenz/config/configs/sample_ghb.yml

  # Run sample LearnCluster (ML-based) prefetcher
  python run.py --config prefetchlenz/config/configs/sample_learncluster.yml
        """)

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help='List all available algorithms and exit'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Override log level from configuration'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Override log file path from configuration'
    )
    parser.add_argument(
        '--template',
        action='store_true',
        help='Show configuration template and exit'
    )

    args = parser.parse_args()

    # Handle --list-algorithms
    if args.list_algorithms:
        list_algorithms()
        return

    # Handle --template
    if args.template:
        template_path = Path(__file__).parent / 'prefetchlenz' / 'config' / 'config_template.yml'
        if template_path.exists():
            print(f"\n{'='*70}")
            print("Configuration Template")
            print(f"{'='*70}\n")
            with open(template_path) as f:
                print(f.read())
        else:
            print(f"Template not found at {template_path}")
        return

    # Handle --config
    if args.config:
        run_with_config(args.config, args.log_level, args.log_file)
    else:
        parser.print_help()
        print("\nError: Please provide --config option or use --list-algorithms", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
