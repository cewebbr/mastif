"""
Main entry point for the Agentic Stack Testing Framework

Usage:
    python main.py [yaml_config_path]
"""

import os
import sys
import datetime

from tester import Mastif
from mind2web_evaluator import JudgeUnavailableError
from config import ConfigExpert
from experiment_logger import ExperimentLogger
from preflight import PreflightChecker, PreflightError

def main():
    """Main execution function with Mind2Web support and graceful Ctrl+C handling."""
    tester = None
    should_close = True

    try:
        if len(sys.argv) >= 3 and sys.argv[1] in ("csv", "to-csv"):
            ExperimentLogger.export_results_csv(sys.argv[2])
            return 0

        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            config_path = "experiments/example.yaml"

        config = ConfigExpert.get_instance(config_path)
        mode = config.get("test_mode", "standard")

        PreflightChecker(config).run()

        tester = Mastif(config_path)

        if mode == "mind2web":
            print("=" * 70)
            print("MIND2WEB BENCHMARK MODE")
            print("=" * 70)

            tester.run_mind2web_evaluation()
            tester.print_summary()

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./logs/mind2web-results-{timestamp}.json"
            tester.export_mind2web_results(filename)

            print(f"\n{'=' * 70}")
            print("Mind2Web evaluation complete!")
            print(f"Results: {filename}")
            print(f"{'=' * 70}\n")
        else:
            print("=" * 70)
            print("STANDARD TESTING MODE")
            print("=" * 70)

            tester.run_comprehensive_test()
            tester.print_summary()

            print(f"\n{'=' * 70}")
            print("Testing complete! Check logs/ for detailed results.")
            print(f"{'=' * 70}\n")

        return 0
    except KeyboardInterrupt:
        should_close = False
        print("\nInterrupted by user (Ctrl+C).", file=sys.stderr)
        return 130
    except JudgeUnavailableError:
        should_close = False
        return 1
    except PreflightError as error:
        print(f"❌ Preflight failed: {error}", file=sys.stderr)
        return 1
    finally:
        if tester is not None and should_close:
            try:
                tester.close()
            except Exception:
                pass

if __name__ == "__main__":
    exit(main())