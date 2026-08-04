"""
Main entry point for the Agentic Stack Testing Framework

Usage:
    python main.py
"""

import os
import sys
import datetime

from tester import Mastif
from config import ConfigExpert

def main():
    """Main execution function with Mind2Web support and graceful Ctrl+C handling."""
    tester = None

    try:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            config_path = "experiments/example.yaml"

        config = ConfigExpert.get_instance(config_path)
        mode = config.get("test_mode", "standard")

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("ERROR: HF_TOKEN environment variable not set.")
            print("Please set it with: export HF_TOKEN='your_token_here'")
            return 1

        if mode == "mind2web":
            open_ai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            judge_model = config.get("judge_model")
            if not open_ai_key and judge_model.startswith("gpt-"):
                print("ERROR: OPENAI_API_KEY environment variable not set.")
                print("Please set it with: export OPENAI_API_KEY='your_key_here'")
                return 1
            if not anthropic_key and judge_model.startswith("claude-"):
                print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
                print("Please set it with: export ANTHROPIC_API_KEY='your_key_here'")
                return 1

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
        print("\nInterrupted by user (Ctrl+C).", file=sys.stderr)
        return 130
    finally:
        if tester is not None:
            try:
                tester.close()
            except Exception:
                pass

if __name__ == "__main__":
    exit(main())