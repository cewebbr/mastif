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
    """Main execution function with Mind2Web support"""
    
    # Check for config file argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "experiments/example.yaml" # Default config file

    # Configuration
    # First singleton initialization requires path to yaml experiment file
    config = ConfigExpert.get_instance(config_path) 
    MODE = config.get("test_mode", "standard")  # "standard" or "mind2web"
    MIND2WEB_NUM_TASKS = config.get("mind2web_num_tasks", 10)
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set!")
        print("Please set it with: export HF_TOKEN='your_token_here'")
        return 1
    
    # Get OpenAI key
    open_ai_key = os.getenv("OPEN_AI_KEY")
    if not open_ai_key:
        print("ERROR: OPEN_AI_KEY environment variable not set!")
        print("Please set it with: export OPEN_AI_KEY='your_key_here'")
        return 1

    # Initialize tester
    tester = Mastif()
    
    if MODE == "mind2web":
        # Run Mind2Web evaluation
        print("="*70)
        print("MIND2WEB BENCHMARK MODE")
        print("="*70)
        
        # Run Mind2Web evaluation
        tester.run_mind2web_evaluation()
        tester.print_summary()
        
        # Export Mind2Web results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./logs/mind2web-results-{timestamp}.json"
        tester.export_mind2web_results(filename)
        
        # Also export standard results
        standard_filename = f"./logs/results-{timestamp}.json"
        tester.export_results(standard_filename)
        
        print(f"\n{'='*70}")
        print(f"Mind2Web evaluation complete!")
        print(f"Results: {filename}")
        print(f"Full logs: {standard_filename}")
        print(f"{'='*70}\n")
        
    else:
        # Run standard evaluation
        print("="*70)
        print("STANDARD TESTING MODE")
        print("="*70)

        tester.run_comprehensive_test()
        tester.print_summary()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./logs/results-{timestamp}.json"
        tester.export_results(filename)
        
        print(f"\n{'='*70}")
        print(f"Testing complete! Check {filename} for detailed results.")
        print(f"{'='*70}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())