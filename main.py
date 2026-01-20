"""
Main entry point for the Agentic Stack Testing Framework

Usage:
    python main.py
"""

import os
import sys
import datetime
from tester import Mastif

def main():
    """Main execution function with Mind2Web support"""
    
    # Check for config file argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "experiments/example.yaml" # Default config file

    # Configuration
    # TODO: Add this to config file as well
    # Instantiate ConfigExpert singleton here later
    MODE = os.getenv("TEST_MODE", "standard")  # "standard" or "mind2web"
    MIND2WEB_NUM_TASKS = int(os.getenv("MIND2WEB_NUM_TASKS", "10"))  # 0 to all tasks
    
    
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
        if(tester.run_mind2web_evaluation(
            config_path=config_path,
            num_tasks=MIND2WEB_NUM_TASKS if MIND2WEB_NUM_TASKS > 0 else None
        )):
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

        if(tester.run_comprehensive_test(config_path)):
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