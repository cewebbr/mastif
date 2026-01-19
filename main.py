"""
Main entry point for the HuggingFace Agentic Stack Testing Framework

Usage:
    python main.py
"""

import os
import sys
import datetime
from tester import AgenticStackTester

# TODO: Compute the number of tests dynamically based on selected models, protocols, and frameworks and provide a warning before people start the tests. It is necessary inform that all combinations will be tested and this may incurr a high number of API calls and associated costs.

def main():
    """Main execution function with Mind2Web support"""
    
    # Check for config file argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # TODO: Change to an example test config with a simple experiment
        config_path = "experiments/w4a2026.yaml" # Default config file

    # Configuration
    # TODO: Add this to config file as well
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
    tester = AgenticStackTester()
    
    if MODE == "mind2web":
        # Run Mind2Web evaluation
        print("="*70)
        print("MIND2WEB BENCHMARK MODE")
        print("="*70)
        # TODO: Print info from config file
        
        # Run Mind2Web evaluation
        tester.run_mind2web_evaluation(
            config_path=config_path,
            num_tasks=MIND2WEB_NUM_TASKS if MIND2WEB_NUM_TASKS > 0 else None
        )
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
        # TODO: Print info from config file
        # print("\nTesting Configuration:")
        # print(f"  Models: {len(models_to_test)}")

        # # Dynamically get protocols and frameworks from tester
        # protocol_names = [p.value if hasattr(p, "value") else str(p) for p in tester.get_supported_protocols()]
        # framework_names = tester.get_supported_frameworks()

        # print(f"  Protocols: {', '.join(protocol_names)}")
        # print(f"  Frameworks: {', '.join(framework_names)}")
        # print(f"\n  Total tests per model: {len(protocol_names) + len(framework_names)} "
        #       f"({len(protocol_names)} protocols + {len(framework_names)} frameworks)")
        # print(f"  Total tests: {len(models_to_test) * (len(protocol_names) + len(framework_names))}")
        # print("\nStarting tests...\n")
        
        tester.run_comprehensive_test(config_path)
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