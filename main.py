"""
Main entry point for the HuggingFace Agentic Stack Testing Framework

Usage:
    python main.py
"""

import os
import datetime
from tester import AgenticStackTester


def main():
    """Main execution function with Mind2Web support"""
    
    # Configuration
    MODE = os.getenv("TEST_MODE", "standard")  # "standard" or "mind2web"
    MIND2WEB_NUM_TASKS = int(os.getenv("MIND2WEB_NUM_TASKS", "10"))  # 0 to all tasks
    
    models_to_test = [ 
        # Workwed well
        # "meta-llama/Llama-3.3-70B-Instruct", 
        # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        "gpt-4o"
        # "deepseek-ai/DeepSeek-V3.2"

        # Worked, but repeated info about the protocols
        # "meta-llama/Llama-3.1-8B-Instruct", 
        
        # Worked, but response was empty in multiple requests
        # "openai/gpt-oss-20b", 
    ]
    
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
        print(f"\nConfiguration:")
        print(f"  Models: {len(models_to_test)}")
        print(f"  Tasks: {MIND2WEB_NUM_TASKS if MIND2WEB_NUM_TASKS > 0 else 'ALL'}")
        print(f"  Frameworks: All 6 frameworks")
        
        # Run Mind2Web evaluation
        tester.run_mind2web_evaluation(
            models=models_to_test,
            hf_token=hf_token,
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
        print("\nTesting Configuration:")
        print(f"  Models: {len(models_to_test)}")

        # Dynamically get protocols and frameworks from tester
        protocol_names = [p.value if hasattr(p, "value") else str(p) for p in tester.get_supported_protocols()]
        framework_names = tester.get_supported_frameworks()

        print(f"  Protocols: {', '.join(protocol_names)}")
        print(f"  Frameworks: {', '.join(framework_names)}")
        print(f"\n  Total tests per model: {len(protocol_names) + len(framework_names)} "
              f"({len(protocol_names)} protocols + {len(framework_names)} frameworks)")
        print(f"  Total tests: {len(models_to_test) * (len(protocol_names) + len(framework_names))}")
        print("\nStarting tests...\n")
        
        # tester.run_comprehensive_test(models_to_test, hf_token)
        tester.run_comprehensive_test(models_to_test, open_ai_key)
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