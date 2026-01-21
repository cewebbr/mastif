"""
Mind2Web dataset loader and task manager

This module handles loading the Mind2Web dataset from HuggingFace
and managing task sampling for testing.
"""

from typing import List, Dict, Optional
import random
import os
from config import ConfigExpert

class Mind2WebLoader:
    """
    Loader for Mind2Web benchmark dataset
    
    Handles dataset loading, filtering, and sampling for evaluation.
    Note: Uses alternative loading method due to dataset formatting issues.
    """
    
    def __init__(self, split: str = "test", hf_token: str = None):
        """
        Initialize Mind2Web loader
        
        Args:
            split: Dataset split to load ('test', 'train', 'valid')
            hf_token: HuggingFace API token for authentication
        """
        self.split = split
        self.hf_token = hf_token
        self.dataset = None
        self.tasks = []
    
    def load_dataset(self):
        """
        Load Mind2Web dataset using alternative method
        
        The official dataset has JSON parsing issues, so we use
        the raw data files directly or a pre-processed version.
        
        Returns:
            Boolean indicating success
        """
        try:
            print(f"Loading Mind2Web dataset (split: {self.split})...")
            
            # Method 1: Try using streaming mode to avoid corrupted files
            try:
                from datasets import load_dataset
                
                token = self.hf_token or os.getenv("HF_TOKEN")
                
                # Use streaming to skip problematic files
                print("Attempting to load with streaming mode...")
                dataset_stream = load_dataset(
                    "osunlp/Mind2Web",
                    split=self.split,
                    streaming=True,
                    token=token
                )
                
                # Convert stream to list (load first N examples that work)
                self.dataset = []
                config = ConfigExpert.get_instance()
                max_examples = int(config.get("mind2web_num_tasks", 2350))
                
                print("Loading examples from stream...")
                for i, example in enumerate(dataset_stream):
                    if i >= max_examples:
                        break
                    self.dataset.append(example)
                    if (i + 1) % 100 == 0:
                        print(f"  Loaded {i + 1} examples...")
                
                if len(self.dataset) > 0:
                    print(f"✓ Loaded {len(self.dataset)} tasks from Mind2Web (streaming mode)")
                    return True
                else:
                    print("✗ No examples loaded from streaming mode")
                    return self._load_sample_data()
                    
            except Exception as stream_error:
                print(f"Streaming mode failed: {stream_error}")
                print("Falling back to sample data...")
                return self._load_sample_data()
                
        except Exception as e:
            print(f"✗ Error loading Mind2Web dataset: {e}")
            print("Using sample data for demonstration...")
            return self._load_sample_data()
    
    def _load_sample_data(self):
        """
        Load sample Mind2Web-style data for testing
        
        This provides a fallback when the actual dataset cannot be loaded.
        
        Returns:
            Boolean indicating success
        """
        print("\n" + "="*70)
        print("USING SAMPLE MIND2WEB DATA")
        print("="*70)
        print("Note: Full dataset could not be loaded due to formatting issues.")
        print("Using representative sample tasks for demonstration.\n")
        
        # Create sample Mind2Web-style tasks
        self.dataset = [
            {
                "annotation_id": "sample_001",
                "website": "amazon.com",
                "domain": "shopping",
                "confirmed_task": "Find the cheapest laptop under $800 and add it to cart",
                "actions": ["CLICK", "TYPE", "CLICK", "SELECT", "CLICK"],
                "action_reprs": ["click search box", "type laptop under 800", "click search", "select sort by price", "click add to cart"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_002",
                "website": "reddit.com",
                "domain": "social_media",
                "confirmed_task": "Find the top post in r/programming and upvote it",
                "actions": ["CLICK", "CLICK", "CLICK"],
                "action_reprs": ["click search", "type r/programming", "click upvote on top post"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_003",
                "website": "wikipedia.org",
                "domain": "general",
                "confirmed_task": "Search for 'Artificial Intelligence' and read the introduction",
                "actions": ["CLICK", "TYPE", "CLICK", "SCROLL"],
                "action_reprs": ["click search box", "type artificial intelligence", "click search button", "scroll to introduction"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_004",
                "website": "booking.com",
                "domain": "travel",
                "confirmed_task": "Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20",
                "actions": ["CLICK", "TYPE", "SELECT", "SELECT", "CLICK", "CLICK"],
                "action_reprs": ["click destination", "type Paris", "select checkin date", "select checkout date", "select 2 adults", "click search"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_005",
                "website": "linkedin.com",
                "domain": "social_media",
                "confirmed_task": "Search for Software Engineer jobs in San Francisco and filter by remote",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click jobs search", "type software engineer san francisco", "click search", "click remote filter"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_006",
                "website": "youtube.com",
                "domain": "entertainment",
                "confirmed_task": "Search for 'Python tutorial' videos and play the most viewed one",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click search box", "type python tutorial", "press enter", "click first video"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_007",
                "website": "github.com",
                "domain": "service",
                "confirmed_task": "Search for 'machine learning' repositories and star the top result",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click search", "type machine learning", "click repositories filter", "click star on top result"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_008",
                "website": "ebay.com",
                "domain": "shopping",
                "confirmed_task": "Find used iPhone 13 under $500 and sort by ending soonest",
                "actions": ["CLICK", "TYPE", "CLICK", "SELECT", "SELECT"],
                "action_reprs": ["click search", "type iphone 13", "click search", "select used condition", "select sort by ending soonest"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_009",
                "website": "stackoverflow.com",
                "domain": "service",
                "confirmed_task": "Search for questions about 'async await in Python' and read the top answer",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click search", "type async await python", "press enter", "click top question"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_010",
                "website": "twitter.com",
                "domain": "social_media",
                "confirmed_task": "Search for tweets about 'AI news' and retweet the most recent one",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click search", "type AI news", "click latest filter", "click retweet on first tweet"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_011",
                "website": "netflix.com",
                "domain": "entertainment",
                "confirmed_task": "Search for sci-fi movies and add the first result to my list",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click search", "type sci-fi", "select movies", "click add to list"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_012",
                "website": "airbnb.com",
                "domain": "travel",
                "confirmed_task": "Find entire apartments in Tokyo for 2 guests from Jan 10-15",
                "actions": ["CLICK", "TYPE", "SELECT", "SELECT", "CLICK", "CLICK"],
                "action_reprs": ["click location", "type tokyo", "select dates", "select 2 guests", "click entire place filter", "click search"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_013",
                "website": "imdb.com",
                "domain": "entertainment",
                "confirmed_task": "Find the highest rated action movies from 2023",
                "actions": ["CLICK", "SELECT", "SELECT", "CLICK"],
                "action_reprs": ["click advanced search", "select action genre", "select year 2023", "sort by rating"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_014",
                "website": "indeed.com",
                "domain": "service",
                "confirmed_task": "Search for data scientist positions and filter for remote work",
                "actions": ["CLICK", "TYPE", "CLICK", "CLICK"],
                "action_reprs": ["click job search", "type data scientist", "click search", "click remote filter"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            },
            {
                "annotation_id": "sample_015",
                "website": "walmart.com",
                "domain": "shopping",
                "confirmed_task": "Search for wireless headphones under $100 and filter by customer rating",
                "actions": ["CLICK", "TYPE", "CLICK", "SELECT", "SELECT"],
                "action_reprs": ["click search", "type wireless headphones", "click search", "select price under 100", "select 4 stars and up"],
                "pos_candidates": [],
                "cleaned_html": "<html>...</html>"
            }
        ]
        
        print(f"✓ Loaded {len(self.dataset)} sample tasks")
        print("These tasks represent typical Mind2Web scenarios.\n")
        
        return True
    
    def get_task_sample(self, num_tasks: Optional[int] = None, seed: int = 42) -> List[Dict]:
        """
        Get a sample of tasks from the dataset
        
        Args:
            num_tasks: Number of tasks to sample (None for all tasks)
            seed: Random seed for reproducibility
            
        Returns:
            List of task dictionaries
        """
        if self.dataset is None:
            if not self.load_dataset():
                return []
        
        # Get all tasks
        all_tasks = list(self.dataset)
        
        # Sample if requested
        if num_tasks is not None and num_tasks < len(all_tasks):
            random.seed(seed)
            sampled_tasks = random.sample(all_tasks, num_tasks)
            print(f"Sampled {num_tasks} tasks from {len(all_tasks)} total tasks")
        else:
            sampled_tasks = all_tasks
            print(f"Using all {len(all_tasks)} tasks")
        
        # Convert to simplified format
        self.tasks = []
        for task in sampled_tasks:
            self.tasks.append({
                "task_id": task.get("annotation_id", "unknown"),
                "website": task.get("website", "unknown"),
                "domain": task.get("domain", "unknown"),
                "confirmed_task": task.get("confirmed_task", ""),
                # "actions": task.get("actions", []), # We don't need the raw actions for now
                "action_reprs": task.get("action_reprs", []),
                "pos_candidates": task.get("pos_candidates", []),
                "raw_html": task.get("cleaned_html", ""),
            })
        
        return self.tasks
    
    def get_task_by_domain(self, domain: str) -> List[Dict]:
        """
        Filter tasks by domain
        
        Args:
            domain: Domain name to filter by
            
        Returns:
            List of tasks in the specified domain
        """
        if not self.tasks:
            self.get_task_sample()
        
        return [task for task in self.tasks if task["domain"] == domain]
    
    def get_available_domains(self) -> List[str]:
        """
        Get list of available domains in the dataset
        
        Returns:
            List of unique domain names
        """
        if not self.tasks:
            self.get_task_sample()
        
        return list(set(task["domain"] for task in self.tasks))
    
    def get_task_statistics(self) -> Dict:
        """
        Get statistics about loaded tasks
        
        Returns:
            Dictionary with task statistics
        """
        if not self.tasks:
            return {}
        
        domains = {}
        websites = {}
        total_actions = 0
        
        for task in self.tasks:
            domain = task["domain"]
            website = task["website"]
            actions = len(task.get("actions", []))
            
            domains[domain] = domains.get(domain, 0) + 1
            websites[website] = websites.get(website, 0) + 1
            total_actions += actions
        
        return {
            "total_tasks": len(self.tasks),
            "unique_domains": len(domains),
            "unique_websites": len(websites),
            "avg_actions_per_task": total_actions / len(self.tasks) if self.tasks else 0,
            "domains": domains,
            "websites": websites
        }