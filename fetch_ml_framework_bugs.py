#!/usr/bin/env python3
"""
Fetch closed bugs from various ML framework GitHub repositories.
Supports: PyTorch, TensorRT, TensorFlow, JAX, and Triton.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import argparse


# Framework configurations with their GitHub repos and filters
FRAMEWORK_CONFIG = {
    "pytorch": {
        "repo": "pytorch/pytorch",
        "base_filter": "is:issue is:closed label:triaged",
        "description": "PyTorch deep learning framework"
    },
    "tensorrt": {
        "repo": "NVIDIA/TensorRT",
        "base_filter": "is:issue is:closed (label:bug OR label:Bug)",
        "description": "NVIDIA TensorRT high-performance deep learning inference"
    },
    "tensorflow": {
        "repo": "tensorflow/tensorflow",
        "base_filter": "is:issue is:closed (label:\"type:bug\" OR label:\"type:build/install\" OR label:\"type:performance\")",
        "description": "TensorFlow machine learning framework"
    },
    "jax": {
        "repo": "jax-ml/jax",
        "base_filter": "is:issue is:closed (label:bug OR label:Bug)",
        "description": "JAX composable transformations of Python+NumPy programs"
    },
    "triton": {
        "repo": "triton-lang/triton",
        "base_filter": "is:issue is:closed (label:bug OR label:Bug)",
        "description": "Triton language and compiler for GPU programming"
    }
}


def fetch_framework_bugs(
    framework: str,
    since_date: str,
    per_page: int = 100,
    max_pages: Optional[int] = None,
    output_file: Optional[str] = None,
    custom_filter: Optional[str] = None
) -> List[Dict]:
    """
    Fetch closed bugs from a specific ML framework repository.
    
    Args:
        framework: Name of the ML framework
        since_date: Date in YYYY-MM-DD format to fetch bugs from
        per_page: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch (None for all)
        output_file: Optional file to save results to
        custom_filter: Optional custom filter to override the default base_filter
    
    Returns:
        List of bug issues
    """
    if framework not in FRAMEWORK_CONFIG:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(FRAMEWORK_CONFIG.keys())}")
    
    config = FRAMEWORK_CONFIG[framework]
    repo = config["repo"]
    
    # Build query
    if custom_filter:
        query = f"repo:{repo} {custom_filter} created:>={since_date}"
    else:
        query = f"repo:{repo} {config['base_filter']} created:>={since_date}"
    
    all_bugs = []
    page = 1
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        # Add your GitHub token here for higher rate limits (optional)
        # "Authorization": "token YOUR_GITHUB_TOKEN"
    }
    
    print(f"Fetching closed {framework.upper()} bugs created since {since_date}...")
    print(f"Query: {query}")
    
    base_url = "https://api.github.com/search/issues"
    
    while True:
        params = {
            "q": query,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "order": "desc"
        }
        
        response = None
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if "items" not in data:
                print(f"Unexpected response format: {data}")
                break
            
            bugs = data["items"]
            all_bugs.extend(bugs)
            
            # Print progress
            print(f"Page {page}: fetched {len(bugs)} bugs (total: {len(all_bugs)})")
            
            # Check if we've fetched all results
            if len(bugs) < per_page:
                print("No more results.")
                break
            
            # Check if we've reached the maximum number of pages
            if max_pages and page >= max_pages:
                print(f"Reached maximum page limit ({max_pages}).")
                break
            
            # GitHub search API has a limit of 1000 results
            if len(all_bugs) >= 1000:
                print("\nReached GitHub's 1000 result limit for search API.")
                print("To get more results, use a more specific filter or date range.")
                break
            
            # GitHub API rate limiting - be nice
            time.sleep(1)
            
            page += 1
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                # Rate limit hit - check headers for reset time
                reset_time = e.response.headers.get('X-RateLimit-Reset')
                remaining = e.response.headers.get('X-RateLimit-Remaining', '0')
                
                if reset_time:
                    reset_datetime = datetime.fromtimestamp(int(reset_time))
                    wait_time = (reset_datetime - datetime.now()).total_seconds()
                    wait_time = max(wait_time, 60)  # Wait at least 60 seconds
                else:
                    wait_time = 300  # Default 5 minutes if no reset time
                
                print(f"\nRate limit exceeded (remaining: {remaining})")
                print(f"Waiting {int(wait_time)} seconds until reset...")
                print("Consider adding a GitHub token for higher rate limits")
                
                # Show progress while waiting
                for i in range(int(wait_time), 0, -10):
                    print(f"  Resuming in {i} seconds...", end='\r')
                    time.sleep(min(i, 10))
                
                print("\nResuming...")
                continue  # Retry the same page
            else:
                print(f"Error fetching page {page}: {e}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    print(f"\nTotal bugs fetched: {len(all_bugs)}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_bugs, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return all_bugs


def analyze_bugs(bugs: List[Dict], framework: str) -> None:
    """Analyze and print summary statistics about the bugs."""
    if not bugs:
        print("No bugs to analyze.")
        return
    
    print(f"\n=== {framework.upper()} Bug Analysis ===")
    print(f"Total closed bugs: {len(bugs)}")
    
    # Count by close reason
    close_reasons = {}
    labels_count = {}
    
    for bug in bugs:
        # Count labels
        for label in bug.get("labels", []):
            label_name = label["name"]
            labels_count[label_name] = labels_count.get(label_name, 0) + 1
        
        # Analyze state reason if available
        state_reason = bug.get("state_reason", "unknown")
        close_reasons[state_reason] = close_reasons.get(state_reason, 0) + 1
    
    print("\nClose reasons:")
    for reason, count in sorted(close_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    print("\nTop 15 labels:")
    for label, count in sorted(labels_count.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {label}: {count}")
    
    # Show some example bugs
    print("\nExample bugs (first 5):")
    for i, bug in enumerate(bugs[:5]):
        print(f"\n{i+1}. #{bug['number']}: {bug['title']}")
        print(f"   Created: {bug['created_at']}")
        print(f"   Closed: {bug['closed_at']}")
        print(f"   Labels: {', '.join([l['name'] for l in bug['labels']])}")
        print(f"   URL: {bug['html_url']}")


def list_frameworks():
    """List all supported frameworks and their configurations."""
    print("Supported ML Frameworks:")
    print("=" * 60)
    for name, config in FRAMEWORK_CONFIG.items():
        print(f"\n{name.upper()}")
        print(f"  Description: {config['description']}")
        print(f"  Repository: {config['repo']}")
        print(f"  Default filter: {config['base_filter']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch closed bugs from ML framework GitHub repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch PyTorch bugs from last 1400 days
  python fetch_ml_framework_bugs.py --framework pytorch --days 1400
  
  # Fetch TensorFlow bugs with custom filter
  python fetch_ml_framework_bugs.py --framework tensorflow --custom-filter "is:issue is:closed label:comp:gpu"
  
  # List all supported frameworks
  python fetch_ml_framework_bugs.py --list-frameworks
        """
    )
    
    parser.add_argument(
        "--framework",
        type=str,
        choices=list(FRAMEWORK_CONFIG.keys()),
        help="ML framework to fetch bugs from"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1825,  # 5 years = 5 * 365 days
        help="Number of days to look back (default: 1825 = 5 years)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to fetch (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (default: {framework}_issues.json)"
    )
    parser.add_argument(
        "--custom-filter",
        type=str,
        help="Custom filter to override the default (e.g., 'is:issue is:closed label:bug')"
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip the analysis output"
    )
    parser.add_argument(
        "--list-frameworks",
        action="store_true",
        help="List all supported frameworks and their configurations"
    )
    
    args = parser.parse_args()
    
    # Handle list frameworks
    if args.list_frameworks:
        list_frameworks()
        exit(0)
    
    # Check if framework is provided
    if not args.framework:
        parser.error("--framework is required unless using --list-frameworks")
    
    # Set default output file if not provided
    if not args.output:
        args.output = f"{args.framework}_issues.json"
    
    # Calculate the date to search from
    since_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    # Fetch the bugs
    bugs = fetch_framework_bugs(
        framework=args.framework,
        since_date=since_date,
        max_pages=args.max_pages,
        output_file=args.output,
        custom_filter=args.custom_filter
    )
    
    # Analyze the results
    if not args.no_analysis:
        analyze_bugs(bugs, args.framework)