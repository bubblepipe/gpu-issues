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
import os


# Framework configurations with their GitHub repos and filters
FRAMEWORK_CONFIG = {
    "pytorch": {
        "repo": "pytorch/pytorch",
        "filter": "is:issue is:closed label:triaged",
        "description": "PyTorch deep learning framework"
    },
    "tensorrt": {
        "repo": "NVIDIA/TensorRT",
        "filter": "is:issue is:closed label:triaged",
        "description": "NVIDIA TensorRT high-performance deep learning inference"
    },
    "tensorflow": {
        "repo": "tensorflow/tensorflow",
        "filter": "is:issue is:closed label:type:bug",
        "description": "TensorFlow machine learning framework"
    },
    "jax": {
        "repo": "jax-ml/jax",
        "filter": "is:issue is:closed label:bug",
        "description": "JAX composable transformations of Python+NumPy programs"
    },
    "triton": {
        "repo": "triton-lang/triton",
        "filter": "is:issue is:closed label:bug",
        "description": "Triton language and compiler for GPU programming"
    }
}


def fetch_bugs_for_date_range(
    framework: str,
    start_date: str,
    end_date: str,
    per_page: int = 100,
    max_pages: Optional[int] = None,
    custom_filter: Optional[str] = None
) -> List[Dict]:
    """
    Fetch closed bugs from a specific ML framework repository for a date range.
    
    Args:
        framework: Name of the ML framework
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        per_page: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch (None for all)
        custom_filter: Optional custom filter to override the default filter
    
    Returns:
        List of bug issues
    """
    if framework not in FRAMEWORK_CONFIG:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(FRAMEWORK_CONFIG.keys())}")
    
    config = FRAMEWORK_CONFIG[framework]
    repo = config["repo"]
    
    # Build query with date range
    if custom_filter:
        query = f"repo:{repo} {custom_filter} created:{start_date}..{end_date}"
    else:
        query = f"repo:{repo} {config['filter']} created:{start_date}..{end_date}"
    
    all_bugs = []
    page = 1
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    
    # Add GitHub token if available (from environment variable)
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        headers["Authorization"] = f"token {github_token}"
        print(f"✓ Using GitHub token for authentication (token starts with: {github_token[:20]}...)")
        print("  Rate limits: 30 requests/minute, 5000 requests/hour")
    else:
        print("✗ No GitHub token found. Consider setting GITHUB_TOKEN environment variable.")
        print("  Rate limits: 10 requests/minute, 60 requests/hour")
    
    print(f"Fetching {framework.upper()} bugs for {start_date} to {end_date}...")
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
            
            # Debug: Show rate limit info on first request
            if page == 1:
                rate_limit = response.headers.get('X-RateLimit-Limit', 'N/A')
                rate_remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
                print(f"  Rate limit: {rate_remaining}/{rate_limit} requests remaining")
            
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
            elif e.response.status_code == 422:
                # GitHub returns 422 when trying to access beyond 1000 results
                print(f"\nReached GitHub's 1000 result limit for this date range.")
                break
            else:
                print(f"Error fetching page {page}: {e}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    print(f"Fetched {len(all_bugs)} bugs for this date range")
    
    return all_bugs


def fetch_framework_bugs(
    framework: str,
    since_date: str,
    per_page: int = 100,
    max_pages: Optional[int] = None,
    output_file: Optional[str] = None,
    custom_filter: Optional[str] = None,
    chunk_days: int = 30
) -> List[Dict]:
    """
    Fetch closed bugs from a specific ML framework repository, automatically
    splitting date ranges to handle GitHub's 1000 result limit.
    
    Args:
        framework: Name of the ML framework
        since_date: Date in YYYY-MM-DD format to fetch bugs from
        per_page: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch (None for all)
        output_file: Optional file to save results to
        custom_filter: Optional custom filter to override the default filter
        chunk_days: Number of days per chunk (default 30)
    
    Returns:
        List of bug issues
    """
    print(f"Fetching closed {framework.upper()} bugs created since {since_date}...")
    
    # Parse dates
    start_date = datetime.strptime(since_date, "%Y-%m-%d")
    end_date = datetime.now()
    
    all_bugs = []
    current_start = start_date
    
    # Process in chunks
    while current_start < end_date:
        # Calculate chunk end date
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        # Fetch bugs for this chunk
        chunk_bugs = fetch_bugs_for_date_range(
            framework=framework,
            start_date=current_start.strftime("%Y-%m-%d"),
            end_date=current_end.strftime("%Y-%m-%d"),
            per_page=per_page,
            max_pages=max_pages,
            custom_filter=custom_filter
        )
        
        all_bugs.extend(chunk_bugs)
        
        # If we got close to 1000 results, use smaller chunks
        if len(chunk_bugs) >= 900:
            chunk_days = max(7, chunk_days // 2)
            print(f"Reducing chunk size to {chunk_days} days due to high result count")
        
        # Move to next chunk
        current_start = current_end + timedelta(days=1)
    
    print(f"\nTotal bugs fetched across all date ranges: {len(all_bugs)}")
    
    # Remove duplicates (in case of date boundary issues)
    unique_bugs = []
    seen_ids = set()
    for bug in all_bugs:
        if bug['id'] not in seen_ids:
            unique_bugs.append(bug)
            seen_ids.add(bug['id'])
    
    if len(unique_bugs) < len(all_bugs):
        print(f"Removed {len(all_bugs) - len(unique_bugs)} duplicate bugs")
    
    # Filter out issues marked as duplicates (based on labels)
    filtered_bugs = []
    duplicate_count = 0
    for bug in unique_bugs:
        # Check if any label indicates this is a duplicate
        is_duplicate = any(
            'duplicate' in label['name'].lower() 
            for label in bug.get('labels', [])
        )
        
        if not is_duplicate:
            filtered_bugs.append(bug)
        else:
            duplicate_count += 1
    
    if duplicate_count > 0:
        print(f"Filtered out {duplicate_count} issues marked as duplicates")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(filtered_bugs, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return filtered_bugs


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
        print(f"  Default filter: {config['filter']}")


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
        choices=list(FRAMEWORK_CONFIG.keys()) + ['all'],
        help="ML framework to fetch bugs from (use 'all' for all frameworks)"
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
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Days per chunk when splitting date ranges (default: 30)"
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
    
    # Handle "all" framework option
    if args.framework == 'all':
        all_bugs = []
        for fw_name in FRAMEWORK_CONFIG.keys():
            print(f"\n{'='*60}")
            print(f"Fetching bugs for {fw_name.upper()}")
            print(f"{'='*60}")
            
            # Always save individual files for each framework
            fw_output = f"{fw_name}_issues.json"
            
            fw_bugs = fetch_framework_bugs(
                framework=fw_name,
                since_date=since_date,
                max_pages=args.max_pages,
                output_file=fw_output,
                custom_filter=args.custom_filter,
                chunk_days=args.chunk_days
            )
            
            # Add framework field to each bug
            for bug in fw_bugs:
                bug['framework'] = fw_name
            
            all_bugs.extend(fw_bugs)
        
        # Save combined results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_bugs, f, indent=2)
            print(f"\nCombined results saved to: {args.output}")
        
        # Analyze combined results
        if not args.no_analysis:
            print(f"\n{'='*60}")
            print("COMBINED ANALYSIS FOR ALL FRAMEWORKS")
            print(f"{'='*60}")
            analyze_bugs(all_bugs, "ALL FRAMEWORKS")
            
            # Per-framework breakdown
            print("\nPer-framework breakdown:")
            for fw_name in FRAMEWORK_CONFIG.keys():
                fw_bugs = [b for b in all_bugs if b.get('framework') == fw_name]
                print(f"  {fw_name}: {len(fw_bugs)} bugs")
    else:
        # Single framework
        bugs = fetch_framework_bugs(
            framework=args.framework,
            since_date=since_date,
            max_pages=args.max_pages,
            output_file=args.output,
            custom_filter=args.custom_filter,
            chunk_days=args.chunk_days
        )
        
        # Analyze the results
        if not args.no_analysis:
            analyze_bugs(bugs, args.framework)