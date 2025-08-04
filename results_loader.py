"""Functions for loading categorized results from files"""

import glob
import re
import sys
from cates import BugType, BugSymptom, BugHeterogeneity


def load_categorized_results(pattern):
    """
    Load categorized issues from result tuple files.
    
    Args:
        pattern: Glob pattern for finding result files
        
    Returns:
        List of tuples containing (title, url, bug_type, bug_symptom, bug_heterogeneity)
    """
    categorized_issues = []
    result_files = glob.glob(pattern)
    
    for file in result_files:
        with open(file, 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    # Replace enum representations with actual enum objects
                    content = re.sub(r"<(Bug\w+\.\w+): '[^']*'>", r"\1", content)
                    
                    # Now eval the cleaned content
                    # This is safe because we control the file format
                    tuples = eval(content)
                    categorized_issues.extend(tuples)
                except Exception as e:
                    print(f"Error parsing tuples from file {file}: {e}")
                    
    return categorized_issues


def get_categorized_urls(categorized_issues):
    """
    Extract URLs from categorized issues for fast lookup.
    
    Args:
        categorized_issues: List of categorized issue tuples
        
    Returns:
        Set of URLs that have been categorized
    """
    return {issue[1] for issue in categorized_issues}