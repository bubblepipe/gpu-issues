#!/usr/bin/env python3
"""Test script for Ollama integration with thinking tag handling."""

import json
import sys
from cate import ask_local_ollama, load_issues_from_categorized_file

# Load a single issue for testing
frameworks = ['pytorch', 'tensorflow', 'jax', 'tensorrt', 'triton']
issue_groups = []

for framework in frameworks:
    try:
        with open(f'./issues/{framework}_issues.json', 'r') as f:
            issues = json.load(f)
            if issues:
                issue_groups.append(issues)
                break  # Just get one framework's issues for testing
    except FileNotFoundError:
        continue

if not issue_groups or not issue_groups[0]:
    print("No issues found to test with")
    sys.exit(1)

# Take the first issue for testing
test_issue = issue_groups[0][0]

print(f"Testing Ollama integration with issue: {test_issue['title']}")
print(f"URL: {test_issue['html_url']}")
print("-" * 50)

# Test the Ollama function
result = ask_local_ollama(test_issue)

if result.is_err():
    print(f"Error: {result.unwrap_err()}")
    sys.exit(1)
else:
    categorization = result.unwrap()
    print("\nSuccessfully categorized!")
    print("Results:")
    for i, category in enumerate(categorization):
        print(f"  {i+1}. {category.value}")
    print("\nOllama integration is working correctly!")