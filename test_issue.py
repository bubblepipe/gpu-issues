#!/usr/bin/env python3
"""Test script for Issue class refactoring"""

import json
from issue import Issue
from cate import load_framework_issues, prepare_full_prompt

def test_issue_creation():
    """Test creating an Issue from JSON data"""
    print("Testing Issue creation from JSON...")
    
    # Sample issue data from JAX
    sample_issue = {
        "number": 17448,
        "title": "BCOO behavior different from dense",
        "html_url": "https://github.com/jax-ml/jax/issues/17448",
        "body": "Test body content",
        "labels": [{"name": "bug"}, {"name": "sparse"}],
        "state": "closed",
        "created_at": "2023-08-30T12:00:00Z",
        "closed_at": "2023-11-29T12:00:00Z",
        "user": {"login": "test_user"},
        "timeline_url": "https://api.github.com/repos/jax-ml/jax/issues/17448/timeline"
    }
    
    issue = Issue.from_json(sample_issue, fetch_timeline=False)
    print(f"✓ Created issue: #{issue.number} - {issue.title}")
    print(f"  State: {issue.state}")
    print(f"  Labels: {[l['name'] for l in issue.labels]}")
    return issue

def test_pretty_print(issue):
    """Test the to_string_pretty method"""
    print("\nTesting pretty print...")
    output = issue.to_string_pretty()
    print("✓ Pretty print output (first 200 chars):")
    print("  " + output[:200].replace("\n", "\n  "))

def test_load_framework_issues():
    """Test loading issues from JSON files"""
    print("\nTesting load_framework_issues...")
    try:
        issue_groups = load_framework_issues(fetch_timelines=False)
        print(f"✓ Loaded {len(issue_groups)} framework groups")
        
        for i, issues in enumerate(issue_groups):
            if issues:
                print(f"  Framework {i+1}: {len(issues)} issues")
                # Test first issue
                first_issue = issues[0]
                print(f"    First issue: #{first_issue.number} - {first_issue.title[:50]}...")
                break
        
        return issue_groups
    except Exception as e:
        print(f"✗ Error loading issues: {e}")
        return []

def test_prepare_prompt():
    """Test prepare_full_prompt with Issue object"""
    print("\nTesting prepare_full_prompt...")
    
    sample_issue = Issue(
        number=123,
        title="Test Issue",
        html_url="https://github.com/test/repo/issues/123",
        body="This is a test issue body",
        labels=[{"name": "bug"}],
        state="open",
        created_at="2023-01-01",
        closed_at=None,
        user={"login": "testuser"}
    )
    
    prompt = prepare_full_prompt(sample_issue)
    print(f"✓ Generated prompt (length: {len(prompt)} chars)")
    print(f"  Contains 'Test Issue': {('Test Issue' in prompt)}")
    print(f"  Contains bug label: {('bug' in prompt)}")

def main():
    print("=" * 60)
    print("Issue Class Refactoring Test Suite")
    print("=" * 60)
    
    # Test 1: Create Issue from JSON
    issue = test_issue_creation()
    
    # Test 2: Pretty print
    test_pretty_print(issue)
    
    # Test 3: Load framework issues
    issue_groups = test_load_framework_issues()
    
    # Test 4: Prepare prompt
    test_prepare_prompt()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()