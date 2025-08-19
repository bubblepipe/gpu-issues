#!/usr/bin/env python3
"""Test timeline fetching for a specific issue"""

from issue import Issue

def test_timeline_fetch():
    """Test fetching timeline data for JAX issue #17448"""
    
    # Create issue with known timeline URL
    issue_data = {
        "number": 17448,
        "title": "BCOO behavior different from dense",
        "html_url": "https://github.com/jax-ml/jax/issues/17448",
        "body": "Test body",
        "labels": [{"name": "bug"}],
        "state": "closed",
        "created_at": "2023-08-30",
        "closed_at": "2023-11-29",
        "user": {"login": "test"},
        "timeline_url": "https://api.github.com/repos/jax-ml/jax/issues/17448/timeline"
    }
    
    print("Creating issue and fetching timeline...")
    issue = Issue.from_json(issue_data, fetch_timeline=True)
    
    print(f"\nIssue #{issue.number}: {issue.title}")
    print(f"Assignees from timeline: {issue.assignees_from_timeline}")
    print(f"Mentioned PRs: {len(issue.mentioned_prs)}")
    for pr in issue.mentioned_prs:
        print(f"  - PR #{pr.number}: {pr.title}")
    print(f"Mentioned Issues: {len(issue.mentioned_issues)}")
    for related_issue in issue.mentioned_issues:
        print(f"  - Issue #{related_issue.get('number')}: {related_issue.get('title')}")
    
    # Test pretty print with timeline data
    print("\n--- Pretty Print Output ---")
    print(issue.to_string_pretty()[:500])

if __name__ == "__main__":
    test_timeline_fetch()