import json
import random

def print_issue(issue):
    """Pretty print a single issue in markdown format."""
    print(f"### [{issue['title']}]({issue['html_url']})")
    print()
    
    # Print metadata
    print(f"**Created:** {issue.get('created_at', 'Unknown')}")
    print()
    
    # Extract tag names from labels
    tags = [label['name'] for label in issue.get('labels', [])]
    if tags:
        print(f"**Tags:** `{' '.join(tags)}`")
    else:
        print("**Tags:** _(none)_")
    print()
    
    # Print issue body/content
    if 'body' in issue and issue['body']:
        print("**Content:**")
        print()
        # Print full content without truncation
        body = issue['body']
        print("> " + body.replace('\n', '\n> '))
    else:
        print("**Content:** _(empty)_")
    
    print()
    print("---")
    print()  # Empty line for readability

def print_issues(issues):
    """Pretty print a list of issues in markdown format."""
    for issue in issues:
        print_issue(issue)

# Count issues in each file
frameworks = ['pytorch', 'tensorflow', 'jax', 'tensorrt', 'triton']

issue_groups = [];

for framework in frameworks:
    try:
        with open(f'{framework}_issues.json', 'r') as f:
            # issues += json.load(f)
            issue_groups.append(json.load(f))
    except FileNotFoundError:
        print(f"{framework}: file not found")

count = 0;
for issues in issue_groups:
    count += len(issues)

print(f"Total issues found: {count}")

for issues in issue_groups:
    selected_issues = random.sample(issues, min(20, len(issues)))
    # Print title and URL
    for issue in selected_issues:
        print(f"{issue['title']} \n{issue['html_url']}\n ")
    print("\n\n=========================\n\n")
    # break  # Only show first framework