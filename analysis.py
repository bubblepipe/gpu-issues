import json
import random
import requests
import os
import sys
from prompts import BUG_CATEGORIZATION_PROMPT
from cates import BUG_TYPE_LOOKUP, BUG_SYMPTOM_LOOKUP, BUG_HETEROGENEITY_LOOKUP
from results_loader import load_categorized_results, get_categorized_urls

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
        with open(f'./issues/{framework}_issues.json', 'r') as f:
            # issues += json.load(f)
            issue_groups.append(json.load(f))
    except FileNotFoundError:
        print(f"{framework}: file not found")

count = 0;
for issues in issue_groups:
    count += len(issues)



# Load categorized issues from result tuple files
categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/llm_categorizations_a81e1dd0/results.tuples.*')

def parse_bug_type(code):
    return BUG_TYPE_LOOKUP.get(code)

def parse_bug_symptom(code):
    return BUG_SYMPTOM_LOOKUP.get(code)

def parse_bug_heterogeneity(code):
    return BUG_HETEROGENEITY_LOOKUP.get(code)


def parse_llm_output(text):
    xs = [x.strip() for x in text.split(',')]
    if (len(xs) != 3 or
        xs[0] not in BUG_TYPE_LOOKUP or
        xs[1] not in BUG_SYMPTOM_LOOKUP or
        xs[2] not in BUG_HETEROGENEITY_LOOKUP):
        print(f"Invalid output format: {text}")
        sys.stderr.write(f"Invalid output format: {text}\n")
        return None
    return (parse_bug_type(xs[0]), parse_bug_symptom(xs[1]), parse_bug_heterogeneity(xs[2]))

def ask_gemini_2_5_flash(issue):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }
        data = {
            "contents": [
                {
                    "parts": [ { "text": f"{BUG_CATEGORIZATION_PROMPT}{issue['html_url']}" } ]
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an error for bad status codes
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return parse_llm_output(text)
        except requests.exceptions.RequestException as e:
            sys.stderr.write(f"Network error calling Gemini API: {e}\n")
            return None
        except (IndexError, KeyError, ValueError) as e:
            sys.stderr.write(f"Error parsing response from Gemini API: {e}\n")
            return None
        except Exception as e:
            sys.stderr.write(f"Unexpected error with Gemini API: {e}\n")
            return None
    else:
        return None


def ask_sonnet_4(issue):
    return None

def ask_opus_4(issue):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-opus-20240229",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"{BUG_CATEGORIZATION_PROMPT}{issue['html_url']}"
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        text = response.json()["content"][0]["text"]
        return parse_llm_output(text)
    else:
        return None


issues_categorized = []

# Create a set of URLs that are already categorized for fast lookup
categorized_urls = get_categorized_urls(categorized_issues)

for issues in issue_groups:
    # Find issues that haven't been categorized yet
    uncategorized_issues = [issue for issue in issues if issue['html_url'] not in categorized_urls]
    
    # Select up to 10 unique issues that haven't been categorized
    num_to_select = min(25, len(uncategorized_issues))
    if num_to_select > 0:
        selected_issues = random.sample(uncategorized_issues, num_to_select)
        
        # Print title and URL
        for issue in selected_issues:
            title = issue['title']
            url = issue['html_url']
            print(f"{title} \n{url}")
            result = ask_gemini_2_5_flash(issue)
            if result is None :
                sys.stderr.write(f"Failed to categorize: {title} - {url}\n")
                print()
                continue
            issues_categorized.append( (title, url, result[0], result[1], result[2] ) )
            for line in list(result):
                print(" - " + line.value)
            print()
            # exit()
    else:
        print(f"All issues in this group have already been categorized")
    
    print("\n\n=========================\n\n")
    
print()
print(issues_categorized)