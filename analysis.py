import json
import random
import requests
import os
from enum import Enum

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

print(f"Total issues found: {count}")
print()


class BugType(Enum):
    NOT_A_BUG = "1.a not a bug"
    SOURCE_CODE_ISSUE = "1.b source code issue"
    LOW_LEVEL_SOFTWARE_STACK = "1.c low-level software stack, like GPU driver, CUDA toolchain, or even hardware bugs"
    WRONG_API_USAGE = "1.d user called the wrong api"
    OTHER = "1.e other"

class BugSymptom(Enum):
    NOT_ABLE_TO_COMPILE = "2.a not able to compile"
    CRASHES_DURING_RUNTIME = "2.b crashes during runtime"
    PRODUCES_WRONG_RESULT = "2.c produces wrong result"
    UNEXPECTED_RUNTIME_DURATION = "2.d unexpected runtime duration"
    UNEXPECTED_MEMORY_USAGE = "2.e unexpected amount of consumed memory"
    OTHER = "2.f other"
    NOT_A_BUG = "2.g not a bug"

class BugUniversality(Enum):
    UNIVERSAL = "3.a no, universal"
    BACKEND_SPECIFIC = "3.b yes, some backend specifically"
    NOT_APPLICABLE = "3.c not applicable"

BUG_TYPE_LOOKUP = {
    "1.a": BugType.NOT_A_BUG,
    "1.b": BugType.SOURCE_CODE_ISSUE,
    "1.c": BugType.LOW_LEVEL_SOFTWARE_STACK,
    "1.d": BugType.WRONG_API_USAGE,
    "1.e": BugType.OTHER,
}

BUG_SYMPTOM_LOOKUP = {
    "2.a": BugSymptom.NOT_ABLE_TO_COMPILE,
    "2.b": BugSymptom.CRASHES_DURING_RUNTIME,
    "2.c": BugSymptom.PRODUCES_WRONG_RESULT,
    "2.d": BugSymptom.UNEXPECTED_RUNTIME_DURATION,
    "2.e": BugSymptom.UNEXPECTED_MEMORY_USAGE,
    "2.f": BugSymptom.OTHER,
    "2.g": BugSymptom.NOT_A_BUG,
}

class BugHeterogeneity(Enum):
    UNIVERSAL = "3.a no, universal"
    BACKEND_SPECIFIC = "3.b yes, some backend specifically"
    NOT_APPLICABLE = "3.c not applicable"
    DONT_KNOW = "3.d dont know"

BUG_HETEROGENEITY_LOOKUP = {
    "3.a": BugHeterogeneity.UNIVERSAL,
    "3.b": BugHeterogeneity.BACKEND_SPECIFIC,
    "3.c": BugHeterogeneity.NOT_APPLICABLE,
}

def parse_bug_type(code):
    return BUG_TYPE_LOOKUP.get(code)

def parse_bug_symptom(code):
    return BUG_SYMPTOM_LOOKUP.get(code)

def parse_bug_heterogeneity(code):
    return BUG_HETEROGENEITY_LOOKUP.get(code)

prompt = """
please categorize the issue in the three following aspects:

1.  bug type: if it is a bug, it is a source code issue, or caused by stuff at a lower level, like GPU drivers, CUDA toolchain, hardware implementation? the answer to this question should be in one of the following: 
    1.a `not a bug`
    1.b `source code issue`
    1.c `low-level software stack, like GPU driver, CUDA toolchain, or even hardware bugs`
    1.d `user called the wrong api`
    1.e `other`
2. 
    2.a `program not able to compile`
    2.b `crashes during runtime`
    2.c `produces wrong result` (comparing to other gpu or cpu)
    2.d `unexpected runtime duration`
    2.e `unexpected amount of consumed memory`
    2.f `other`
    2.g `not a bug`
3. is the bug universal across all backends, or does it only cause problem within some specific architecture? please only answer 3.b when you have strong evidence. 
    3.a `no, universal`
    3.b `yes, some backend specifically`
    3.c `not applicable`
    3.d `dont know`
please reply with only the code representing your option, with comma splitting in between, like `1.a, 2.f, 3.c`. I don’t need any further explanation. 
The url to the issue is: 
"""

def parse_llm_output(text):
    xs = [x.strip() for x in text.split(',')]
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
                    "parts": [ { "text": f"{prompt}{issue['html_url']}" } ]
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return parse_llm_output(text)
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
                    "content": f"{prompt}{issue['html_url']}"
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        text = response.json()["content"][0]["text"]
        return parse_llm_output(text)
    else:
        return None


issues_categorized = []

for issues in issue_groups:
    selected_issues = random.sample(issues, min(10, len(issues)))    
    # Print title and URL
    for issue in selected_issues:
        title = issue['title']
        url = issue['html_url']
        print(f"{title} \n{url}")
        result = ask_gemini_2_5_flash(issue)
        issues_categorized.append( (title, url, result[0], result[1], result[2] ) )
        for line in list(result):
            print(" - " + line.value)
        print()
    print("\n\n=========================\n\n")
    
print()
print(issues_categorized)