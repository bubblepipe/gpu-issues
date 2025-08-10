import json
import random
import requests
import os
import sys
import time
from result import Ok, Err, Result
from prompts import BUG_CATEGORIZATION_PROMPT
from cates import IS_REALLY_BUG_LOOKUP, USER_PERSPECTIVE_LOOKUP, DEVELOPER_PERSPECTIVE_LOOKUP, ACCELERATOR_SPECIFIC_LOOKUP, USER_EXPERTISE_LOOKUP
from results_loader import load_categorized_results, get_categorized_urls


# Choose source for issues: either from fresh selection or from previously categorized file
USE_CATEGORIZED_FILE = False  # Set to False to select fresh issues
NUM_PER_FRAMEWORK = 1

def load_issues_from_categorized_file(categorized_file_path, issue_groups):
    """Load issues from a previously categorized JSON file."""
    try:
        with open(categorized_file_path, 'r') as f:
            categorized_data = json.load(f)
        
        # Extract URLs from categorized data
        all_selected_issues = []
        for item in categorized_data:
            # Find the original issue from issue_groups
            found = False
            for issues in issue_groups:
                for issue in issues:
                    if issue['html_url'] == item['url']:
                        all_selected_issues.append(issue)
                        found = True
                        break
                if found:
                    break
        
        print(f"Loaded {len(all_selected_issues)} issues from {categorized_file_path}")
        return all_selected_issues
    except FileNotFoundError:
        print(f"File not found: {categorized_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from: {categorized_file_path}")
        return []

def select_random_uncategorized_issues(issue_groups, categorized_urls, num_per_framework=NUM_PER_FRAMEWORK):
    """Select random uncategorized issues from each framework."""
    all_selected_issues = []
    for issues in issue_groups:
        # Find issues that haven't been categorized yet
        uncategorized_issues = [issue for issue in issues if issue['html_url'] not in categorized_urls]
        
        num_to_select = min(num_per_framework, len(uncategorized_issues))
        if num_to_select > 0:
            selected_issues = random.sample(uncategorized_issues, num_to_select)
            all_selected_issues.extend(selected_issues)
            print(f"Selected {num_to_select} uncategorized issues from framework")
        else:
            print(f"All issues in this framework have already been categorized")
    
    return all_selected_issues

def fetch_issue_comments(issue_url):
    """Fetch comments for a GitHub issue."""
    # Extract owner, repo, and issue number from URL
    # Example: https://github.com/pytorch/pytorch/issues/12345
    parts = issue_url.replace('https://github.com/', '').split('/')
    if len(parts) < 4 or parts[2] != 'issues':
        return []
    
    owner = parts[0]
    repo = parts[1]
    issue_number = parts[3]
    
    # Construct API URL for comments
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    
    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }
    
    # Add GitHub token if available
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    try:
        response = requests.get(api_url, headers=headers)
        
        # Handle rate limiting
        if response.status_code == 403:
            print(f"Rate limited when fetching comments. Waiting...")
            time.sleep(60)
            return []
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

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
    
    # Print comments if available
    if 'comments_data' in issue and issue['comments_data']:
        print()
        print(f"**Comments ({len(issue['comments_data'])}):**")
        print()
        for i, comment in enumerate(issue['comments_data'], 1):
            print(f"#### Comment {i} by {comment.get('user', {}).get('login', 'Unknown')} at {comment.get('created_at', 'Unknown')}")
            print()
            if comment.get('body'):
                print("> " + comment['body'].replace('\n', '\n> '))
            else:
                print("> _(empty comment)_")
            print()
    elif 'comments' in issue and issue['comments'] > 0:
        # Comments count is available but actual comments not fetched
        print()
        print(f"**Comments:** {issue['comments']} comments available (not fetched)")
    
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



# Load categorized issues from JSON files
# Update the pattern to match your JSON files location
categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/categorized_issues_*.json')

def parse_is_really_bug(code):
    return IS_REALLY_BUG_LOOKUP.get(code)

def parse_user_perspective(code):
    return USER_PERSPECTIVE_LOOKUP.get(code)

def parse_developer_perspective(code):
    return DEVELOPER_PERSPECTIVE_LOOKUP.get(code)

def parse_accelerator_specific(code):
    return ACCELERATOR_SPECIFIC_LOOKUP.get(code)

def parse_user_expertise(code):
    return USER_EXPERTISE_LOOKUP.get(code)


def parse_llm_output(text):
    # Try to find pattern like "1.x, 2.x, 3.x, 4.x, 5.x" anywhere in the text
    import re
    pattern = r'([1-5]\.[a-l]),\s*([1-5]\.[a-l]),\s*([1-5]\.[a-l]),\s*([1-5]\.[a-l]),\s*([1-5]\.[a-l])'
    match = re.search(pattern, text)
    
    if match:
        xs = list(match.groups())
    else:
        # Fallback to original parsing
        xs = [x.strip() for x in text.split(',')]
        if len(xs) != 5:
            return Err(f"Expected 5 comma-separated values, got {len(xs)}. Original response: {text[:200]}...")
    
    if xs[0] not in IS_REALLY_BUG_LOOKUP:
        return Err(f"Invalid is_really_bug code: {xs[0]}. Original response: {text}")
    
    if xs[1] not in USER_PERSPECTIVE_LOOKUP:
        return Err(f"Invalid user_perspective code: {xs[1]}. Original response: {text}")
    
    if xs[2] not in DEVELOPER_PERSPECTIVE_LOOKUP:
        return Err(f"Invalid developer_perspective code: {xs[2]}. Original response: {text}")
    
    if xs[3] not in ACCELERATOR_SPECIFIC_LOOKUP:
        return Err(f"Invalid accelerator_specific code: {xs[3]}. Original response: {text}")
    
    if xs[4] not in USER_EXPERTISE_LOOKUP:
        return Err(f"Invalid user_expertise code: {xs[4]}. Original response: {text}")
    
    return Ok((parse_is_really_bug(xs[0]), parse_user_perspective(xs[1]), parse_developer_perspective(xs[2]), parse_accelerator_specific(xs[3]), parse_user_expertise(xs[4])))

def ask_gemini_2_5_flash(issue):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return Err("GEMINI_API_KEY environment variable not set")
    
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
        
        # Extract the text from the response
        json_response = response.json()
        text = json_response['candidates'][0]['content']['parts'][0]['text']
        
        # Parse the LLM output and return the Result
        return parse_llm_output(text)
        
    except requests.exceptions.RequestException as e:
        return Err(f"Network error calling Gemini API: {e}")
    except (IndexError, KeyError) as e:
        return Err(f"Error parsing response from Gemini API: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
    except ValueError as e:
        return Err(f"Invalid JSON response from Gemini API: {e}")
    except Exception as e:
        return Err(f"Unexpected error with Gemini API: {e}")


def ask_local_ollama(issue):
    """Query Ollama API on remote h100 server with full issue content including comments."""
    import subprocess
    import json as json_module
    
    ollama_url = "http://localhost:11434/api/generate"
    model = "qwen3:235b"
    
    # Build the full issue content including body and comments
    issue_content = f"Title: {issue['title']}\n"
    issue_content += f"URL: {issue['html_url']}\n"
    
    # Add labels
    labels = [label['name'] for label in issue.get('labels', [])]
    if labels:
        issue_content += f"Labels: {', '.join(labels)}\n"
    
    issue_content += "\nIssue Description:\n"
    if issue.get('body'):
        issue_content += issue['body'] + "\n"
    else:
        issue_content += "(No description provided)\n"
    
    # Add comments if available
    if 'comments_data' in issue and issue['comments_data']:
        issue_content += f"\n--- Comments ({len(issue['comments_data'])}) ---\n"
        for i, comment in enumerate(issue['comments_data'], 1):
            issue_content += f"\nComment {i} by {comment.get('user', {}).get('login', 'Unknown')} at {comment.get('created_at', 'Unknown')}:\n"
            if comment.get('body'):
                issue_content += comment['body'] + "\n"
            else:
                issue_content += "(Empty comment)\n"
    
    # Combine prompt with issue content
    full_prompt = BUG_CATEGORIZATION_PROMPT + "\n\nISSUE CONTENT:\n" + issue_content
    
    # Prepare JSON data for curl
    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent categorization
            "num_predict": 30,   # We need a slightly longer response like "1.d, 2.c, 3.b, 4.a, 5.b"
            "num_ctx": 4096,     # Limit context window to speed up processing
            "repeat_penalty": 1.0
        }
    }
    
    # Convert data to JSON string
    json_data = json_module.dumps(data)
    
    # Create a local temporary file
    import tempfile
    import uuid
    
    # Write JSON to local temp file
    local_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    local_temp.write(json_data)
    local_temp.close()
    
    # Remote temp filename
    remote_temp = f"/tmp/ollama_request_{uuid.uuid4().hex}.json"
    
    # Build the SSH + SCP + curl command
    # First copy the file to remote, then use it with curl, then clean up both files
    ssh_command = f"""
    scp {local_temp.name} h100:{remote_temp} && \
    ssh h100 'curl -s -X POST {ollama_url} -H "Content-Type: application/json" -d @{remote_temp} --max-time 120; rm -f {remote_temp}' && \
    rm -f {local_temp.name}
    """
    
    try:
        # Print debug info
        print(f"Sending request to Ollama for: {issue.get('title', 'Unknown')[:50]}...")
        
        # Print the command for debugging
        print(f"DEBUG: SSH Command:\n{ssh_command}\n")
        print(f"DEBUG: Local temp file: {local_temp.name}")
        print(f"DEBUG: Remote temp file: {remote_temp}")
        
        # Execute SSH command (use shell=True for complex command with pipes)
        result = subprocess.run(
            ssh_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=130  # Increased timeout for large model
        )
        
        if result.returncode != 0:
            return Err(f"SSH/curl command failed: {result.stdout} \n{result.stderr}")
        
        # Debug output
        if not result.stdout:
            return Err(f"Empty response from Ollama. stderr: {result.stderr}")
        
        # Parse the JSON response
        try:
            json_response = json_module.loads(result.stdout)
        except json_module.JSONDecodeError as e:
            return Err(f"Failed to parse JSON. Response: {result.stdout[:500]}... Error: {e}")
        text = json_response.get('response', '').strip()
        
        # Debug: print the raw response
        print(f"DEBUG: Raw Ollama response: {text[:200]}...")
        
        # Parse the LLM output and return the Result
        return parse_llm_output(text)
        
    except subprocess.TimeoutExpired:
        # Clean up local temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass
        return Err(f"SSH command timed out after 130 seconds - Ollama may be overloaded or the model is too slow")
    except json_module.JSONDecodeError as e:
        # Clean up local temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass
        return Err(f"Invalid JSON response from Ollama API: {e}. Response: {result.stdout if 'result' in locals() else 'No response'}")
    except subprocess.SubprocessError as e:
        # Clean up local temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass
        return Err(f"SSH subprocess error: {e}")
    except Exception as e:
        # Clean up local temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass
        return Err(f"Unexpected error with remote Ollama API: {e}")

def ask_sonnet_4(issue):
    return None

def ask_opus_4(issue):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return Err("ANTHROPIC_API_KEY environment variable not set")
    
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
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        json_response = response.json()
        text = json_response["content"][0]["text"]
        
        return parse_llm_output(text)
        
    except requests.exceptions.RequestException as e:
        return Err(f"Network error calling Anthropic API: {e}")
    except (IndexError, KeyError) as e:
        return Err(f"Error parsing response from Anthropic API: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
    except ValueError as e:
        return Err(f"Invalid JSON response from Anthropic API: {e}")
    except Exception as e:
        return Err(f"Unexpected error with Anthropic API: {e}")


issues_categorized = []

# Create a set of URLs that are already categorized for fast lookup
categorized_urls = get_categorized_urls(categorized_issues)



all_selected_issues = []
if USE_CATEGORIZED_FILE:
    # Load issues from previously categorized file
    categorized_file_path = '/Users/bubblepipe/repo/gpu-bugs/categorized/categorized_issues_20250808_041918.json'
    all_selected_issues = load_issues_from_categorized_file(categorized_file_path, issue_groups)
else:
    # Select random uncategorized issues
    all_selected_issues = select_random_uncategorized_issues(issue_groups, categorized_urls, num_per_framework=40)

print(f"\nTotal issues selected: {len(all_selected_issues)}")
print("\n\n=========================\n\n")


for issue in all_selected_issues:
    # Fetch comments for this issue
    comments = fetch_issue_comments(issue['html_url'])
    issue['comments_data'] = comments
    # print_issue(issue)
    # exit()
    
    title = issue['title']
    url = issue['html_url']
    print(f"{title} \n{url}")
    
    # Choose which LLM to use
    result = ask_gemini_2_5_flash(issue)
    # result = ask_local_ollama(issue)
    # result = ask_opus_4(issue)
    if result.is_err():
        error_msg = result.unwrap_err()
        sys.stderr.write(f"Failed to categorize: {title} - {url}\n")
        sys.stderr.write(f"Error: {error_msg}\n")
        print()
        # exit()
        continue
    
    # Unwrap the successful result
    categorization = result.unwrap()
    issues_categorized.append( (title, url, categorization[0], categorization[1], categorization[2], categorization[3], categorization[4] ) )
    for item in categorization:
        print(" - " + item.value)
    print()
    # exit()
    

# Save categorized issues to a file
if issues_categorized:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"categorized_issues_{timestamp}.json"
    
    # Convert enum objects to strings for JSON serialization
    json_data = []
    for item in issues_categorized:
        json_data.append({
            "title": item[0],
            "url": item[1],
            "is_really_bug": item[2].value if item[2] else None,
            "user_perspective": item[3].value if item[3] else None,
            "developer_perspective": item[4].value if item[4] else None,
            "accelerator_specific": item[5].value if item[5] else None,
            "user_expertise": item[6].value if item[6] else None
        })
    
    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nSaved {len(issues_categorized)} categorized issues to {output_filename}")
else:
    print("\nNo new issues were categorized.")