import json
import random
from sre_parse import CATEGORIES
import requests
import os
import sys
import time
import urllib3
from result import Ok, Err
from prompts import BUG_CATEGORIZATION_PROMPT
from cates import IS_REALLY_BUG_LOOKUP, USER_PERSPECTIVE_LOOKUP, DEVELOPER_PERSPECTIVE_LOOKUP, ACCELERATOR_SPECIFIC_LOOKUP, PLATFORM_SPECIFICITY_LOOKUP
from results_loader import load_categorized_results, get_categorized_urls
from issue import Issue

# Suppress SSL warnings when we fallback to unverified requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected_examples.json'
CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected25.json'
# CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected50.json'
USE_CATEGORIZED_FILE = True # Set to False to select fresh issues
# USE_CATEGORIZED_FILE = False  # Set to False to select fresh issues

NUM_PER_FRAMEWORK = 10

# Options: "gemini", "gemini-pro", "ollama", "opus", "gpt5", "dummy"
# LLM_CHOICE = "gemini-pro"  
# LLM_CHOICE = "dummy"  
# LLM_CHOICE = "gpt5"
LLM_CHOICE = "opus"  

# GPT-5/OpenAI API Configuration
# Set to "openai" for official OpenAI API or "neko" for NekoAPI alternative
GPT_API_PROVIDER = "neko"  # Options: "openai" or "neko"
NEKO_API_BASE = "https://nekoapi.com/v1"  # NekoAPI endpoint

# Opus/Claude API Configuration
# Set to "anthropic" for official Anthropic API or "neko" for NekoAPI alternative
OPUS_API_PROVIDER = "neko"  # Options: "anthropic" or "neko"

OLLAMA_MODEL = "qwen3:235b"  # Change this to match your available model


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
                    if issue.html_url == item['url']:
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

def has_unwanted_labels(issue):
    """Check if issue has stale or awaiting response labels."""
    if not hasattr(issue, 'labels'):
        return False
    for label in issue.labels:
        label_name = label.get('name', '').lower()
        if 'stale' in label_name or 'awaiting response' in label_name:
            return True
    return False

def select_random_uncategorized_issues(issue_groups, categorized_urls, num_per_framework=NUM_PER_FRAMEWORK):
    """Select random uncategorized issues from each framework."""
    all_selected_issues = []
    for issues in issue_groups:
        # Find issues that haven't been categorized yet, have at least one comment, and don't have unwanted labels
        uncategorized_issues = [issue for issue in issues 
                              if issue.html_url not in categorized_urls 
                              and hasattr(issue, 'comments_count') 
                              and issue.comments_count > 0
                              and not has_unwanted_labels(issue)]
        num_to_select = min(num_per_framework, len(uncategorized_issues))
        if num_to_select > 0:
            selected_issues = random.sample(uncategorized_issues, num_to_select)
            all_selected_issues.extend(selected_issues)
            print(f"Selected {num_to_select} uncategorized issues with comments from framework")
        else:
            print(f"No uncategorized issues with comments available in this framework")
    
    return all_selected_issues

def fetch_issue_comments(issue):
    """Fetch comments for a GitHub issue and store in the Issue object."""
    # Extract owner, repo, and issue number from URL
    # Example: https://github.com/pytorch/pytorch/issues/12345
    parts = issue.html_url.replace('https://github.com/', '').split('/')
    if len(parts) < 4 or parts[2] != 'issues':
        return
    
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
            return
        
        response.raise_for_status()
        issue.comments_data = response.json()
    except Exception as e:
        print(f"Error fetching comments: {e}")
        issue.comments_data = []

def print_issue(issue):
    """Pretty print a single issue in markdown format."""
    print(f"### [{issue.title}]({issue.html_url})")
    print()
    
    # Print metadata
    print(f"**Created:** {issue.created_at or 'Unknown'}")
    print()
    
    # Extract tag names from labels
    tags = [label['name'] for label in issue.labels]
    if tags:
        print(f"**Tags:** `{' '.join(tags)}`")
    else:
        print("**Tags:** _(none)_")
    print()
    
    # Print issue body/content
    if issue.body:
        print("**Content:**")
        print()
        # Print full content without truncation
        print("> " + issue.body.replace('\n', '\n> '))
    else:
        print("**Content:** _(empty)_")
    
    # Print comments if available
    if issue.comments_data:
        print()
        print(f"**Comments ({len(issue.comments_data)}):**")
        print()
        for i, comment in enumerate(issue.comments_data, 1):
            print(f"#### Comment {i} by {comment.get('user', {}).get('login', 'Unknown')} at {comment.get('created_at', 'Unknown')}")
            print()
            if comment.get('body'):
                print("> " + comment['body'].replace('\n', '\n> '))
            else:
                print("> _(empty comment)_")
            print()
    
    print()
    print("---")
    print()  # Empty line for readability

def print_issues(issues):
    """Pretty print a list of issues in markdown format."""
    for issue in issues:
        print_issue(issue)

def load_framework_issues(fetch_timelines=False):
    """Load issues from all framework JSON files as Issue objects."""
    frameworks = ['pytorch', 'tensorflow', 'jax', 'tensorrt', 'triton']
    issue_groups = []
    
    for framework in frameworks:
        try:
            with open(f'./issues/{framework}_issues.json', 'r') as f:
                json_issues = json.load(f)
                issues = [Issue.from_json(issue_data, fetch_timeline=fetch_timelines) 
                         for issue_data in json_issues]
                issue_groups.append(issues)
        except FileNotFoundError:
            print(f"{framework}: file not found")
    
    return issue_groups

def parse_is_really_bug(code):
    return IS_REALLY_BUG_LOOKUP.get(code)

def parse_user_perspective(code):
    return USER_PERSPECTIVE_LOOKUP.get(code)

def parse_developer_perspective(code):
    return DEVELOPER_PERSPECTIVE_LOOKUP.get(code)

def parse_accelerator_specific(code):
    return ACCELERATOR_SPECIFIC_LOOKUP.get(code)

def parse_platform_specificity(code):
    return PLATFORM_SPECIFICITY_LOOKUP.get(code)


def fetch_mentioned_issue_content(url, cache={}):
    """Fetch the content of a mentioned issue or PR from GitHub API.
    
    Args:
        url: GitHub issue/PR URL
        cache: Dictionary to cache fetched content
    
    Returns:
        String containing the issue/PR content or None if failed
    """
    # Check cache first
    if url in cache:
        return cache[url]
    
    # Convert HTML URL to API URL
    # Example: https://github.com/pytorch/pytorch/issues/123 -> https://api.github.com/repos/pytorch/pytorch/issues/123
    if not url.startswith('https://github.com/'):
        return None
    
    api_url = url.replace('https://github.com/', 'https://api.github.com/repos/')
    api_url = api_url.replace('/pull/', '/pulls/')  # Handle PRs
    
    headers = {'Accept': 'application/vnd.github.v3+json'}
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    # Retry logic for SSL and connection errors
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Add timeout to prevent hanging
            response = requests.get(api_url, headers=headers, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 403:
                print(f"Rate limited when fetching {url}")
                return None
            
            if response.status_code == 200:
                data = response.json()
                
                # Build content string
                content = f"Title: {data.get('title', 'Unknown')}\n"
                content += f"State: {data.get('state', 'unknown')}\n"
                content += f"Created: {data.get('created_at', 'unknown')}\n"
                
                # Add labels
                labels = data.get('labels', [])
                if labels:
                    label_names = [label['name'] for label in labels]
                    content += f"Labels: {', '.join(label_names)}\n"
                
                # Add body
                body = data.get('body', '')
                if body:
                    content += f"\nDescription:\n{body}\n"
                else:
                    content += "\nDescription: (empty)\n"
                
                # Fetch first few comments (limit to 3 to avoid huge prompts)
                comments_url = data.get('comments_url')
                if comments_url:
                    try:
                        comments_response = requests.get(comments_url, headers=headers, timeout=30)
                        if comments_response.status_code == 200:
                            comments = comments_response.json()[:3]  # Limit to first 3 comments
                            if comments:
                                content += f"\nFirst {len(comments)} comments:\n"
                                for i, comment in enumerate(comments, 1):
                                    user = comment.get('user', {}).get('login', 'Unknown')
                                    body = comment.get('body', '(empty)')
                                    content += f"\nComment {i} by {user}:\n{body}\n"
                    except:
                        pass  # Ignore comment fetching errors
                
                # Cache the result
                cache[url] = content
                return content
                
        except requests.exceptions.SSLError as e:
            if attempt < max_retries - 1:
                print(f"SSL error fetching {url}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Last attempt - try without SSL verification
                try:
                    print(f"SSL verification failed for {url}, trying without verification...")
                    response = requests.get(api_url, headers=headers, timeout=30, verify=False)
                    if response.status_code == 200:
                        data = response.json()
                        # Build minimal content without comments
                        content = f"Title: {data.get('title', 'Unknown')}\n"
                        content += f"State: {data.get('state', 'unknown')}\n"
                        body = data.get('body', '')
                        if body:
                            content += f"\nDescription:\n{body[:500]}...\n" if len(body) > 500 else f"\nDescription:\n{body}\n"
                        cache[url] = content
                        return content
                except:
                    print(f"Failed to fetch {url} even without SSL verification")
                    return None
                    
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url} (attempt {attempt + 1}/{max_retries})")
            if attempt >= max_retries - 1:
                return None
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching {url}: {e}")
            if attempt >= max_retries - 1:
                return None
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except Exception as e:
            print(f"Unexpected error fetching {url}: {e}")
            return None
    
    return None


def parse_llm_output(text):
    # Split response by lines and get the last non-empty line
    lines = text.strip().split('\n')
    last_line = None
    
    # Find the last non-empty line
    for line in reversed(lines):
        if line.strip():
            last_line = line.strip()
            break
    
    if not last_line:
        print(f"DEBUG: No non-empty lines found in response. Full response:\n{text}")
        return Err(f"No non-empty lines found in response")
    
    # Try to find pattern in the last line
    import re
    pattern = r'([1-5]\.[a-i]),\s*([1-5]\.[a-i]),\s*([1-5]\.[a-i]),\s*([1-5]\.[a-i]),\s*([1-5]\.[a-i])'
    match = re.search(pattern, last_line)
    
    if match:
        xs = list(match.groups())
    else:
        # If pattern not found in last line, try splitting by comma
        xs = [x.strip() for x in last_line.split(',')]
        if len(xs) != 5:
            # Log the full response for debugging
            print(f"DEBUG: Failed to parse LLM output. Last line: {last_line}")
            print(f"DEBUG: Full response:\n{text}")
            return Err(f"Expected 5 comma-separated values in last line, got {len(xs)}. Last line: {last_line}")
    
    if xs[0] not in IS_REALLY_BUG_LOOKUP:
        print(f"DEBUG: Invalid is_really_bug code: {xs[0]}")
        print(f"DEBUG: Full response:\n{text}")
        return Err(f"Invalid is_really_bug code: {xs[0]}. Last line: {last_line}")
    
    if xs[1] not in USER_PERSPECTIVE_LOOKUP:
        print(f"DEBUG: Invalid user_perspective code: {xs[1]}")
        print(f"DEBUG: Full response:\n{text}")
        return Err(f"Invalid user_perspective code: {xs[1]}. Last line: {last_line}")
    
    if xs[2] not in DEVELOPER_PERSPECTIVE_LOOKUP:
        print(f"DEBUG: Invalid developer_perspective code: {xs[2]}")
        print(f"DEBUG: Full response:\n{text}")
        return Err(f"Invalid developer_perspective code: {xs[2]}. Last line: {last_line}")
    
    if xs[3] not in ACCELERATOR_SPECIFIC_LOOKUP:
        print(f"DEBUG: Invalid accelerator_specific code: {xs[3]}")
        print(f"DEBUG: Full response:\n{text}")
        return Err(f"Invalid accelerator_specific code: {xs[3]}. Last line: {last_line}")
    
    if xs[4] not in PLATFORM_SPECIFICITY_LOOKUP:
        print(f"DEBUG: Invalid platform_specificity code: {xs[4]}")
        print(f"DEBUG: Full response:\n{text}")
        return Err(f"Invalid platform_specificity code: {xs[4]}. Last line: {last_line}")
    
    return Ok((parse_is_really_bug(xs[0]), parse_user_perspective(xs[1]), parse_developer_perspective(xs[2]), parse_accelerator_specific(xs[3]), parse_platform_specificity(xs[4])))



def prepare_full_prompt(issue):
    # Fetch timeline data for this specific issue only when needed
    if hasattr(issue, 'fetch_timeline'):
        issue.fetch_timeline()
    
    issue_content = issue.to_string_pretty()
    full_prompt = BUG_CATEGORIZATION_PROMPT + "\n\nISSUE CONTENT:\n" + issue_content
    
    # Append mentioned issues and PRs from timeline
    mentioned_content = []
    content_cache = {}  # Cache to avoid re-fetching
    
    # Limit total number of mentioned items to avoid huge prompts
    max_mentioned_items = 10
    items_added = 0
    
    # Add mentioned PRs
    if hasattr(issue, 'mentioned_prs') and issue.mentioned_prs:
        for pr in issue.mentioned_prs[:max_mentioned_items]:
            if items_added >= max_mentioned_items:
                break
            pr_content = fetch_mentioned_issue_content(pr.html_url, content_cache)
            if pr_content:
                mentioned_content.append(f"\n\n=== MENTIONED PULL REQUEST #{pr.number} ===\n{pr_content}")
                items_added += 1
    
    # Add mentioned issues
    if hasattr(issue, 'mentioned_issues') and issue.mentioned_issues:
        for mentioned_issue in issue.mentioned_issues[:max_mentioned_items - items_added]:
            if items_added >= max_mentioned_items:
                break
            issue_url = mentioned_issue.get('url')
            if issue_url:
                issue_content = fetch_mentioned_issue_content(issue_url, content_cache)
                if issue_content:
                    mentioned_content.append(f"\n\n=== MENTIONED ISSUE #{mentioned_issue.get('number')} ===\n{issue_content}")
                    items_added += 1
    
    # Append all mentioned content to the prompt
    if mentioned_content:
        full_prompt += "\n\n--- RELATED ISSUES AND PULL REQUESTS FROM TIMELINE ---"
        full_prompt += "".join(mentioned_content)
    
    return full_prompt


def ask_gemini_2_5_pro(issue):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return Err("GEMINI_API_KEY environment variable not set")
    
    # Prepare the full prompt with issue content
    full_prompt = prepare_full_prompt(issue)
    print(full_prompt)
    print()

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    data = {
        "contents": [
            {
                "parts": [ { "text": full_prompt } ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Extract the text from the response
        json_response = response.json()
        text = json_response['candidates'][0]['content']['parts'][0]['text']
        print(text)
        print()
        
        # Parse the LLM output and return the Result
        return parse_llm_output(text)
        
    except requests.exceptions.RequestException as e:
        return Err(f"Network error calling Gemini Pro API: {e}")
    except (IndexError, KeyError) as e:
        return Err(f"Error parsing response from Gemini Pro API: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
    except ValueError as e:
        return Err(f"Invalid JSON response from Gemini Pro API: {e}")
    except Exception as e:
        return Err(f"Unexpected error with Gemini Pro API: {e}")


def ask_gpt5(issue):
    """Query OpenAI's latest GPT model using the OpenAI API or compatible alternative."""
    import openai
    import os
    
    # Determine which API to use based on configuration
    if GPT_API_PROVIDER == "neko":
        # Use NekoAPI
        api_key = os.getenv("NEKO_API_KEY")
        if not api_key:
            return Err("NEKO_API_KEY not found. Please set the NEKO_API_KEY environment variable.")
        api_base = NEKO_API_BASE
        print(f"Using NekoAPI endpoint: {api_base}")
    else:
        # Use OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return Err("OPENAI_API_KEY not found. Please set the OPENAI_API_KEY environment variable.")
        api_base = None  # Use default OpenAI endpoint
        print("Using OpenAI API")
    
    # Prepare the full prompt with issue content
    full_prompt = prepare_full_prompt(issue)
    
    try:
        # Initialize OpenAI client
        if api_base:
            # Use alternative API endpoint
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base
            )
        else:
            # Use default OpenAI API
            client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-5",  
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing GitHub issues for GPU-accelerated machine learning frameworks. Analyze the issue thoroughly and provide categorization codes."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
        )
        
        # Extract the response text
        text = response.choices[0].message.content
        print()
        print(text)
        print()
        return parse_llm_output(text)
        
    except openai.APIError as e:
        return Err(f"OpenAI API error: {e}")
    except openai.APIConnectionError as e:
        return Err(f"Failed to connect to OpenAI API: {e}")
    except openai.RateLimitError as e:
        return Err(f"OpenAI API rate limit exceeded: {e}")
    except openai.AuthenticationError as e:
        return Err(f"OpenAI API authentication failed: {e}")
    except Exception as e:
        return Err(f"Unexpected error with OpenAI API: {e}")


def ask_local_ollama(issue):
    """Query Ollama API on remote h100 server with full issue content including comments."""
    import subprocess
    import json as json_module
    
    ollama_url = "http://localhost:11434/api/generate"
    model = OLLAMA_MODEL  # Use the configured model
    
    # Prepare the full prompt with issue content
    full_prompt = prepare_full_prompt(issue)
    
    # Update system message to encourage reasoning before the final answer
    system_message = "Please analyze the issue thoroughly, provide your reasoning, and put the categorization codes in the last line in the format: 1.x, 2.x, 3.x, 4.x, 5.x"
    
    # Prepare JSON data for curl
    data = {
        "model": model,
        "prompt": full_prompt,
        "system": system_message,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent categorization
            "num_predict": 10000,  # Very large to allow full thinking process
            "num_ctx": 16384,    # Use larger context window (16k tokens) - balance between speed and capacity
            "repeat_penalty": 1.0,
            "stop": ["</think>"]  # Only stop at the end of thinking tags
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
    # Add -w flag to curl to show HTTP status code
    # Increased --max-time to 1800 seconds (30 minutes) for very slow models
    ssh_command = f"""
    scp {local_temp.name} h100:{remote_temp} && \
    ssh h100 'curl -s -X POST {ollama_url} -H "Content-Type: application/json" -d @{remote_temp} --max-time 1800 -w "\\nHTTP_STATUS:%{{http_code}}\\n"; rm -f {remote_temp}' && \
    rm -f {local_temp.name}
    """
    
    try:
        # Print minimal info
        print(f"Sending request to Ollama for: {issue.get('title', 'Unknown')[:50]}...")
        
        # Execute SSH command (use shell=True for complex command with pipes)
        result = subprocess.run(
            ssh_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1820  # Increased timeout to 1820 seconds (30+ minutes) for very slow models
        )
        
        if result.returncode != 0:
            return Err(f"SSH/curl command failed: {result.stdout} \n{result.stderr}")
        
        # Debug output
        if not result.stdout:
            return Err(f"Empty response from Ollama. stderr: {result.stderr}")
        
        # Check for HTTP status code in response
        if "HTTP_STATUS:" in result.stdout:
            parts = result.stdout.split("HTTP_STATUS:")
            status_code = parts[-1].strip()
            response_body = parts[0].strip()
            
            if status_code == "404":
                return Err(f"Ollama API returned 404. Make sure Ollama is running on h100 and the model '{model}' is available. Try: ssh h100 'ollama list'")
            elif status_code != "200":
                return Err(f"Ollama API returned HTTP {status_code}. Response: {response_body[:200]}")
        else:
            response_body = result.stdout
        
        # Parse the JSON response
        try:
            json_response = json_module.loads(response_body)
        except json_module.JSONDecodeError as e:
            return Err(f"Failed to parse JSON. Response: {response_body[:500]}... Error: {e}")
        
        # Get the response text and handle backslash escaping
        text = json_response.get('response', '').strip()
        
        # Handle cases where response might be cut off with backslashes
        if text.endswith('\\'):
            # Response was likely truncated, try to extract what we can
            print(f"DEBUG: Response appears truncated (ends with backslash)")
            # Remove trailing backslashes
            text = text.rstrip('\\')
        
        # No special tag handling needed - the parse_llm_output function will extract the last line
        # Just pass the full text to the parser
        
        # Parse the LLM output and return the Result
        return parse_llm_output(text)
        
    except subprocess.TimeoutExpired:
        # Clean up local temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass
        return Err(f"SSH command timed out after 1820 seconds (30 minutes) - Ollama may be overloaded or the model is too slow")
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

def ask_dummy(_):
    """Return a dummy response for testing without API calls."""
    # Simulate some thinking/reasoning
    dummy_response = """Looking at this issue, I need to analyze several aspects:

1. Bug classification: This appears to be a confirmed bug based on the title.
2. User symptoms: Seems like incorrect results are being produced.
3. Root cause: Likely missing safeguards or validation.
4. Resolution status: Not fixed yet.
5. Platform specificity: Universal issue.

Based on my analysis:
1.d, 2.c, 3.b, 4.c, 5.b"""
    
    # Parse and return the dummy response
    return parse_llm_output(dummy_response)

def ask_opus_4(issue):
    """Query Claude Opus using Anthropic API or compatible alternative."""
    
    # Prepare the full prompt with issue content
    full_prompt = prepare_full_prompt(issue)
    
    # Determine which API to use based on configuration
    if OPUS_API_PROVIDER == "neko":
        # Use NekoAPI endpoint for Claude models
        api_key = os.getenv("NEKO_API_KEY")
        if not api_key:
            return Err("NEKO_API_KEY not found. Please set the NEKO_API_KEY environment variable.")
        
        # NekoAPI uses the same format as Anthropic API but different endpoint
        url = f"{NEKO_API_BASE}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        print(f"Using NekoAPI endpoint for Claude Opus 4.1: {url}")
        
    else:
        # Use official Anthropic API
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return Err("ANTHROPIC_API_KEY environment variable not set")
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        print("Using Anthropic API")
    
    # Common request data for both APIs
    data = {
        "model": "claude-opus-4-1-20250805",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }
    
    # Debug prints
    print(f"DEBUG: Request URL: {url}")
    print(f"DEBUG: Headers: {headers}")
    print(f"DEBUG: Model: {data['model']}")
    print(f"DEBUG: Prompt length: {len(full_prompt)} characters")
    print("DEBUG: Sending request...")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Debug response info
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"DEBUG: Response body: {response.text[:500]}")  # First 500 chars of error response
        
        response.raise_for_status()
        
        json_response = response.json()
        print(f"DEBUG: Response JSON keys: {json_response.keys()}")
        
        text = json_response["content"][0]["text"]
        print(text)
        print()
        return parse_llm_output(text)
        
    except requests.exceptions.RequestException as e:
        provider_name = "NekoAPI" if OPUS_API_PROVIDER == "neko" else "Anthropic API"
        return Err(f"Network error calling {provider_name}: {e}")
    except (IndexError, KeyError) as e:
        provider_name = "NekoAPI" if OPUS_API_PROVIDER == "neko" else "Anthropic API"
        return Err(f"Error parsing response from {provider_name}: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
    except ValueError as e:
        provider_name = "NekoAPI" if OPUS_API_PROVIDER == "neko" else "Anthropic API"
        return Err(f"Invalid JSON response from {provider_name}: {e}")
    except Exception as e:
        provider_name = "NekoAPI" if OPUS_API_PROVIDER == "neko" else "Anthropic API"
        return Err(f"Unexpected error with {provider_name}: {e}")


def main():
    """Main function to categorize issues."""
    import datetime
    
    # Load framework issues without timeline data (will be fetched lazily)
    issue_groups = load_framework_issues(fetch_timelines=False)
    
    # Load categorized issues from JSON files
    categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/categorized_issues_*.json')
    
    # Create a set of URLs that are already categorized for fast lookup
    categorized_urls = get_categorized_urls(categorized_issues)
    
    # Select issues to categorize
    all_selected_issues = []
    if USE_CATEGORIZED_FILE:
        # Load issues from previously categorized file
        categorized_file_path = CATEGORIZED_FILE_PATH
        all_selected_issues = load_issues_from_categorized_file(categorized_file_path, issue_groups)
    else:
        # Select random uncategorized issues
        all_selected_issues = select_random_uncategorized_issues(issue_groups, categorized_urls, num_per_framework=NUM_PER_FRAMEWORK)
    
    print(f"\nTotal issues selected: {len(all_selected_issues)}")
    print("\n\n=========================\n\n")
    
    issues_categorized = []
    
    for issue in all_selected_issues:
        # Fetch comments for this issue
        fetch_issue_comments(issue)
        
        title = issue.title
        url = issue.html_url
        print(f"{title} \n{url}")
        
        # Choose which LLM to use based on configuration
        if LLM_CHOICE == "gemini":
            result = ask_gemini_2_5_flash(issue)
        elif LLM_CHOICE == "gemini-pro":
            result = ask_gemini_2_5_pro(issue)
        elif LLM_CHOICE == "ollama":
            result = ask_local_ollama(issue)
        elif LLM_CHOICE == "opus":
            result = ask_opus_4(issue)
        elif LLM_CHOICE == "gpt5":
            result = ask_gpt5(issue)
        elif LLM_CHOICE == "dummy":
            result = ask_dummy(issue)
        else:
            result = ask_gemini_2_5_flash(issue)  # Default to Gemini Flash
            
        if result.is_err():
            error_msg = result.unwrap_err()
            sys.stderr.write(f"Failed to categorize: {title} - {url}\n")
            sys.stderr.write(f"Error: {error_msg}\n")
            print()
            continue
        
        # Unwrap the successful result
        categorization = result.unwrap()
        issues_categorized.append((title, url, categorization[0], categorization[1], categorization[2], categorization[3], categorization[4]))
        for item in categorization:
            print(" - " + item.value)
        print()
    
    # Save categorized issues to a file
    if issues_categorized:
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
                "platform_specificity": item[6].value if item[6] else None
            })
        
        with open(output_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nSaved {len(issues_categorized)} categorized issues to {output_filename}")
    else:
        print("\nNo new issues were categorized.")


if __name__ == "__main__":
    main()