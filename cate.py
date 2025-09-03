import json
import json5
import random
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
# CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected25.json'
# CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected50.json'
CATEGORIZED_FILE_PATH = '/Users/bubblepipe/repo/gpu-bugs/selected50_head25.json'
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
            categorized_data = json5.load(f)
        
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

def has_closing_no_activity_comment(issue):
    """Check if the issue's last comment starts with 'Closing since no activity'."""
    # First fetch comments if not already fetched
    if not hasattr(issue, 'comments_data'):
        fetch_issue_comments(issue)
    
    # Check if there are any comments
    if not hasattr(issue, 'comments_data') or not issue.comments_data:
        return False
    
    # Get the last comment
    last_comment = issue.comments_data[-1]
    comment_body = last_comment.get('body', '').strip()
    
    # Check if it starts with "Closing since no activity" (case-insensitive)
    return comment_body.lower().startswith('closing since no activity')

def select_random_uncategorized_issues(issue_groups, categorized_urls, num_per_framework=NUM_PER_FRAMEWORK):
    """Select random uncategorized issues from each framework."""
    all_selected_issues = []
    for issues in issue_groups:
        # Find issues that haven't been categorized yet, have at least one comment, 
        # don't have unwanted labels, and don't have "closing since no activity" as last comment
        uncategorized_issues = [issue for issue in issues 
                              if issue.html_url not in categorized_urls 
                              and hasattr(issue, 'comments_count') 
                              and issue.comments_count > 0
                              and not has_unwanted_labels(issue)
                              and not has_closing_no_activity_comment(issue)]
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


def extract_github_urls_from_text(text, base_repo_url=None):
    """Extract GitHub issue and PR URLs from text content, including shorthand references.
    
    Args:
        text: Text content to search for URLs
        base_repo_url: Base repository URL (e.g., "https://github.com/pytorch/pytorch") 
                      for resolving shorthand references like #123
    
    Returns:
        List of GitHub issue/PR URLs found in the text
    """
    import re
    urls = []
    
    # Debug: Check if we're looking at text with #10394 or #10395
    # if '#10394' in text or '#10395' in text:
    #     print(f"DEBUG: Found #10394 or #10395 in text! Base repo: {base_repo_url}")
    #     print(f"DEBUG: Text snippet: {text[:200]}...")
    
    # 1. Extract full GitHub URLs
    full_url_pattern = r'https://github\.com/([\w\-]+)/([\w\-]+)/(?:issues|pull)/(\d+)'
    full_urls = re.findall(full_url_pattern, text)
    for owner, repo, number in full_urls:
        urls.append(f'https://github.com/{owner}/{repo}/issues/{number}')
    
    # 2. Extract cross-repo references (owner/repo#123)
    cross_repo_pattern = r'(?:^|[^/\w])([\w\-]+)/([\w\-]+)#(\d+)'
    cross_refs = re.findall(cross_repo_pattern, text)
    for owner, repo, number in cross_refs:
        # Validate that these look like real GitHub owner/repo names
        if len(owner) > 0 and len(repo) > 0 and not owner.isdigit():
            urls.append(f'https://github.com/{owner}/{repo}/issues/{number}')
    
    # 3. Extract same-repo references (#123) - only if base_repo_url is provided
    if base_repo_url:
        # Clean up base URL to ensure it's in the right format
        base_match = re.match(r'https://github\.com/([\w\-]+)/([\w\-]+)', base_repo_url)
        if base_match:
            base_owner, base_repo = base_match.groups()
            
            # Match #123 style references (but not if part of owner/repo#123)
            # Also exclude stack trace patterns like "#10 0x00007..." 
            # Stack traces have #N followed by space and hex address (0x...)
            # Use word boundary after the number to ensure we get the full number
            same_repo_pattern = r'(?:^|[^/\w])#(\d+)\b(?!\s+0x)'
            same_repo_refs = re.findall(same_repo_pattern, text)
            for number in same_repo_refs:
                url = f'https://github.com/{base_owner}/{base_repo}/issues/{number}'
                # if '10394' in number or '10395' in number:
                #     print(f"DEBUG: Extracted #{number} -> {url}")
                urls.append(url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        # Normalize /pull/ to /issues/ for consistency
        url = url.replace('/pull/', '/issues/')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def fetch_mentioned_issue_content_recursive(url, cache={}, visited=None, depth=0, max_depth=3):
    """Recursively fetch content of mentioned issues/PRs with cycle detection.
    
    Args:
        url: GitHub issue/PR URL
        cache: Dictionary to cache fetched content
        visited: Set of URLs already being processed (for cycle detection)
        depth: Current recursion depth
        max_depth: Maximum recursion depth allowed
    
    Returns:
        Dictionary with 'content' (string) and 'related' (list of related content)
    """
    # Initialize visited set if not provided
    if visited is None:
        visited = set()
    
    # Debug: Track recursion for specific issues
    # if '10394' in url or '10395' in url:
    #     print(f"DEBUG: Recursive fetch for {url} at depth {depth}")
    
    # Check if we've reached max depth
    if depth >= max_depth:
        # if '10394' in url or '10395' in url:
        #     print(f"DEBUG: Max depth reached for {url}!")
        return None
    
    # Check for cycles
    if url in visited:
        print(f"Cycle detected: {url} already being processed")
        return None
    
    # Add to visited set
    visited.add(url)
    
    # Fetch the main content
    content = fetch_mentioned_issue_content(url, cache)
    if not content:
        visited.remove(url)  # Remove from visited if fetch failed
        return None
    
    # Extract base repo URL from the current URL for resolving shorthand references
    import re
    base_repo_match = re.match(r'(https://github\.com/[\w\-]+/[\w\-]+)', url)
    base_repo_url = base_repo_match.group(1) if base_repo_match else None
    
    # Extract URLs from the fetched content, including shorthand references
    mentioned_urls = extract_github_urls_from_text(content, base_repo_url)
    
    # Debug: Show what URLs were extracted
    # if mentioned_urls and depth < 3:
    #     print(f"DEBUG: At depth {depth}, extracted {len(mentioned_urls)} URLs from {url}")
    #     for i, u in enumerate(mentioned_urls[:5]):
    #         print(f"  {i+1}. {u}")
    
    # Recursively fetch mentioned issues/PRs
    related_content = []
    # Debug: Show which URLs will be processed  
    # if mentioned_urls:
    #     print(f"DEBUG: Processing first 3 of {len(mentioned_urls)} URLs at depth {depth}")
    #     for i, u in enumerate(mentioned_urls[:3]):
    #         status = "SKIP (visited)" if u in visited else "FETCH"
    #         print(f"  {i+1}. {u} - {status}")
    
    for mentioned_url in mentioned_urls[:3]:  # Limit to 3 related items per level
        if mentioned_url not in visited:  # Skip if already visited
            # if '10394' in mentioned_url or '10395' in mentioned_url:
            #     print(f"DEBUG: About to recursively fetch {mentioned_url} at depth {depth+1}")
            related = fetch_mentioned_issue_content_recursive(
                mentioned_url, cache, visited, depth + 1, max_depth
            )
            if related:
                # Extract issue/PR number from URL for labeling
                import re
                match = re.search(r'/(\d+)$', mentioned_url)
                number = match.group(1) if match else "Unknown"
                issue_or_pr = "PULL REQUEST" if '/pull/' in mentioned_url else "ISSUE"
                
                related_item = {
                    'url': mentioned_url,
                    'number': number,
                    'type': issue_or_pr,
                    'depth': depth + 1,
                    'content': related['content'] if isinstance(related, dict) else related,
                    'related': related.get('related', []) if isinstance(related, dict) else []
                }
                related_content.append(related_item)
    
    # Remove from visited set after processing
    visited.remove(url)
    
    return {
        'content': content,
        'related': related_content
    }


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
                                    # Debug: Check if this comment mentions our issues
                                    # if '#10394' in body or '#10395' in body:
                                    #     print(f"DEBUG: Comment {i} by {user} mentions #10394 or #10395!")
                                    #     print(f"DEBUG: Comment text: {body[:200]}...")
                    except:
                        pass  # Ignore comment fetching errors
                
                # For Pull Requests, fetch the diff
                if '/pull/' in url:
                    try:
                        # Use the diff endpoint for the PR
                        diff_headers = headers.copy()
                        diff_headers['Accept'] = 'application/vnd.github.v3.diff'
                        
                        diff_response = requests.get(api_url, headers=diff_headers, timeout=30)
                        if diff_response.status_code == 200:
                            diff_text = diff_response.text
                            # Limit diff size to avoid overwhelming the prompt
                            max_diff_length = 3000
                            if len(diff_text) > max_diff_length:
                                diff_text = diff_text[:max_diff_length] + "\n... (diff truncated)"
                            
                            content += "\n=== CODE CHANGES (DIFF) ===\n"
                            content += diff_text
                            content += "\n"
                    except Exception as e:
                        print(f"Failed to fetch diff for PR {url}: {e}")
                        # Continue without diff if it fails
                
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



def format_recursive_mentions(related_items, indent_level=0):
    """Format recursively fetched related issues/PRs with proper indentation.
    
    Args:
        related_items: List of related items with content and sub-relations
        indent_level: Current indentation level for nested items
    
    Returns:
        Formatted string with all related content
    """
    formatted = ""
    indent = "  " * indent_level
    
    for item in related_items:
        # Add header for this item
        formatted += f"\n\n{indent}=== {item['type']} #{item['number']} (Depth {item['depth']}) ===\n"
        formatted += item['content']
        
        # Recursively add any related items from this one
        if item.get('related'):
            formatted += format_recursive_mentions(item['related'], indent_level + 1)
    
    return formatted


def prepare_full_prompt(issue):
    # Fetch timeline data for this specific issue only when needed
    if hasattr(issue, 'fetch_timeline'):
        issue.fetch_timeline()
    
    # Collect all mentions using the Issue's method
    issue.collect_all_mentions()
    
    # Print mention report for debugging
    issue.print_mention_report()
    
    issue_content = issue.to_string_pretty()
    full_prompt = BUG_CATEGORIZATION_PROMPT + "\n\nISSUE CONTENT:\n" + issue_content
    
    # Prepare for recursive fetching
    content_cache = {}  # Cache to avoid re-fetching
    visited = set()  # Track visited URLs to prevent cycles
    all_related_content = []
    
    # Process all collected unique URLs with recursive fetching
    max_mentioned_items = 10
    
    # Get source mapping for each URL
    url_to_sources = {}
    for source, urls in issue.mentioned_urls_by_source.items():
        for url in urls:
            if url not in url_to_sources:
                url_to_sources[url] = []
            url_to_sources[url].append(source)
    
    # Process unique URLs (from the deduplicated set)
    for url in list(issue.all_mentioned_urls)[:max_mentioned_items]:
        result = fetch_mentioned_issue_content_recursive(
            url, content_cache, visited, depth=0
        )
        
        if result:
            # Extract issue/PR number from URL
            import re
            match = re.search(r'/(\d+)$', url)
            number = match.group(1) if match else "Unknown"
            
            # Get all sources for this URL
            sources = url_to_sources.get(url, ['UNKNOWN'])
            source_str = ', '.join(sources)
            
            # Determine if it's a PR or issue
            is_pr = '/pull/' in url or 'Pull Request' in result.get('content', '')[:200]
            issue_or_pr = "PULL REQUEST" if is_pr else "ISSUE"
            
            related_item = {
                'url': url,
                'number': number,
                'type': f'{issue_or_pr} ({source_str})',
                'depth': 0,
                'content': result['content'],
                'related': result.get('related', [])
            }
            all_related_content.append(related_item)
    
    # Format and append all related content
    if all_related_content:
        full_prompt += "\n\n--- RELATED ISSUES AND PULL REQUESTS (WITH NESTED MENTIONS) ---"
        full_prompt += format_recursive_mentions(all_related_content)
    
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
        api_key = os.getenv("NEKO_DEFAULT_API_KEY")
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
    
# Update system message to encourage reasoning befoyere the final answer
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
    print()
    print(full_prompt)
    print()
    
    # Determine which API to use based on configuration
    if OPUS_API_PROVIDER == "neko":
        # Use NekoAPI endpoint for Claude models
        api_key = os.getenv("NEKO_CLAUDE_API_KEY")
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
        "model": "claude-opus-4-1-20250805-thinking", 
        "group": "claude",
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        json_response = response.json()
        
        # Handle different response formats (with or without thinking)
        content = json_response.get("content", [])
        text = None
        
        # Look for the text content (skip thinking content)
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                break
        
        # Fallback to old format if no text type found
        if text is None and content and isinstance(content[0], dict):
            text = content[0].get("text", "")
        
        if not text:
            raise KeyError("No text content found in response")
            
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
        if LLM_CHOICE == "gemini-pro":
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
            print(f"Unknown LLM choice: {LLM_CHOICE}")
            
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
            print(" - " + item.value, file=sys.stderr)
        print()
        exit() # Temporary exit for testing one issue at a time
    
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