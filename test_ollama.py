#!/usr/bin/env python3
"""Test script for Ollama integration with detailed response debugging."""

import json
import sys
import subprocess
import tempfile
import uuid
import os

# Import necessary components from cate.py
from cate import (
    load_framework_issues, 
    fetch_issue_comments,
    BUG_CATEGORIZATION_PROMPT,
    OLLAMA_MODEL,
    parse_llm_output
)

def test_ollama_with_full_response(issue):
    """Modified version of ask_local_ollama that prints the full response."""
    import json as json_module
    
    ollama_url = "http://localhost:11434/api/generate"
    model = OLLAMA_MODEL
    
    # Build the full issue content
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
            issue_content += f"\nComment {i} by {comment.get('user', {}).get('login', 'Unknown')}:\n"
            if comment.get('body'):
                issue_content += comment['body'][:500] + "\n"  # Truncate long comments for testing
    
    # Modified prompt
    modified_prompt = BUG_CATEGORIZATION_PROMPT.replace(
        "please reply with only the code representing your option",
        "IMPORTANT: Reply with ONLY the codes, no explanation or reasoning. Just output exactly 5 codes separated by commas"
    )
    
    full_prompt = modified_prompt + "\n\nISSUE CONTENT:\n" + issue_content
    system_message = "You must respond with ONLY the categorization codes in the format: 1.x, 2.x, 3.x, 4.x, 5.x. Do not include any reasoning, thinking, or explanation."
    
    # Prepare JSON data
    data = {
        "model": model,
        "prompt": full_prompt,
        "system": system_message,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10000,  # Very large to allow full thinking process
            "num_ctx": 16384,
            "repeat_penalty": 1.0,
            "stop": ["</think>"]  # Only stop at the end of thinking
        }
    }
    
    json_data = json_module.dumps(data)
    
    # Create temp file
    local_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    local_temp.write(json_data)
    local_temp.close()
    
    remote_temp = f"/tmp/ollama_test_{uuid.uuid4().hex}.json"
    
    # SSH command with increased visibility
    ssh_command = f"""
    scp {local_temp.name} h100:{remote_temp} && \
    ssh h100 'curl -s -X POST {ollama_url} -H "Content-Type: application/json" -d @{remote_temp} --max-time 1800 -w "\\nHTTP_STATUS:%{{http_code}}\\n"; rm -f {remote_temp}' && \
    rm -f {local_temp.name}
    """
    
    print(f"Sending request to Ollama (model: {model})...")
    print(f"Issue: {issue['title'][:80]}...")
    print("-" * 80)
    
    try:
        # Execute SSH command
        result = subprocess.run(
            ssh_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1820
        )
        
        if result.returncode != 0:
            print(f"ERROR: SSH/curl command failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Check for HTTP status
        response_body = result.stdout
        if "HTTP_STATUS:" in result.stdout:
            parts = result.stdout.split("HTTP_STATUS:")
            status_code = parts[-1].strip()
            response_body = parts[0].strip()
            print(f"HTTP Status: {status_code}")
            
            if status_code != "200":
                print(f"ERROR: HTTP {status_code}")
                print(f"Response: {response_body[:500]}")
                return None
        
        # Parse JSON response
        try:
            json_response = json_module.loads(response_body)
        except json_module.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON")
            print(f"Raw response: {response_body[:1000]}")
            return None
        
        # Get the full response text
        text = json_response.get('response', '').strip()
        
        # Save the full response to file FIRST
        response_file = f"/tmp/ollama_full_response_{uuid.uuid4().hex}.txt"
        with open(response_file, 'w') as f:
            f.write(text)
        print(f"Full response saved to: {response_file}")
        
        # Also save the full JSON response
        json_file = f"/tmp/ollama_json_response_{uuid.uuid4().hex}.json"
        with open(json_file, 'w') as f:
            json.dump(json_response, f, indent=2)
        print(f"Full JSON saved to: {json_file}")
        
        # Print the FULL response for debugging (in chunks to avoid terminal issues)
        print("=" * 80)
        print("FULL OLLAMA RESPONSE:")
        print("=" * 80)
        
        # Print in chunks of 5000 characters to avoid any terminal truncation
        chunk_size = 5000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            print(f"[Chunk {i//chunk_size + 1}, chars {i}-{min(i+chunk_size, len(text))}]")
            print(chunk)
            if i + chunk_size < len(text):
                print("... [continuing] ...")
        
        print("=" * 80)
        print(f"Response length: {len(text)} characters")
        print(f"Files saved:")
        print(f"  - Text: {response_file}")
        print(f"  - JSON: {json_file}")
        print("=" * 80)
        
        # Analyze the response structure
        print("\nRESPONSE ANALYSIS:")
        print(f"- Contains <think>: {'<think>' in text}")
        print(f"- Contains </think>: {'</think>' in text}")
        print(f"- Contains <answer>: {'<answer>' in text}")
        print(f"- Contains pattern (1.x, 2.x...): {bool(parse_llm_output(text).is_ok() if parse_llm_output else False)}")
        
        if '<think>' in text and '</think>' in text:
            think_start = text.find('<think>')
            think_end = text.find('</think>')
            print(f"- Think tag position: start={think_start}, end={think_end}")
            print(f"- Content after </think>: '{text[think_end+8:].strip()[:100]}'")
        
        # Try to parse the response
        print("\nPARSING ATTEMPT:")
        result = parse_llm_output(text)
        if result.is_ok():
            categorization = result.unwrap()
            print("✅ Successfully parsed categorization:")
            for i, cat in enumerate(categorization):
                print(f"   {i+1}. {cat.value if hasattr(cat, 'value') else cat}")
        else:
            print(f"❌ Failed to parse: {result.unwrap_err()}")
            
            # Try manual extraction
            import re
            pattern = r'([1-5]\.[a-k]),\s*([1-5]\.[a-k]),\s*([1-5]\.[a-k]),\s*([1-5]\.[a-k]),\s*([1-5]\.[a-k])'
            matches = list(re.finditer(pattern, text))
            if matches:
                print(f"\nFound {len(matches)} pattern matches in response:")
                for i, match in enumerate(matches):
                    print(f"  Match {i+1}: {match.group()}")
                    print(f"    Position: {match.start()}-{match.end()}")
            
            # Check the last 100 characters to see if response was cut off
            print("\nLast 100 characters of response:")
            print(repr(text[-100:]))
            
            # Check if response seems truncated
            if text.endswith('\\') or text.endswith('...') or not text[-1] in '.!?\n ':
                print("\n⚠️  Response may be truncated!")
        
        # Return both the text and the file paths for further inspection
        return {
            'text': text,
            'response_file': response_file,
            'json_file': json_file
        }
        
    except subprocess.TimeoutExpired:
        print("ERROR: Request timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(local_temp.name)
        except:
            pass

# Main test execution
print("=" * 80)
print("OLLAMA INTEGRATION TEST WITH FULL RESPONSE")
print("=" * 80)

# Load issues
issue_groups = load_framework_issues()
if not issue_groups or not issue_groups[0]:
    print("No issues found to test with")
    sys.exit(1)

# Take the first issue
test_issue = issue_groups[0][0]

# Fetch comments for more complete testing
print("Fetching issue comments...")
comments = fetch_issue_comments(test_issue['html_url'])
test_issue['comments_data'] = comments
print(f"Found {len(comments)} comments")

# Run the test
result = test_ollama_with_full_response(test_issue)

if result:
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    if isinstance(result, dict):
        print("\nFILES SAVED FOR INSPECTION:")
        print(f"  Text response: {result['response_file']}")
        print(f"  JSON response: {result['json_file']}")
        print("\nTo view the full response, run:")
        print(f"  cat {result['response_file']}")
        print("\nTo check if truncated, run:")
        print(f"  tail -c 500 {result['response_file']}")
else:
    print("\n" + "=" * 80)
    print("TEST FAILED")
    print("=" * 80)
    sys.exit(1)