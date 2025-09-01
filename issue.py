"""Issue and PullRequest data classes for GitHub issue management"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import requests
import os
import time
import urllib3
import re

# Suppress SSL warnings when we fallback to unverified requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request"""
    number: int
    title: str
    html_url: str
    state: str
    merged: bool = False
    
    @classmethod
    def from_timeline_event(cls, event_data: dict):
        """Create PullRequest from timeline event data"""
        source = event_data.get('source', {})
        issue = source.get('issue', {})
        
        # For cross-reference events, the PR info is in source.issue
        if issue and 'pull_request' in issue:
            return cls(
                number=issue.get('number', 0),
                title=issue.get('title', ''),
                html_url=issue.get('html_url', ''),
                state=issue.get('state', 'unknown'),
                merged=issue.get('pull_request', {}).get('merged_at') is not None
            )
        return None


@dataclass
class Issue:
    """Represents a GitHub Issue with timeline data"""
    # Core fields from JSON
    number: int
    title: str
    html_url: str
    body: Optional[str]
    labels: List[Dict]
    state: str
    created_at: str
    closed_at: Optional[str]
    user: Dict  # author info
    
    # Additional fields
    comments_count: int = 0  # Number of comments/discussions
    comments_data: List[Dict] = field(default_factory=list)
    timeline_url: Optional[str] = None
    
    # New fields from timeline
    mentioned_issues: List[Dict] = field(default_factory=list)  # Issues referenced
    mentioned_prs: List[PullRequest] = field(default_factory=list)  # PRs referenced
    closed_by_pr: Optional[PullRequest] = None  # PR that closed this issue
    assignees_from_timeline: List[str] = field(default_factory=list)  # Self-assigned users
    
    # Comprehensive tracking of all mentioned URLs
    all_mentioned_urls: set = field(default_factory=set)  # All unique URLs mentioned
    mentioned_urls_by_source: Dict[str, List[str]] = field(default_factory=dict)  # URLs grouped by source
    content_mentioned_issues: List[str] = field(default_factory=list)  # Issues from body/comments
    content_mentioned_prs: List[str] = field(default_factory=list)  # PRs from body/comments
    
    def fetch_timeline(self):
        """Fetch and parse timeline data to populate mentioned_issues and mentioned_prs"""
        if not self.timeline_url:
            return
        
        # Note: Intentionally NOT using GitHub token for timeline API
        # The timeline API behaves differently with authentication - it excludes 'cross-referenced' events
        # when a token is provided, which causes us to miss important related issues/PRs
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        # Retry logic for SSL and connection errors
        max_retries = 3
        retry_delay = 1  # Starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging connections
                response = requests.get(self.timeline_url, headers=headers, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 403:
                    print(f"Rate limited when fetching timeline for issue #{self.number}. Waiting...")
                    time.sleep(60)
                    return
                
                if response.status_code == 200:
                    timeline = response.json()
                    self._parse_timeline(timeline)
                    # Also fetch connected PRs via GraphQL
                    self.fetch_connected_prs_graphql()
                    # Parse body for issue/PR references
                    self.parse_body_references()
                    return  # Success, exit the retry loop
                    
            except requests.exceptions.SSLError as e:
                if attempt < max_retries - 1:
                    print(f"SSL error fetching timeline for issue #{self.number}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt - try with verify=False as fallback
                    try:
                        print(f"SSL verification failed for issue #{self.number}, trying without verification...")
                        response = requests.get(self.timeline_url, headers=headers, timeout=30, verify=False)
                        if response.status_code == 200:
                            timeline = response.json()
                            self._parse_timeline(timeline)
                            # Also fetch connected PRs via GraphQL
                            self.fetch_connected_prs_graphql()
                            # Parse body for issue/PR references
                            self.parse_body_references()
                            return
                    except Exception as fallback_error:
                        print(f"Failed to fetch timeline for issue #{self.number} even without SSL verification: {fallback_error}")
                        return
                        
            except requests.exceptions.Timeout:
                print(f"Timeout fetching timeline for issue #{self.number} (attempt {attempt + 1}/{max_retries})")
                if attempt >= max_retries - 1:
                    return
                time.sleep(retry_delay)
                retry_delay *= 2
                
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error fetching timeline for issue #{self.number}: {e}")
                if attempt >= max_retries - 1:
                    return
                time.sleep(retry_delay)
                retry_delay *= 2
                
            except Exception as e:
                print(f"Unexpected error fetching timeline for issue #{self.number}: {e}")
                return  # Don't retry on unexpected errors
    
    def parse_body_references(self):
        """Parse issue body and comments for #number references to other issues/PRs."""
        if not self.body:
            return
        
        # Find all #number patterns in the body
        # Exclude common false positives like #define, #include
        pattern = r'(?<![\w#])#(\d+)\b'
        
        # Extract owner and repo from html_url
        parts = self.html_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return
        
        owner = parts[0]
        repo = parts[1]
        
        # Find all matches in body
        matches = re.findall(pattern, self.body)
        
        # Also search in comments if they exist
        if self.comments_data:
            for comment in self.comments_data:
                comment_body = comment.get('body', '')
                matches.extend(re.findall(pattern, comment_body))
        
        # Remove duplicates and self-references
        unique_numbers = set(int(m) for m in matches if int(m) != self.number)
        
        # For each reference, check if it's an issue or PR (limited to avoid too many API calls)
        github_token = os.getenv('GITHUB_TOKEN')
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        for number in list(unique_numbers)[:10]:  # Limit to 10 to avoid too many API calls
            try:
                # First try as issue
                issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
                response = requests.get(issue_url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    # Check if it's a PR or issue
                    if 'pull_request' in data:
                        # It's a PR
                        pr = PullRequest(
                            number=data.get('number', 0),
                            title=data.get('title', ''),
                            html_url=data.get('html_url', ''),
                            state=data.get('state', 'unknown'),
                            merged=data.get('pull_request', {}).get('merged_at') is not None
                        )
                        if pr not in self.mentioned_prs:
                            self.mentioned_prs.append(pr)
                    else:
                        # It's an issue
                        issue_info = {
                            'number': data.get('number'),
                            'title': data.get('title'),
                            'url': data.get('html_url'),
                            'state': data.get('state')
                        }
                        if issue_info not in self.mentioned_issues:
                            self.mentioned_issues.append(issue_info)
            except:
                # Silently skip if we can't fetch the reference
                pass
    
    def fetch_connected_prs_graphql(self):
        """Fetch PRs connected via GitHub's 'linked to close' feature using GraphQL API."""
        # Extract owner and repo from html_url
        # Example: https://github.com/pytorch/pytorch/issues/93372
        parts = self.html_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return
        
        owner = parts[0]
        repo = parts[1]
        
        # GraphQL query to get connected PRs
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              timelineItems(first: 100, itemTypes: [CONNECTED_EVENT, DISCONNECTED_EVENT]) {
                nodes {
                  __typename
                  ... on ConnectedEvent {
                    subject {
                      __typename
                      ... on PullRequest {
                        number
                        title
                        url
                        state
                        merged
                      }
                    }
                  }
                  ... on DisconnectedEvent {
                    subject {
                      __typename
                      ... on PullRequest {
                        number
                        title
                        url
                        state
                        merged
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            # Can't use GraphQL without token
            return
        
        headers = {
            'Authorization': f'Bearer {github_token}',
            'Content-Type': 'application/json'
        }
        
        variables = {
            'owner': owner,
            'repo': repo,
            'number': self.number
        }
        
        try:
            response = requests.post(
                'https://api.github.com/graphql',
                json={'query': query, 'variables': variables},
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    repository = data['data'].get('repository', {})
                    issue = repository.get('issue', {})
                    timeline_items = issue.get('timelineItems', {})
                    nodes = timeline_items.get('nodes', [])
                    
                    for node in nodes:
                        if node and 'subject' in node:
                            subject = node['subject']
                            if subject and subject.get('__typename') == 'PullRequest':
                                # Create PullRequest object
                                pr = PullRequest(
                                    number=subject.get('number', 0),
                                    title=subject.get('title', ''),
                                    html_url=subject.get('url', ''),
                                    state=subject.get('state', 'unknown').lower(),
                                    merged=subject.get('merged', False)
                                )
                                # Add if not already in list
                                if pr not in self.mentioned_prs:
                                    self.mentioned_prs.append(pr)
                                    
        except Exception as e:
            # Silently fail - GraphQL is supplementary
            pass
    
    def _parse_timeline(self, timeline_events):
        """Parse timeline events to extract mentioned issues/PRs"""
        for event in timeline_events:
            event_type = event.get('event')
            
            if event_type == 'cross-referenced':
                # Extract mentioned issues/PRs
                source = event.get('source', {})
                issue = source.get('issue', {})
                
                if issue:
                    # Check if it's a PR or issue
                    if 'pull_request' in issue:
                        pr = PullRequest.from_timeline_event(event)
                        if pr and pr not in self.mentioned_prs:
                            self.mentioned_prs.append(pr)
                    else:
                        # It's an issue
                        issue_info = {
                            'number': issue.get('number'),
                            'title': issue.get('title'),
                            'url': issue.get('html_url'),
                            'state': issue.get('state')
                        }
                        if issue_info not in self.mentioned_issues:
                            self.mentioned_issues.append(issue_info)
            
            elif event_type == 'referenced':
                # Parse 'referenced' events which often contain commit references
                # These events have 'commit_id' and 'commit_url' fields
                # GitHub automatically creates 'referenced' events when commits mention issues
                # For now, we just note these exist but don't extract issue/PR info from them
                # as that would require additional API calls to fetch commit details
                pass
            
            elif event_type == 'closed':
                # Check if closed by PR
                commit_id = event.get('commit_id')
                if commit_id:
                    # Try to find the PR that introduced this commit
                    # This requires additional API calls, so we'll skip for now
                    pass
            
            elif event_type == 'assigned' or event_type == 'self-assigned':
                # Track assignees
                assignee = event.get('assignee', {})
                login = assignee.get('login')
                if login and login not in self.assignees_from_timeline:
                    self.assignees_from_timeline.append(login)
    
    def collect_all_mentions(self):
        """Collect all mentioned issues/PRs from all sources and deduplicate."""
        # Import here to avoid circular dependency
        from cate import extract_github_urls_from_text
        
        # Clear existing collections
        self.all_mentioned_urls.clear()
        self.mentioned_urls_by_source.clear()
        self.content_mentioned_issues.clear()
        self.content_mentioned_prs.clear()
        
        # Get base repo for resolving shorthand references
        base_repo_match = re.match(r'(https://github\.com/[\w\-]+/[\w\-]+)', self.html_url)
        base_repo_url = base_repo_match.group(1) if base_repo_match else None
        
        # 1. Extract from content (body + comments)
        content = self.to_string_pretty()
        content_urls = extract_github_urls_from_text(content, base_repo_url)
        
        for url in content_urls:
            if url != self.html_url:  # Skip self-reference
                self.all_mentioned_urls.add(url)
                self.mentioned_urls_by_source.setdefault('CONTENT', []).append(url)
                
                # Classify as issue or PR
                if '/pull/' in url:
                    self.content_mentioned_prs.append(url)
                else:
                    self.content_mentioned_issues.append(url)
        
        # 2. Add from timeline PRs
        for pr in self.mentioned_prs:
            self.all_mentioned_urls.add(pr.html_url)
            self.mentioned_urls_by_source.setdefault('TIMELINE_PR', []).append(pr.html_url)
        
        # 3. Add from timeline issues
        for issue_dict in self.mentioned_issues:
            if issue_dict.get('url'):
                url = issue_dict['url']
                self.all_mentioned_urls.add(url)
                self.mentioned_urls_by_source.setdefault('TIMELINE_ISSUE', []).append(url)
    
    def get_mention_stats(self):
        """Get statistics about mentioned issues/PRs."""
        return {
            'total_unique': len(self.all_mentioned_urls),
            'from_content': len(self.mentioned_urls_by_source.get('CONTENT', [])),
            'from_timeline_pr': len(self.mentioned_urls_by_source.get('TIMELINE_PR', [])),
            'from_timeline_issue': len(self.mentioned_urls_by_source.get('TIMELINE_ISSUE', [])),
            'content_prs': len(self.content_mentioned_prs),
            'content_issues': len(self.content_mentioned_issues),
            'has_duplicates': self.has_duplicate_mentions()
        }
    
    def has_duplicate_mentions(self):
        """Check if any URL appears in multiple sources."""
        all_urls_list = []
        for urls in self.mentioned_urls_by_source.values():
            all_urls_list.extend(urls)
        return len(all_urls_list) != len(set(all_urls_list))
    
    def print_mention_report(self):
        """Print a detailed report of all mentioned issues/PRs."""
        stats = self.get_mention_stats()
        print(f"\n=== Mention Report for {self.html_url} ===")
        print(f"Total unique URLs: {stats['total_unique']}")
        print(f"  From content: {stats['from_content']}")
        print(f"  From timeline PRs: {stats['from_timeline_pr']}")
        print(f"  From timeline issues: {stats['from_timeline_issue']}")
        print(f"  Content PRs: {stats['content_prs']}")
        print(f"  Content issues: {stats['content_issues']}")
        print(f"  Has duplicates: {stats['has_duplicates']}")
        
        if self.all_mentioned_urls:
            print("\nAll mentioned URLs:")
            for url in sorted(self.all_mentioned_urls):
                sources = []
                for source, urls in self.mentioned_urls_by_source.items():
                    if url in urls:
                        sources.append(source)
                print(f"  - {url} (from: {', '.join(sources)})")
    
    def to_string_pretty(self) -> str:
        """Format issue for display"""
        content = f"Title: {self.title}\n"
        content += f"URL: {self.html_url}\n"
        
        # Add labels
        if self.labels:
            label_names = [label['name'] for label in self.labels]
            content += f"Labels: {', '.join(label_names)}\n"
        
        # Add assignees if available
        if self.assignees_from_timeline:
            content += f"Assignees: {', '.join(self.assignees_from_timeline)}\n"
        
        # Add mentioned PRs/Issues if available
        if self.mentioned_prs:
            content += f"\nRelated Pull Requests:\n"
            for pr in self.mentioned_prs:
                status = f"merged" if pr.merged else pr.state
                content += f"  - PR #{pr.number}: {pr.title} ({status})\n"
        
        if self.mentioned_issues:
            content += f"\nRelated Issues:\n"
            for issue in self.mentioned_issues:
                content += f"  - Issue #{issue['number']}: {issue['title']} ({issue['state']})\n"
        
        content += "\nIssue Description:\n"
        if self.body:
            content += self.body + "\n"
        else:
            content += "(No description provided)\n"
        
        # Add comments
        if self.comments_data:
            content += f"\n--- Comments ({len(self.comments_data)}) ---\n"
            for i, comment in enumerate(self.comments_data, 1):
                content += f"\nComment {i} by {comment.get('user', {}).get('login', 'Unknown')} at {comment.get('created_at', 'Unknown')}:\n"
                if comment.get('body'):
                    content += comment['body'] + "\n"
                else:
                    content += "(Empty comment)\n"
        
        return content
    
    @classmethod
    def from_json(cls, json_data: dict, fetch_timeline: bool = False):
        """Create Issue from JSON dict and optionally fetch timeline"""
        issue = cls(
            number=json_data.get('number', 0),
            title=json_data.get('title', ''),
            html_url=json_data.get('html_url', ''),
            body=json_data.get('body'),
            labels=json_data.get('labels', []),
            state=json_data.get('state', 'unknown'),
            created_at=json_data.get('created_at', ''),
            closed_at=json_data.get('closed_at'),
            user=json_data.get('user', {}),
            comments_count=json_data.get('comments', 0),  # Extract comments count
            timeline_url=json_data.get('timeline_url')
        )
        
        # Store comments if they exist in the JSON
        if 'comments_data' in json_data:
            issue.comments_data = json_data['comments_data']
        
        if fetch_timeline and issue.timeline_url:
            issue.fetch_timeline()
        
        return issue