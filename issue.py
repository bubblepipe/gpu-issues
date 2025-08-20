"""Issue and PullRequest data classes for GitHub issue management"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import requests
import os
import time
import urllib3

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
    comments_data: List[Dict] = field(default_factory=list)
    timeline_url: Optional[str] = None
    
    # New fields from timeline
    mentioned_issues: List[Dict] = field(default_factory=list)  # Issues referenced
    mentioned_prs: List[PullRequest] = field(default_factory=list)  # PRs referenced
    closed_by_pr: Optional[PullRequest] = None  # PR that closed this issue
    assignees_from_timeline: List[str] = field(default_factory=list)  # Self-assigned users
    
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
            timeline_url=json_data.get('timeline_url')
        )
        
        # Store comments if they exist in the JSON
        if 'comments_data' in json_data:
            issue.comments_data = json_data['comments_data']
        
        if fetch_timeline and issue.timeline_url:
            issue.fetch_timeline()
        
        return issue