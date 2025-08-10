"""Functions for loading categorized results from files"""

import glob
import json
import sys
from cates import (
    IsReallyBug, UserPerspective, DeveloperPerspective, AcceleratorSpecific, UserExpertise,
    IS_REALLY_BUG_LOOKUP, USER_PERSPECTIVE_LOOKUP, 
    DEVELOPER_PERSPECTIVE_LOOKUP, ACCELERATOR_SPECIFIC_LOOKUP, USER_EXPERTISE_LOOKUP
)


def load_categorized_results(pattern):
    """
    Load categorized issues from JSON result files.
    
    Args:
        pattern: Glob pattern for finding JSON result files (e.g., 'categorized_issues_*.json')
        
    Returns:
        List of tuples containing (title, url, is_really_bug, user_perspective, developer_perspective, accelerator_specific, user_expertise)
    """
    categorized_issues = []
    result_files = glob.glob(pattern)
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                json_data = json.load(f)
                
            # Convert JSON data back to tuples with enum objects
            for item in json_data:
                # Parse the enum values from their string representations
                is_really_bug = None
                user_perspective = None
                developer_perspective = None
                accelerator_specific = None
                user_expertise = None
                
                # Find the enum objects by matching their value strings
                if item.get('is_really_bug'):
                    for code, enum_obj in IS_REALLY_BUG_LOOKUP.items():
                        if enum_obj.value == item['is_really_bug']:
                            is_really_bug = enum_obj
                            break
                
                if item.get('user_perspective'):
                    for code, enum_obj in USER_PERSPECTIVE_LOOKUP.items():
                        if enum_obj.value == item['user_perspective']:
                            user_perspective = enum_obj
                            break
                
                if item.get('developer_perspective'):
                    for code, enum_obj in DEVELOPER_PERSPECTIVE_LOOKUP.items():
                        if enum_obj.value == item['developer_perspective']:
                            developer_perspective = enum_obj
                            break
                
                if item.get('accelerator_specific'):
                    for code, enum_obj in ACCELERATOR_SPECIFIC_LOOKUP.items():
                        if enum_obj.value == item['accelerator_specific']:
                            accelerator_specific = enum_obj
                            break
                
                if item.get('user_expertise'):
                    for code, enum_obj in USER_EXPERTISE_LOOKUP.items():
                        if enum_obj.value == item['user_expertise']:
                            user_expertise = enum_obj
                            break
                
                # Create tuple in the expected format
                categorized_issues.append((
                    item['title'],
                    item['url'],
                    is_really_bug,
                    user_perspective,
                    developer_perspective,
                    accelerator_specific,
                    user_expertise
                ))
                
        except FileNotFoundError:
            print(f"File not found: {file}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from file {file}: {e}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
                    
    return categorized_issues


def get_categorized_urls(categorized_issues):
    """
    Extract URLs from categorized issues for fast lookup.
    
    Args:
        categorized_issues: List of categorized issue tuples
        
    Returns:
        Set of URLs that have been categorized
    """
    return {issue[1] for issue in categorized_issues}


def load_categorized_json_files(pattern):
    """
    Load categorized issues directly as dictionaries from JSON files.
    
    Args:
        pattern: Glob pattern for finding JSON result files
        
    Returns:
        List of dictionaries with keys: title, url, is_really_bug, user_perspective, 
        developer_perspective, accelerator_specific, user_expertise
    """
    all_issues = []
    result_files = glob.glob(pattern)
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                json_data = json.load(f)
                all_issues.extend(json_data)
        except FileNotFoundError:
            print(f"File not found: {file}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from file {file}: {e}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    return all_issues