"""Plotting functions for visualizing bug categorization distributions"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from results_loader import load_categorized_results
from cates import IsReallyBug, UserPerspective, DeveloperPerspective, AcceleratorSpecific


def plot_bug_distributions(categorized_issues, save_path=None):
    """
    Create bar plots showing the distribution of all four categorization dimensions.
    
    Args:
        categorized_issues: List of tuples (title, url, is_really_bug, user_perspective, developer_perspective, accelerator_specific)
        save_path: Optional path to save the figure
    """
    # Extract the categorizations
    is_really_bug = [issue[2] for issue in categorized_issues if issue[2] is not None]
    user_perspective = [issue[3] for issue in categorized_issues if issue[3] is not None]
    developer_perspective = [issue[4] for issue in categorized_issues if issue[4] is not None]
    accelerator_specific = [issue[5] for issue in categorized_issues if issue[5] is not None]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of GPU Bug Categorizations', fontsize=16)
    
    # Plot Is Really Bug
    bug_counts = Counter(is_really_bug)
    bug_labels = [bt.name.replace('_', ' ').title() for bt in bug_counts.keys()]
    bug_values = list(bug_counts.values())
    
    ax1.bar(bug_labels, bug_values, color='skyblue', edgecolor='black')
    ax1.set_title('Is Really Bug Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(bug_values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot User Perspective
    user_counts = Counter(user_perspective)
    user_labels = [up.name.replace('_', ' ').title() for up in user_counts.keys()]
    user_values = list(user_counts.values())
    
    ax2.bar(user_labels, user_values, color='lightcoral', edgecolor='black')
    ax2.set_title('User Perspective Distribution')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(user_values):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot Developer Perspective
    dev_counts = Counter(developer_perspective)
    dev_labels = [dp.name.replace('_', ' ').title() for dp in dev_counts.keys()]
    dev_values = list(dev_counts.values())
    
    ax3.bar(dev_labels, dev_values, color='lightgreen', edgecolor='black')
    ax3.set_title('Developer Perspective Distribution')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(dev_values):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot Accelerator Specific
    accel_counts = Counter(accelerator_specific)
    accel_labels = [ac.name.replace('_', ' ').title() for ac in accel_counts.keys()]
    accel_values = list(accel_counts.values())
    
    ax4.bar(accel_labels, accel_values, color='gold', edgecolor='black')
    ax4.set_title('Accelerator Specific Distribution')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(accel_values):
        ax4.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_combined_heatmap(categorized_issues, save_path=None):
    """
    Create a heatmap showing the co-occurrence of user perspective and developer perspective.
    
    Args:
        categorized_issues: List of tuples (title, url, is_really_bug, user_perspective, developer_perspective, accelerator_specific)
        save_path: Optional path to save the figure
    """
    # Create co-occurrence matrix
    user_dev_pairs = [(issue[3], issue[4]) for issue in categorized_issues 
                      if issue[3] is not None and issue[4] is not None]
    
    # Get unique perspectives
    unique_user = sorted(set(pair[0] for pair in user_dev_pairs))
    unique_dev = sorted(set(pair[1] for pair in user_dev_pairs))
    
    # Create matrix
    matrix = np.zeros((len(unique_user), len(unique_dev)))
    
    for user_persp, dev_persp in user_dev_pairs:
        i = unique_user.index(user_persp)
        j = unique_dev.index(dev_persp)
        matrix[i, j] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(unique_dev)))
    ax.set_yticks(np.arange(len(unique_user)))
    ax.set_xticklabels([d.name.replace('_', ' ').title() for d in unique_dev])
    ax.set_yticklabels([u.name.replace('_', ' ').title() for u in unique_user])
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(unique_user)):
        for j in range(len(unique_dev)):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white")
    
    ax.set_title("Co-occurrence of User Perspective and Developer Perspective")
    ax.set_xlabel("Developer Perspective")
    ax.set_ylabel("User Perspective")
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    
    return fig


def print_statistics(categorized_issues):
    """Print summary statistics about the categorized issues."""
    total = len(categorized_issues)
    print(f"\nTotal categorized issues: {total}")
    
    # Count by is_really_bug
    bug_counts = Counter(issue[2] for issue in categorized_issues if issue[2] is not None)
    print("\nIs Really Bug Distribution:")
    for bug_type, count in bug_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {bug_type.name}: {count} ({percentage:.1f}%)")
    
    # Count by user perspective
    user_counts = Counter(issue[3] for issue in categorized_issues if issue[3] is not None)
    print("\nUser Perspective Distribution:")
    for user_persp, count in user_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {user_persp.name}: {count} ({percentage:.1f}%)")
    
    # Count by developer perspective
    dev_counts = Counter(issue[4] for issue in categorized_issues if issue[4] is not None)
    print("\nDeveloper Perspective Distribution:")
    for dev_persp, count in dev_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {dev_persp.name}: {count} ({percentage:.1f}%)")
    
    # Count by accelerator specific
    accel_counts = Counter(issue[5] for issue in categorized_issues if issue[5] is not None)
    print("\nAccelerator Specific Distribution:")
    for accel, count in accel_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {accel.name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Load categorized issues from JSON files
    categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/categorized-gemini-2.5-flash/categorized_issues_*.json')
    
    if categorized_issues:
        # Print statistics
        print_statistics(categorized_issues)
        
        # Create plots
        plot_bug_distributions(categorized_issues, save_path="bug_distributions.png")
        # plot_combined_heatmap(categorized_issues, save_path="bug_heatmap.png")
    else:
        print("No categorized issues found.")