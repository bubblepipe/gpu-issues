"""Plotting functions for visualizing bug categorization distributions"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from results_loader import load_categorized_results
from cates import BugType, BugSymptom, BugHeterogeneity


def plot_bug_distributions(categorized_issues, save_path=None):
    """
    Create bar plots showing the distribution of bug types, symptoms, and heterogeneity.
    
    Args:
        categorized_issues: List of tuples (title, url, bug_type, bug_symptom, bug_heterogeneity)
        save_path: Optional path to save the figure
    """
    # Extract the categorizations
    bug_types = [issue[2] for issue in categorized_issues if issue[2] is not None]
    bug_symptoms = [issue[3] for issue in categorized_issues if issue[3] is not None]
    bug_heterogeneity = [issue[4] for issue in categorized_issues if issue[4] is not None]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Distribution of GPU Bug Categorizations', fontsize=16)
    
    # Plot Bug Types
    type_counts = Counter(bug_types)
    type_labels = [bt.name.replace('_', ' ').title() for bt in type_counts.keys()]
    type_values = list(type_counts.values())
    
    ax1.bar(type_labels, type_values, color='skyblue', edgecolor='black')
    ax1.set_title('Bug Type Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(type_values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot Bug Symptoms
    symptom_counts = Counter(bug_symptoms)
    symptom_labels = [bs.name.replace('_', ' ').title() for bs in symptom_counts.keys()]
    symptom_values = list(symptom_counts.values())
    
    ax2.bar(symptom_labels, symptom_values, color='lightcoral', edgecolor='black')
    ax2.set_title('Bug Symptom Distribution')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(symptom_values):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot Bug Heterogeneity
    hetero_counts = Counter(bug_heterogeneity)
    hetero_labels = [bh.name.replace('_', ' ').title() for bh in hetero_counts.keys()]
    hetero_values = list(hetero_counts.values())
    
    ax3.bar(hetero_labels, hetero_values, color='lightgreen', edgecolor='black')
    ax3.set_title('Bug Heterogeneity Distribution')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(hetero_values):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
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
    Create a heatmap showing the co-occurrence of bug types and symptoms.
    
    Args:
        categorized_issues: List of tuples (title, url, bug_type, bug_symptom, bug_heterogeneity)
        save_path: Optional path to save the figure
    """
    # Create co-occurrence matrix
    type_symptom_pairs = [(issue[2], issue[3]) for issue in categorized_issues 
                          if issue[2] is not None and issue[3] is not None]
    
    # Get unique types and symptoms
    unique_types = sorted(set(pair[0] for pair in type_symptom_pairs))
    unique_symptoms = sorted(set(pair[1] for pair in type_symptom_pairs))
    
    # Create matrix
    matrix = np.zeros((len(unique_types), len(unique_symptoms)))
    
    for bug_type, bug_symptom in type_symptom_pairs:
        i = unique_types.index(bug_type)
        j = unique_symptoms.index(bug_symptom)
        matrix[i, j] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(unique_symptoms)))
    ax.set_yticks(np.arange(len(unique_types)))
    ax.set_xticklabels([s.name.replace('_', ' ').title() for s in unique_symptoms])
    ax.set_yticklabels([t.name.replace('_', ' ').title() for t in unique_types])
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(unique_types)):
        for j in range(len(unique_symptoms)):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white")
    
    ax.set_title("Co-occurrence of Bug Types and Symptoms")
    ax.set_xlabel("Bug Symptom")
    ax.set_ylabel("Bug Type")
    
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
    
    # Count by type
    type_counts = Counter(issue[2] for issue in categorized_issues if issue[2] is not None)
    print("\nBug Type Distribution:")
    for bug_type, count in type_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {bug_type.name}: {count} ({percentage:.1f}%)")
    
    # Count by symptom
    symptom_counts = Counter(issue[3] for issue in categorized_issues if issue[3] is not None)
    print("\nBug Symptom Distribution:")
    for symptom, count in symptom_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {symptom.name}: {count} ({percentage:.1f}%)")
    
    # Count by heterogeneity
    hetero_counts = Counter(issue[4] for issue in categorized_issues if issue[4] is not None)
    print("\nBug Heterogeneity Distribution:")
    for hetero, count in hetero_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {hetero.name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Load categorized issues
    categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/llm_categorizations_a81e1dd0/results.tuples.*')
    
    if categorized_issues:
        # Print statistics
        print_statistics(categorized_issues)
        
        # Create plots
        plot_bug_distributions(categorized_issues, save_path="bug_distributions.png")
        # plot_combined_heatmap(categorized_issues, save_path="bug_heatmap.png")
    else:
        print("No categorized issues found.")