"""Plotting functions for visualizing bug categorization distributions"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from results_loader import load_categorized_results, load_categorized_json_files
from cates import IsReallyBug, UserPerspective, DeveloperPerspective, AcceleratorSpecific, UserExpertise

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def get_framework_from_url(url):
    """Extract framework name from GitHub URL."""
    if 'pytorch/pytorch' in url:
        return 'PyTorch'
    elif 'tensorflow/tensorflow' in url:
        return 'TensorFlow'
    elif 'jax-ml/jax' in url:
        return 'JAX'
    elif 'NVIDIA/TensorRT' in url:
        return 'TensorRT'
    elif 'triton-lang/triton' in url:
        return 'Triton'
    else:
        return 'Unknown'


def plot_platform_distributions(categorized_issues, title="", ax=None):
    """
    Plot categorization distributions for a set of issues on a given axis.
    
    Args:
        categorized_issues: List of tuples with categorization data
        title: Title for the subplot
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract categorizations
    is_really_bug = [issue[2] for issue in categorized_issues if issue[2] is not None]
    user_perspective = [issue[3] for issue in categorized_issues if issue[3] is not None]
    developer_perspective = [issue[4] for issue in categorized_issues if issue[4] is not None]
    accelerator_specific = [issue[5] for issue in categorized_issues if issue[5] is not None]
    user_expertise = [issue[6] for issue in categorized_issues if issue[6] is not None]
    
    # Count each category
    categories = {
        'Is Bug': Counter(is_really_bug),
        'User View': Counter(user_perspective),
        'Dev View': Counter(developer_perspective),
        'Platform': Counter(accelerator_specific),
        'Expertise': Counter(user_expertise)
    }
    
    # Prepare data for grouped bar plot with subtitles
    category_info = {
        'Is Bug': 'Bug classification',
        'User View': "User's perspective",
        'Dev View': "Developer's approach",
        'Platform': 'Hardware specificity',
        'Expertise': 'User skill level'
    }
    category_names = list(categories.keys())
    
    # Calculate max items based on all possible enum values to ensure consistent bar width
    # Use the maximum number of enum values across all categories
    max_possible_items = max(
        len(list(IsReallyBug)),        # 5 items
        len(list(UserPerspective)),     # 11 items
        len(list(DeveloperPerspective)), # 9 items
        len(list(AcceleratorSpecific)), # 8 items
        len(list(UserExpertise))       # 4 items
    )  # This will be 11 (UserPerspective has the most)
    bar_width = min(0.15, 0.8 / max_possible_items)  # Dynamically adjust bar width
    x_pos = np.arange(len(category_names))
    
    # Define extended color palettes for each category type
    # Using colorblind-friendly and visually appealing colors with gradients
    category_palettes = {
        'Is Bug': ['#1e5f8e', '#2E86AB', '#5EB1BF', '#84D2F6', '#A4C3D2', '#CFE5E7'],  # Blues (6 colors for 5 options)
        'User View': ['#2d6a4f', '#40916c', '#52B788', '#74C69D', '#95D5B2', '#B7E4C7', '#D8F3DC', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d'],  # Greens (12 colors for 11 options)
        'Dev View': ['#d84a05', '#F77F00', '#F9A03F', '#FCBF49', '#FFD166', '#FFE5A5', '#ffe8c8', '#ffd4a3', '#ffc07e'],  # Oranges (9 colors for 9 options)
        'Platform': ['#a61e4d', '#D62828', '#F94144', '#F3722C', '#F8961E', '#F9C74F', '#ffd166', '#ffe169'],  # Reds to Yellows (8 colors for 8 options)
        'Expertise': ['#5b0e8c', '#7209B7', '#9D4EDD', '#B298DC', '#C77DFF']  # Purples (5 colors for 4 options)
    }
    
    # Track all unique values across categories for legend
    all_values = set()
    for cat_counts in categories.values():
        all_values.update(cat_counts.keys())
    
    # Plot bars for each unique value
    plotted_items = {}
    
    # Define all possible enum values for each category in their original order
    all_enums = {
        'Is Bug': list(IsReallyBug),
        'User View': list(UserPerspective),
        'Dev View': list(DeveloperPerspective),
        'Platform': list(AcceleratorSpecific),
        'Expertise': list(UserExpertise)
    }
    
    for i, (cat_name, cat_counts) in enumerate(categories.items()):
        # Get all enum values for this category in their defined order
        all_items = all_enums[cat_name]
        # Create list of (item, count) tuples, using 0 for missing items
        sorted_items = [(item, cat_counts.get(item, 0)) for item in all_items]
        
        # Get the color palette for this category
        palette = category_palettes[cat_name]
        
        # Calculate offset to center the bars
        num_items = len(sorted_items)
        start_offset = -(num_items - 1) * bar_width / 2
        
        for j, (item, count) in enumerate(sorted_items):
            label = item.name.replace('_', ' ').title()[:15]  # Truncate long labels
            # Extract the code (e.g., "1.a", "2.b") from the enum value
            code = item.value.split()[0]  # Get the first part before the space
            
            # Use modulo to cycle through colors if needed
            color = palette[j % len(palette)]
            bar_position = i + start_offset + j * bar_width
            ax.bar(bar_position, count, bar_width * 0.9,  # Slightly smaller to add gaps
                   label=label if label not in plotted_items else "",
                   color=color, edgecolor='#2D3436', linewidth=0.5, alpha=0.85)
            plotted_items[label] = True
            
            # Add labels on the bar
            # Extract the code (e.g., "1.a", "2.b") from the enum value
            code = item.value.split()[0]  # Get the first part before the space
            
            if count > 0:
                # Add code label on top (vertical, black)
                ax.text(bar_position, count + 1.5, code,
                       ha='center', va='bottom', fontsize=9, fontweight='bold', 
                       color='black', rotation=90)
                # Add count value below the code
                ax.text(bar_position, count + 0.2, str(count),
                       ha='center', va='bottom', fontsize=8, color='black')
            else:
                # For zero-count bars, show the code at the bottom with lighter color
                ax.text(bar_position, 1.5, code,
                       ha='center', va='bottom', fontsize=8, 
                       color='gray', rotation=90)
                ax.text(bar_position, 0.1, '0',
                       ha='center', va='bottom', fontsize=7, color='gray')
    
    ax.set_xlabel('Category', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Count', fontsize=11, fontweight='semibold')
    ax.set_title(f'{title} (n={len(categorized_issues)})', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_names, rotation=0, fontsize=10, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFBFC')
    
    # Don't show legend on individual plots (too crowded)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


def plot_all_platforms_distributions(categorized_issues, save_path="platform_distributions.png"):
    """
    Create a single figure with 6 subplots: one for each platform and one combined.
    
    Args:
        categorized_issues: List of all categorized issue tuples
        save_path: Path to save the combined figure
    """
    # Separate issues by platform
    platform_issues = defaultdict(list)
    
    for issue in categorized_issues:
        url = issue[1]  # URL is at index 1
        framework = get_framework_from_url(url)
        platform_issues[framework].append(issue)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Bug Categorization Distributions by Platform', fontsize=18, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('#F8F9FA')
    
    # Define platform order
    platforms = ['PyTorch', 'TensorFlow', 'JAX', 'TensorRT', 'Triton']
    
    # Plot individual platforms
    for idx, platform in enumerate(platforms):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        platform_data = platform_issues.get(platform, [])
        plot_platform_distributions(platform_data, title=platform, ax=ax)
    
    # Plot combined (all platforms)
    ax = axes[1, 2]
    plot_platform_distributions(categorized_issues, title="All Platforms Combined", ax=ax)
    
    # Ensure consistent y-axis limits across all subplots for uniform appearance
    # Find the maximum y value across all subplots
    max_y = 0
    for ax in axes.flat:
        if ax.get_ylim()[1] > max_y:
            max_y = ax.get_ylim()[1]
    
    # Apply the same y-axis limit to all subplots
    for ax in axes.flat:
        ax.set_ylim(0, max_y)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Platform distributions saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_bug_distributions(categorized_issues, save_path=None):
    """
    Create bar plots showing the distribution of all five categorization dimensions.
    
    Args:
        categorized_issues: List of tuples (title, url, is_really_bug, user_perspective, developer_perspective, accelerator_specific, user_expertise)
        save_path: Optional path to save the figure
    """
    # Extract the categorizations
    is_really_bug = [issue[2] for issue in categorized_issues if issue[2] is not None]
    user_perspective = [issue[3] for issue in categorized_issues if issue[3] is not None]
    developer_perspective = [issue[4] for issue in categorized_issues if issue[4] is not None]
    accelerator_specific = [issue[5] for issue in categorized_issues if issue[5] is not None]
    user_expertise = [issue[6] for issue in categorized_issues if issue[6] is not None]
    
    # Create subplots - now with 6 plots (2x3 grid)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Distribution of GPU Bug Categorizations', fontsize=18, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')
    
    # Plot Is Really Bug
    bug_counts = Counter(is_really_bug)
    # Get all enum values in order, not just the ones with data
    bug_items = list(IsReallyBug)
    bug_labels = [bt.name.replace('_', ' ').title() for bt in bug_items]
    bug_codes = [bt.value.split()[0] for bt in bug_items]  # Extract codes like "1.a"
    bug_values = [bug_counts.get(bt, 0) for bt in bug_items]  # Use 0 for missing values
    
    ax1.bar(bug_labels, bug_values, color='#5EB1BF', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax1.set_title('Is Really Bug Distribution', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_facecolor('#FAFBFC')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and code labels on bars
    for i, (v, code) in enumerate(zip(bug_values, bug_codes)):
        if v > 0:
            # Add code label on top (vertical, black)
            ax1.text(i, v + 1.5, code, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='black', rotation=90)
            # Add count value below the code
            ax1.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=8, color='black')
        else:
            # For zero-count bars, show the code at the bottom with lighter color
            ax1.text(i, 1.5, code, ha='center', va='bottom', 
                    fontsize=8, color='gray', rotation=90)
            ax1.text(i, 0.1, '0', ha='center', va='bottom', fontsize=7, color='gray')
    
    # Plot User Perspective
    user_counts = Counter(user_perspective)
    # Get all enum values in order, not just the ones with data
    user_items = list(UserPerspective)
    user_labels = [up.name.replace('_', ' ').title() for up in user_items]
    user_codes = [up.value.split()[0] for up in user_items]  # Extract codes like "2.a"
    user_values = [user_counts.get(up, 0) for up in user_items]  # Use 0 for missing values
    
    ax2.bar(user_labels, user_values, color='#74C69D', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax2.set_title('User Perspective Distribution', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_facecolor('#FAFBFC')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and code labels on bars
    for i, (v, code) in enumerate(zip(user_values, user_codes)):
        if v > 0:
            # Add code label on top (vertical, black)
            ax2.text(i, v + 1.5, code, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='black', rotation=90)
            # Add count value below the code
            ax2.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=8, color='black')
        else:
            # For zero-count bars, show the code at the bottom with lighter color
            ax2.text(i, 1.5, code, ha='center', va='bottom', 
                    fontsize=8, color='gray', rotation=90)
            ax2.text(i, 0.1, '0', ha='center', va='bottom', fontsize=7, color='gray')
    
    # Plot Developer Perspective  
    dev_counts = Counter(developer_perspective)
    # Get all enum values in order, not just the ones with data
    dev_items = list(DeveloperPerspective)
    dev_labels = [dp.name.replace('_', ' ').title() for dp in dev_items]
    dev_codes = [dp.value.split()[0] for dp in dev_items]  # Extract codes like "3.a"
    dev_values = [dev_counts.get(dp, 0) for dp in dev_items]  # Use 0 for missing values
    
    ax3.bar(dev_labels, dev_values, color='#F9A03F', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax3.set_title('Developer Perspective Distribution', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_facecolor('#FAFBFC')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and code labels on bars
    for i, (v, code) in enumerate(zip(dev_values, dev_codes)):
        if v > 0:
            # Add code label on top (vertical, black)
            ax3.text(i, v + 1.5, code, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='black', rotation=90)
            # Add count value below the code
            ax3.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=8, color='black')
        else:
            # For zero-count bars, show the code at the bottom with lighter color
            ax3.text(i, 1.5, code, ha='center', va='bottom', 
                    fontsize=8, color='gray', rotation=90)
            ax3.text(i, 0.1, '0', ha='center', va='bottom', fontsize=7, color='gray')
    
    # Plot Accelerator Specific
    accel_counts = Counter(accelerator_specific)
    # Get all enum values in order, not just the ones with data
    accel_items = list(AcceleratorSpecific)
    accel_labels = [ac.name.replace('_', ' ').title() for ac in accel_items]
    accel_codes = [ac.value.split()[0] for ac in accel_items]  # Extract codes like "4.a"
    accel_values = [accel_counts.get(ac, 0) for ac in accel_items]  # Use 0 for missing values
    
    ax4.bar(accel_labels, accel_values, color='#F94144', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax4.set_title('Accelerator Specific Distribution', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_facecolor('#FAFBFC')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and code labels on bars
    for i, (v, code) in enumerate(zip(accel_values, accel_codes)):
        if v > 0:
            # Add code label on top (vertical, black)
            ax4.text(i, v + 1.5, code, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='black', rotation=90)
            # Add count value below the code
            ax4.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=8, color='black')
        else:
            # For zero-count bars, show the code at the bottom with lighter color
            ax4.text(i, 1.5, code, ha='center', va='bottom', 
                    fontsize=8, color='gray', rotation=90)
            ax4.text(i, 0.1, '0', ha='center', va='bottom', fontsize=7, color='gray')
    
    # Plot User Expertise
    expertise_counts = Counter(user_expertise)
    # Get all enum values in order, not just the ones with data
    expertise_items = list(UserExpertise)
    expertise_labels = [ue.name.replace('_', ' ').title() for ue in expertise_items]
    expertise_codes = [ue.value.split()[0] for ue in expertise_items]  # Extract codes like "5.a"
    expertise_values = [expertise_counts.get(ue, 0) for ue in expertise_items]  # Use 0 for missing values
    
    ax5.bar(expertise_labels, expertise_values, color='#9D4EDD', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax5.set_title('User Expertise Distribution', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    ax5.set_facecolor('#FAFBFC')
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and code labels on bars
    for i, (v, code) in enumerate(zip(expertise_values, expertise_codes)):
        if v > 0:
            # Add code label on top (vertical, black)
            ax5.text(i, v + 1.5, code, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='black', rotation=90)
            # Add count value below the code
            ax5.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=8, color='black')
        else:
            # For zero-count bars, show the code at the bottom with lighter color
            ax5.text(i, 1.5, code, ha='center', va='bottom', 
                    fontsize=8, color='gray', rotation=90)
            ax5.text(i, 0.1, '0', ha='center', va='bottom', fontsize=7, color='gray')
    
    # Hide the 6th subplot (we only have 5 categories)
    ax6.axis('off')
    
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
        categorized_issues: List of tuples (title, url, is_really_bug, user_perspective, developer_perspective, accelerator_specific, user_expertise)
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
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FAFBFC')
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', alpha=0.9)
    
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
            ax.text(j, i, int(matrix[i, j]),
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
    
    # Count by user expertise
    expertise_counts = Counter(issue[6] for issue in categorized_issues if issue[6] is not None)
    print("\nUser Expertise Distribution:")
    for expertise, count in expertise_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {expertise.name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Load categorized issues from JSON files
    categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/categorized_issues_*.json')
    
    if categorized_issues:
        # Print statistics
        print_statistics(categorized_issues)
        
        # Create platform-specific plots
        plot_all_platforms_distributions(categorized_issues, save_path="platform_distributions.png")
        
        # Also create the original detailed plots if needed
        # plot_bug_distributions(categorized_issues, save_path="bug_distributions.png")
        # plot_combined_heatmap(categorized_issues, save_path="bug_heatmap.png")
    else:
        print("No categorized issues found.")