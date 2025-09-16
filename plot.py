"""Plotting functions for visualizing bug categorization distributions"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from results_loader import load_categorized_results, load_categorized_json_files
from cates import IsReallyBug, UserPerspective, DeveloperPerspective, AcceleratorSpecific, PlatformSpecificity

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
    platform_specificity = [issue[6] for issue in categorized_issues if issue[6] is not None]
    
    # Count each category
    categories = {
        'Bug Class': Counter(is_really_bug),
        'User Symptom': Counter(user_perspective),
        'Root Cause': Counter(developer_perspective),
        'Resolution': Counter(accelerator_specific),
        'Platform': Counter(platform_specificity)
    }
    
    # Prepare data for grouped bar plot with subtitles
    category_info = {
        'Bug Class': 'Bug classification',
        'User Symptom': 'User-visible symptoms',
        'Root Cause': 'Root cause analysis',
        'Resolution': 'Resolution status',
        'Platform': 'Platform specificity'
    }
    category_names = list(categories.keys())
    
    # Calculate max items based on all possible enum values to ensure consistent bar width
    # Use the maximum number of enum values across all categories
    max_possible_items = max(
        len(list(IsReallyBug)),        # 6 items
        len(list(UserPerspective)),     # 9 items
        len(list(DeveloperPerspective)), # 7 items
        len(list(AcceleratorSpecific)), # 5 items
        len(list(PlatformSpecificity))  # 4 items
    )  # This will be 9 (UserPerspective has the most)
    bar_width = min(0.15, 0.8 / max_possible_items)  # Consistent bar width across all plots
    x_pos = np.arange(len(category_names))
    
    # Define extended color palettes for each category type
    # Using colorblind-friendly and visually appealing colors with gradients
    category_palettes = {
        'Bug Class': ['#1e5f8e', '#2E86AB', '#5EB1BF', '#84D2F6', '#A4C3D2'],  # Blues (5 colors for 5 options)
        'User Symptom': ['#2d6a4f', '#40916c', '#52B788', '#74C69D', '#95D5B2', '#B7E4C7', '#D8F3DC'],  # Greens (7 colors for 7 options)
        'Root Cause': ['#d84a05', '#F77F00', '#F9A03F', '#FCBF49', '#FFD166', '#FFE5A5'],  # Oranges (6 colors for 6 options)
        'Resolution': ['#a61e4d', '#D62828', '#F94144', '#F3722C'],  # Reds (4 colors for 4 options)
        'Platform': ['#5b0e8c', '#7209B7', '#9D4EDD', '#B298DC']  # Purples (4 colors for 4 options)
    }
    
    # Track all unique values across categories for legend
    all_values = set()
    for cat_counts in categories.values():
        all_values.update(cat_counts.keys())
    
    # Plot bars for each unique value
    plotted_items = {}
    
    # Define all possible enum values for each category in their original order
    all_enums = {
        'Bug Class': list(IsReallyBug),
        'User Symptom': list(UserPerspective),
        'Root Cause': list(DeveloperPerspective),
        'Resolution': list(AcceleratorSpecific),
        'Platform': list(PlatformSpecificity)
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
            # Use the full enum value text, truncate if too long
            full_text = item.value
            # Truncate to first 25 characters for readability
            display_text = full_text[:25] + "..." if len(full_text) > 25 else full_text
            
            if count > 0:
                # Add full text label on top (vertical, black) with left anchor
                ax.text(bar_position, count + 0.5, display_text,
                       ha='left', va='bottom', fontsize=10, fontweight='semibold',
                       color='black', rotation=90)
                # Add count value below the label
                ax.text(bar_position, count + 0.2, str(count),
                       ha='center', va='bottom', fontsize=10, color='black')
            else:
                # For zero-count bars, show the text at the bottom with lighter color
                ax.text(bar_position, 1.5, display_text,
                       ha='left', va='bottom', fontsize=10, 
                       color='gray', rotation=90)
                ax.text(bar_position, 0.1, '0',
                       ha='center', va='bottom', fontsize=10, color='gray')
    
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
    
    # Create figure with 2x3 subplots - more horizontal layout
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.suptitle('Bug Categorization Distributions by Platform', fontsize=18, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('#F8F9FA')
    
    # Define platform order
    platforms = ['PyTorch', 'TensorFlow', 'JAX', 'TensorRT', 'Triton', 'All']
    
    # Plot individual platforms
    for idx, platform in enumerate(platforms):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        if platform == 'All':
            plot_platform_distributions(categorized_issues, title="All Platforms Combined", ax=ax)
        else:
            platform_data = platform_issues.get(platform, [])
            plot_platform_distributions(platform_data, title=platform, ax=ax)

    # Adjust layout with manual spacing for horizontal layout
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08, hspace=0.35, wspace=0.15)

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Platform distributions saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_definitely_bugs_distributions(categorized_issues, save_path="definitely_bugs_distributions.png"):
    """
    Create bar plots showing the distribution of categorization dimensions for only confirmed bugs (1.d).
    Similar to plot_bug_distributions but filtered to only confirmed bugs.

    Args:
        categorized_issues: List of all categorized issue tuples
        save_path: Path to save the combined figure
    """
    # Filter for only definitely bugs (1.d)
    definitely_bugs = [issue for issue in categorized_issues
                      if issue[2] is not None and issue[2] == IsReallyBug.CONFIRMED_BUG]

    if not definitely_bugs:
        print("No issues categorized as 'confirmed bug' (1.d) found.")
        return None

    # Extract the categorizations for confirmed bugs only
    user_perspective = [issue[3] for issue in definitely_bugs if issue[3] is not None]
    developer_perspective = [issue[4] for issue in definitely_bugs if issue[4] is not None]
    accelerator_specific = [issue[5] for issue in definitely_bugs if issue[5] is not None]
    platform_specificity = [issue[6] for issue in definitely_bugs if issue[6] is not None]

    # Create subplots - 2x2 grid for the 4 remaining dimensions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Distribution of Confirmed Bugs Only (n={len(definitely_bugs)})', fontsize=18, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')

    # Plot User Perspective
    user_counts = Counter(user_perspective)
    user_items = list(UserPerspective)
    user_labels = [up.name.replace('_', ' ').title() for up in user_items]
    user_texts = [up.value[:25] + "..." if len(up.value) > 25 else up.value for up in user_items]
    user_values = [user_counts.get(up, 0) for up in user_items]

    ax1.bar(user_labels, user_values, color='#74C69D', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax1.set_title('User-Visible Symptoms (Confirmed Bugs)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_facecolor('#FAFBFC')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(user_values, user_texts)):
        if v > 0:
            ax1.text(i, v + 0.5, text, ha='left', va='bottom',
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=7, color='black')
        else:
            ax1.text(i, 0.5, text, ha='left', va='bottom',
                    fontsize=10, color='gray', rotation=90)
            ax1.text(i, 0.05, '0', ha='center', va='bottom', fontsize=10, color='gray')

    # Plot Developer Perspective
    dev_counts = Counter(developer_perspective)
    dev_items = list(DeveloperPerspective)
    dev_labels = [dp.name.replace('_', ' ').title() for dp in dev_items]
    dev_texts = [dp.value[:25] + "..." if len(dp.value) > 25 else dp.value for dp in dev_items]
    dev_values = [dev_counts.get(dp, 0) for dp in dev_items]

    ax2.bar(dev_labels, dev_values, color='#F9A03F', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax2.set_title('Root Cause Analysis (Confirmed Bugs)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_facecolor('#FAFBFC')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(dev_values, dev_texts)):
        if v > 0:
            ax2.text(i, v + 0.5, text, ha='left', va='bottom',
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=7, color='black')
        else:
            ax2.text(i, 0.5, text, ha='left', va='bottom',
                    fontsize=10, color='gray', rotation=90)
            ax2.text(i, 0.05, '0', ha='center', va='bottom', fontsize=10, color='gray')

    # Plot Resolution Status
    accel_counts = Counter(accelerator_specific)
    accel_items = list(AcceleratorSpecific)
    accel_labels = [ac.name.replace('_', ' ').title() for ac in accel_items]
    accel_texts = [ac.value[:25] + "..." if len(ac.value) > 25 else ac.value for ac in accel_items]
    accel_values = [accel_counts.get(ac, 0) for ac in accel_items]

    ax3.bar(accel_labels, accel_values, color='#F94144', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax3.set_title('Resolution Status (Confirmed Bugs)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_facecolor('#FAFBFC')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(accel_values, accel_texts)):
        if v > 0:
            ax3.text(i, v + 0.5, text, ha='left', va='bottom',
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=7, color='black')
        else:
            ax3.text(i, 0.5, text, ha='left', va='bottom',
                    fontsize=10, color='gray', rotation=90)
            ax3.text(i, 0.05, '0', ha='center', va='bottom', fontsize=10, color='gray')

    # Plot Platform Specificity
    platform_counts = Counter(platform_specificity)
    platform_items = list(PlatformSpecificity)
    platform_labels = [ps.name.replace('_', ' ').title() for ps in platform_items]
    platform_texts = [ps.value[:25] + "..." if len(ps.value) > 25 else ps.value for ps in platform_items]
    platform_values = [platform_counts.get(ps, 0) for ps in platform_items]

    ax4.bar(platform_labels, platform_values, color='#9D4EDD', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax4.set_title('Platform Specificity (Confirmed Bugs)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_facecolor('#FAFBFC')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(platform_values, platform_texts)):
        if v > 0:
            ax4.text(i, v + 0.5, text, ha='left', va='bottom',
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            ax4.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=7, color='black')
        else:
            ax4.text(i, 0.5, text, ha='left', va='bottom',
                    fontsize=10, color='gray', rotation=90)
            ax4.text(i, 0.05, '0', ha='center', va='bottom', fontsize=10, color='gray')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confirmed bugs distributions saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_platform_filtered_distributions(categorized_issues, platform_level, save_path=None):
    """
    Create a single figure with 6 subplots showing only issues for a specific platform specificity.
    
    Args:
        categorized_issues: List of all categorized issue tuples
        platform_level: PlatformSpecificity enum value to filter by
        save_path: Path to save the combined figure
    """
    # Filter for specific platform specificity
    filtered_issues = [issue for issue in categorized_issues 
                       if issue[6] is not None and issue[6] == platform_level]
    
    if not filtered_issues:
        print(f"No issues found for platform specificity: {platform_level.name}")
        return None
    
    # Separate filtered issues by platform
    platform_issues = defaultdict(list)
    
    for issue in filtered_issues:
        url = issue[1]  # URL is at index 1
        framework = get_framework_from_url(url)
        platform_issues[framework].append(issue)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Bug Categorization Distributions - {platform_level.name.title()} Platform Specificity', 
                 fontsize=18, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('#F8F9FA')
    
    # Define platform order
    platforms = ['PyTorch', 'TensorFlow', 'JAX', 'TensorRT', 'Triton', 'All']
    
    # Plot individual platforms
    for idx, platform in enumerate(platforms):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        if platform == 'All':
            plot_platform_distributions(filtered_issues, 
                                       title=f"All Platforms ({platform_level.name.title()})", ax=ax)
        else:
            platform_data = platform_issues.get(platform, [])
            plot_platform_distributions(platform_data, 
                                       title=f"{platform} ({platform_level.name.title()})", ax=ax)
        
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"{platform_level.name.title()} platform specificity distributions saved to {save_path}")
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
    platform_specificity = [issue[6] for issue in categorized_issues if issue[6] is not None]
    
    # Create subplots - now with 6 plots (2x3 grid)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Distribution of GPU Bug Categorizations', fontsize=18, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')
    
    # Plot Is Really Bug
    bug_counts = Counter(is_really_bug)
    # Get all enum values in order, not just the ones with data
    bug_items = list(IsReallyBug)
    bug_labels = [bt.name.replace('_', ' ').title() for bt in bug_items]
    bug_texts = [bt.value[:25] + "..." if len(bt.value) > 25 else bt.value for bt in bug_items]  # Full text, truncated
    bug_values = [bug_counts.get(bt, 0) for bt in bug_items]  # Use 0 for missing values
    
    ax1.bar(bug_labels, bug_values, color='#5EB1BF', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax1.set_title(f'Is Really Bug Distribution (n={len(is_really_bug)})', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_facecolor('#FAFBFC')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(bug_values, bug_texts)):
        if v > 0:
            # Add full text label on top (vertical, black) with left anchor
            ax1.text(i, v + 1.5, text, ha='left', va='bottom', 
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            # Add count value below the label
            ax1.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10, color='black')
        else:
            # For zero-count bars, show the text at the bottom with lighter color
            ax1.text(i, 1.5, text, ha='left', va='bottom', 
                    fontsize=10, color='gray', rotation=90)
            ax1.text(i, 0.1, '0', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Plot User Perspective
    user_counts = Counter(user_perspective)
    # Get all enum values in order, not just the ones with data
    user_items = list(UserPerspective)
    user_labels = [up.name.replace('_', ' ').title() for up in user_items]
    user_texts = [up.value[:25] + "..." if len(up.value) > 25 else up.value for up in user_items]  # Full text, truncated
    user_values = [user_counts.get(up, 0) for up in user_items]  # Use 0 for missing values
    
    ax2.bar(user_labels, user_values, color='#74C69D', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax2.set_title(f'User Perspective Distribution (n={len(user_perspective)})', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_facecolor('#FAFBFC')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(user_values, user_texts)):
        if v > 0:
            # Add full text label on top (vertical, black) with left anchor
            ax2.text(i, v + 1.5, text, ha='left', va='bottom', 
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            # Add count value below the label
            ax2.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10, color='black')
        else:
            # For zero-count bars, show the text at the bottom with lighter color
            ax2.text(i, 1.5, text, ha='left', va='bottom', 
                    fontsize=10, color='gray', rotation=90)
            ax2.text(i, 0.1, '0', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Plot Developer Perspective  
    dev_counts = Counter(developer_perspective)
    # Get all enum values in order, not just the ones with data
    dev_items = list(DeveloperPerspective)
    dev_labels = [dp.name.replace('_', ' ').title() for dp in dev_items]
    dev_texts = [dp.value[:25] + "..." if len(dp.value) > 25 else dp.value for dp in dev_items]  # Full text, truncated
    dev_values = [dev_counts.get(dp, 0) for dp in dev_items]  # Use 0 for missing values
    
    ax3.bar(dev_labels, dev_values, color='#F9A03F', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax3.set_title(f'Developer Perspective Distribution (n={len(developer_perspective)})', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_facecolor('#FAFBFC')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(dev_values, dev_texts)):
        if v > 0:
            # Add full text label on top (vertical, black) with left anchor
            ax3.text(i, v + 1.5, text, ha='left', va='bottom', 
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            # Add count value below the label
            ax3.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10, color='black')
        else:
            # For zero-count bars, show the text at the bottom with lighter color
            ax3.text(i, 1.5, text, ha='left', va='bottom', 
                    fontsize=10, color='gray', rotation=90)
            ax3.text(i, 0.1, '0', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Plot Accelerator Specific
    accel_counts = Counter(accelerator_specific)
    # Get all enum values in order, not just the ones with data
    accel_items = list(AcceleratorSpecific)
    accel_labels = [ac.name.replace('_', ' ').title() for ac in accel_items]
    accel_texts = [ac.value[:25] + "..." if len(ac.value) > 25 else ac.value for ac in accel_items]  # Full text, truncated
    accel_values = [accel_counts.get(ac, 0) for ac in accel_items]  # Use 0 for missing values
    
    ax4.bar(accel_labels, accel_values, color='#F94144', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax4.set_title(f'Accelerator Specific Distribution (n={len(accelerator_specific)})', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_facecolor('#FAFBFC')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(accel_values, accel_texts)):
        if v > 0:
            # Add full text label on top (vertical, black) with left anchor
            ax4.text(i, v + 1.5, text, ha='left', va='bottom', 
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            # Add count value below the label
            ax4.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10, color='black')
        else:
            # For zero-count bars, show the text at the bottom with lighter color
            ax4.text(i, 1.5, text, ha='left', va='bottom', 
                    fontsize=10, color='gray', rotation=90)
            ax4.text(i, 0.1, '0', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Plot Platform Specificity
    platform_counts = Counter(platform_specificity)
    # Get all enum values in order, not just the ones with data
    platform_items = list(PlatformSpecificity)
    platform_labels = [ps.name.replace('_', ' ').title() for ps in platform_items]
    platform_texts = [ps.value[:25] + "..." if len(ps.value) > 25 else ps.value for ps in platform_items]  # Full text, truncated
    platform_values = [platform_counts.get(ps, 0) for ps in platform_items]  # Use 0 for missing values
    
    ax5.bar(platform_labels, platform_values, color='#9D4EDD', edgecolor='#2D3436', linewidth=0.8, alpha=0.85)
    ax5.set_title(f'Platform Specificity Distribution (n={len(platform_specificity)})', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    ax5.set_facecolor('#FAFBFC')
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value and text labels on bars
    for i, (v, text) in enumerate(zip(platform_values, platform_texts)):
        if v > 0:
            # Add full text label on top (vertical, black) with left anchor
            ax5.text(i, v + 1.5, text, ha='left', va='bottom', 
                    fontsize=10, fontweight='semibold', color='black', rotation=90)
            # Add count value below the label
            ax5.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10, color='black')
        else:
            # For zero-count bars, show the text at the bottom with lighter color
            ax5.text(i, 1.5, text, ha='left', va='bottom', 
                    fontsize=10, color='gray', rotation=90)
            ax5.text(i, 0.1, '0', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Hide the 6th subplot (we only have 5 categories now)
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
    
    # Count by bug classification
    bug_counts = Counter(issue[2] for issue in categorized_issues if issue[2] is not None)
    print("\nBug Classification Distribution:")
    for bug_type, count in bug_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {bug_type.name}: {count} ({percentage:.1f}%)")
    
    # Count by user-visible symptoms
    user_counts = Counter(issue[3] for issue in categorized_issues if issue[3] is not None)
    print("\nUser-Visible Symptoms Distribution:")
    for user_persp, count in user_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {user_persp.name}: {count} ({percentage:.1f}%)")
    
    # Count by root cause
    dev_counts = Counter(issue[4] for issue in categorized_issues if issue[4] is not None)
    print("\nRoot Cause Distribution:")
    for dev_persp, count in dev_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {dev_persp.name}: {count} ({percentage:.1f}%)")
    
    # Count by resolution status (was accelerator specific)
    resolution_counts = Counter(issue[5] for issue in categorized_issues if issue[5] is not None)
    print("\nResolution Status Distribution:")
    for resolution, count in resolution_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {resolution.name}: {count} ({percentage:.1f}%)")
    
    # Count by platform specificity (was user expertise)
    platform_counts = Counter(issue[6] for issue in categorized_issues if issue[6] is not None)
    print("\nPlatform Specificity Distribution:")
    for platform, count in platform_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {platform.name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Load categorized issues from JSON files
    categorized_issues = load_categorized_results('/Users/bubblepipe/repo/gpu-bugs/64.json')

    if categorized_issues:
        print(f'Loaded {len(categorized_issues)} issues from 64.json\n')

        # Print statistics
        print_statistics(categorized_issues)

        print('\nGenerating plots...')

        # # Create platform-specific plots for all issues
        # plot_all_platforms_distributions(categorized_issues, save_path="platform_distributions.png")
        # print('  ✓ Platform distributions saved to platform_distributions.png')

        # Create platform-specific plots for only confirmed bugs (1.d)
        plot_definitely_bugs_distributions(categorized_issues, save_path="confirmed_bugs_distributions.png")
        print('  ✓ Confirmed bugs distributions saved to confirmed_bugs_distributions.png')

        # Create the detailed bug distribution plots
        plot_bug_distributions(categorized_issues, save_path="bug_distributions.png")
        print('  ✓ Bug distributions saved to bug_distributions.png')

        # Optionally create heatmap
        # plot_combined_heatmap(categorized_issues, save_path="64_bug_heatmap.png")
        # print('  ✓ Bug heatmap saved to 64_bug_heatmap.png')

        print('\nAll plots generated successfully!')
    else:
        print("No categorized issues found.")