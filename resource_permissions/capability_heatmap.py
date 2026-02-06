"""
Generate heatmaps showing when each cluster/role has capability to perform work.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_organizational_model(path="organizational_model.pkl"):
    """Load the organizational model from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_cluster_time_heatmap(org_model, role_names=None):
    """
    Create heatmap: Cluster/Role (Y-axis) × Time Type (X-axis)
    Shows when each cluster is available to work.
    """
    capabilities = org_model['organizational_model']['cap']

    # Default role names if not provided
    if role_names is None:
        role_names = {k: f"Cluster {k}" for k in capabilities.keys()}

    # Aggregate by role and time_type
    heatmap_data = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for rg_id, caps in capabilities.items():
        role_label = role_names.get(rg_id, f"Cluster {rg_id}")
        for cap in caps:
            time_type = cap['time_type']
            heatmap_data[role_label][time_type] += cap['overall_score']
            counts[role_label][time_type] += 1

    # Average the scores
    for role in heatmap_data:
        for tt in heatmap_data[role]:
            if counts[role][tt] > 0:
                heatmap_data[role][tt] /= counts[role][tt]

    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data).T

    # Order time types logically
    time_order = ['Morning', 'Lunch', 'Afternoon', 'Evening', 'Off-Hours', 'Weekend']
    df = df.reindex(columns=[t for t in time_order if t in df.columns])

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={'label': 'Average Capability Score'},
                linewidths=0.5)

    plt.title("Cluster Availability by Time of Day", fontsize=14, fontweight='bold')
    plt.xlabel("Time Type", fontsize=12)
    plt.ylabel("Cluster / Role", fontsize=12)
    plt.tight_layout()
    plt.savefig('cluster_time_heatmap.png', dpi=300)
    plt.close()

    return df


def create_cluster_casetype_heatmap(org_model, role_names=None):
    """
    Create heatmap: Cluster/Role (Y-axis) × Case Type (X-axis)
    Shows which complexity levels each cluster handles.
    """
    capabilities = org_model['organizational_model']['cap']

    if role_names is None:
        role_names = {k: f"Cluster {k}" for k in capabilities.keys()}

    heatmap_data = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for rg_id, caps in capabilities.items():
        role_label = role_names.get(rg_id, f"Cluster {rg_id}")
        for cap in caps:
            case_type = cap['case_type']
            heatmap_data[role_label][case_type] += cap['overall_score']
            counts[role_label][case_type] += 1

    for role in heatmap_data:
        for ct in heatmap_data[role]:
            if counts[role][ct] > 0:
                heatmap_data[role][ct] /= counts[role][ct]

    df = pd.DataFrame(heatmap_data).T

    # Order case types by complexity
    case_order = ['Simple', 'Standard', 'Complex', 'Very Complex']
    df = df.reindex(columns=[c for c in case_order if c in df.columns])

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn",
                cbar_kws={'label': 'Average Capability Score'},
                linewidths=0.5)

    plt.title("Cluster Capability by Case Complexity", fontsize=14, fontweight='bold')
    plt.xlabel("Case Type (Complexity)", fontsize=12)
    plt.ylabel("Cluster / Role", fontsize=12)
    plt.tight_layout()
    plt.savefig('cluster_casetype_heatmap.png', dpi=300)
    plt.close()

    return df


def create_full_context_heatmap(org_model, role_names=None):
    """
    Create comprehensive heatmap: Role × (CaseType | TimeType)
    Shows all execution contexts where each role has capability.
    """
    capabilities = org_model['organizational_model']['cap']

    if role_names is None:
        role_names = {k: f"Cluster {k}" for k in capabilities.keys()}

    # Build rows for heatmap
    heatmap_rows = []
    for rg_id, caps in capabilities.items():
        role_label = role_names.get(rg_id, f"Cluster {rg_id}")
        for cap in caps:
            heatmap_rows.append({
                'Role': role_label,
                'Context': f"{cap['case_type']} | {cap['time_type']}",
                'Score': cap['overall_score']
            })

    if not heatmap_rows:
        print("No capabilities found!")
        return None

    df_heat = pd.DataFrame(heatmap_rows)

    # Pivot: Roles on Y-axis, Context on X-axis
    pivot_heat = df_heat.pivot_table(index='Role', columns='Context',
                                      values='Score', aggfunc='mean')

    plt.figure(figsize=(18, 10))
    sns.heatmap(pivot_heat, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={'label': 'Capability Score (λ)'},
                linewidths=0.3)

    plt.title("Complete Role-Context Capability Map", fontsize=16, fontweight='bold')
    plt.xlabel("Execution Context (Complexity | Time)", fontsize=12)
    plt.ylabel("Organizational Role", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('capability_heatmap_full.png', dpi=300)
    plt.close()

    return pivot_heat


def create_activity_cluster_heatmap(org_model):
    """
    Create heatmap showing which activities belong to which cluster.
    Activities (Y-axis) × Cluster (marked with 1/0)
    """
    activity_clusters = org_model['activity_clusters']

    # Get unique clusters
    clusters = sorted(set(activity_clusters.values()))
    activities = sorted(activity_clusters.keys())

    # Build matrix
    matrix = []
    for act in activities:
        row = [1 if activity_clusters[act] == c else 0 for c in clusters]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=activities,
                      columns=[f"Cluster {c}" for c in clusters])

    plt.figure(figsize=(10, 14))
    sns.heatmap(df, cmap="Blues", cbar=False, linewidths=0.5,
                annot=False)

    plt.title("Activity to Cluster Assignment", fontsize=14, fontweight='bold')
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("Activity", fontsize=12)
    plt.tight_layout()
    plt.savefig('activity_cluster_heatmap.png', dpi=300)
    plt.close()

    return df


if __name__ == "__main__":
    # Load model
    org_model = load_organizational_model()

    # Define role names (from your results)
    role_names = {
        1: "Core Process Handler (12 activities)",
        2: "Completion Specialist (1 activity)",
        3: "Submission Handler (1 activity)",
        4: "Denial & Fraud (3 activities)",
        5: "Validation & Pending (8 activities)",
        6: "Collections Specialist (1 activity)"
    }

    print("Generating heatmaps...")

    # 1. Cluster × Time heatmap (when is each cluster available?)
    print("\n1. Cluster × Time Availability:")
    df_time = create_cluster_time_heatmap(org_model, role_names)
    print(df_time)

    # 2. Cluster × Case Type heatmap (what complexity each cluster handles)
    print("\n2. Cluster × Case Complexity:")
    df_case = create_cluster_casetype_heatmap(org_model, role_names)
    print(df_case)

    # 3. Full context heatmap
    print("\n3. Full Context Map:")
    df_full = create_full_context_heatmap(org_model, role_names)

    # 4. Activity → Cluster assignment
    print("\n4. Activity Assignments:")
    df_act = create_activity_cluster_heatmap(org_model)

    print("\nHeatmaps saved to:")
    print("  - cluster_time_heatmap.png")
    print("  - cluster_casetype_heatmap.png")
    print("  - capability_heatmap_full.png")
    print("  - activity_cluster_heatmap.png")
