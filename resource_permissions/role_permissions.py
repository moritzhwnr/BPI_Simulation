import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import seaborn as sns
import pickle
import json
import os

# Constants
DEFAULT_XES_PATH = "/Users/moritz_hawener/Documents/Work/Studium/Master/WS25/BPI/BPI Challenge 2017_1_all/BPI Challenge 2017.xes.gz"
CLUSTERING_THRESHOLD = 5
CLUSTERING_CRITERION = "maxclust"
LAMBDA_THRESHOLD = 0.6
ALPHA = 0.5
RARE_ACTIVITY_THRESHOLD = 1000
GENERAL_ACTIVITY_THRESHOLD = 100000

def load_event_log(file_path):
    """Imports XES log and converts to DataFrame."""
    print(f"Loading event log from: {file_path}")
    event_log = xes_importer.apply(file_path)
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df

def discover_roles(df):
    """Discovers organizational roles using pm4py and clusters them."""
    print("Discovering organizational roles...")
    roles = pm4py.discover_organizational_roles(
        df,
        resource_key='org:resource',
        activity_key='concept:name',
        timestamp_key='time:timestamp',
        case_id_key='case:concept:name'
    )
    
    activity_user = {}
    for role in roles:
        for act in role.activities:
            activity_user[act] = role.originator_importance

    resource_allocation_df = (
        pd.DataFrame(activity_user)
        .fillna(0)
        .astype(float)
    )

    df_log = np.log1p(resource_allocation_df)
    corr_matrix = df_log.corr(method="pearson")
    distance = 1 - corr_matrix
    
    Z = linkage(squareform(distance), method="average")
    clusters = fcluster(Z, t=CLUSTERING_THRESHOLD, criterion=CLUSTERING_CRITERION)

    activity_clusters = pd.Series(
        clusters,
        index=corr_matrix.columns,
        name="cluster"
    ).sort_values()
    
    return activity_clusters, resource_allocation_df, Z

def assign_resources_to_clusters(activity_clusters, resource_allocation_df, df):
    """Assigns resources to clusters based on workload and specialization."""
    print("Assigning resources to clusters...")
    resource_groups = {}

    for cluster_id in sorted(activity_clusters.unique()):
        cluster_activities = activity_clusters[activity_clusters == cluster_id].index
        cluster_events = df[df['concept:name'].isin(cluster_activities)].shape[0]
        cluster_workload = resource_allocation_df[cluster_activities].sum(axis=1)
        resources_with_work = (cluster_workload > 0).sum()
        
        total_workload = resource_allocation_df.sum(axis=1)
        specialization_pct = (cluster_workload / total_workload * 100).fillna(0)
        
        if cluster_events < RARE_ACTIVITY_THRESHOLD:
            # Rare/Specialized
            selected = cluster_workload[cluster_workload > 0].index.tolist()
        elif resources_with_work == 1:
            # Single Expert
            selected = cluster_workload[cluster_workload > 0].index.tolist()
        elif cluster_events > GENERAL_ACTIVITY_THRESHOLD:
            # General
            threshold_pct = 5
            threshold_events = 100
            by_specialization = specialization_pct[specialization_pct >= threshold_pct].index
            by_absolute = cluster_workload[cluster_workload >= threshold_events].index
            selected = list(set(by_specialization) | set(by_absolute))
        else:
            # Medium
            threshold_pct = 10
            threshold_events = 10
            by_specialization = specialization_pct[specialization_pct >= threshold_pct].index
            by_absolute = cluster_workload[cluster_workload >= threshold_events].index
            selected = list(set(by_specialization) | set(by_absolute))
        
        resource_groups[cluster_id] = selected
        
    return resource_groups

def define_execution_contexts(df, activity_clusters, resource_groups):
    """Enriches DataFrame with context information (case_type, activity_type, time_type)."""
    print("Defining execution contexts...")
    
    # helper for case type
    case_case = df.groupby('case:concept:name')['concept:name'].count()
    def get_case_type(n_events):
        if n_events < 10: return 'Simple'
        elif n_events < 20: return 'Standard'
        elif n_events < 30: return 'Complex'
        return 'Very Complex'
    
    case_types = case_case.map(get_case_type)
    df['case_type'] = df['case:concept:name'].map(case_types)
    
    # activity type
    df['activity_type'] = df['concept:name'].map(activity_clusters)
    
    # time type
    date_series = pd.to_datetime(df['time:timestamp'])
    df['hour'] = date_series.dt.hour
    df['day_of_week'] = date_series.dt.dayofweek
    
    def get_time_type(row):
        h, d = row['hour'], row['day_of_week']
        if d >= 5: return 'Weekend'
        if 9 <= h < 12: return 'Morning'
        if 12 <= h < 14: return 'Lunch'
        if 14 <= h < 17: return 'Afternoon'
        if 17 <= h < 20: return 'Evening'
        return 'Off-Hours'
        
    df['time_type'] = df.apply(get_time_type, axis=1)
    return df

def calculate_capabilities(df, resource_groups, role_names):
    """Calculates capabilities for each resource group based on execution contexts."""
    print("Calculating capabilities...")
    capabilities = defaultdict(list)
    
    for rg_id in sorted(resource_groups.keys()):
        rg_resources = set(resource_groups[rg_id])
        
        execution_contexts = df.groupby(['case_type', 'activity_type', 'time_type']).size().reset_index(name='total_events')
        
        for _, row in execution_contexts.iterrows():
            ct, at, tt = row['case_type'], row['activity_type'], row['time_type']
            
            if at != rg_id:
                continue
                
            context_events = df[
                (df['case_type'] == ct) &
                (df['activity_type'] == at) &
                (df['time_type'] == tt)
            ]
            
            if len(context_events) == 0: continue
            
            group_events = context_events[context_events['org:resource'].isin(rg_resources)]
            rel_stake = len(group_events) / len(context_events)
            
            unique_performers = set(group_events['org:resource'].unique())
            coverage = len(unique_performers) / len(rg_resources) if len(rg_resources) > 0 else 0
            
            overall_score = (ALPHA * rel_stake) + ((1 - ALPHA) * coverage)
            
            if overall_score >= LAMBDA_THRESHOLD:
                capabilities[rg_id].append({
                    'case_type': ct,
                    'activity_type': at,
                    'time_type': tt,
                    'rel_stake': rel_stake,
                    'coverage': coverage,
                    'overall_score': overall_score
                })
                
    return capabilities

def build_organizational_model(resource_groups, capabilities):
    """Constructs the formal OM = (RG, mem, cap)."""
    RG = set(resource_groups.keys())
    
    mem = defaultdict(set)
    for rg_id, resources in resource_groups.items():
        for resource in resources:
            mem[resource].add(rg_id)
            
    return {
        'RG': RG,
        'mem': dict(mem),
        'cap': dict(capabilities)
    }

def has_permission(resource, case_type, activity, time_type, organizational_model, activity_clusters):
    """Permission check function."""
    if activity not in activity_clusters:
        return False
        
    activity_type = activity_clusters[activity]
    resource_groups_for_r = organizational_model['mem'].get(resource, set())
    
    if not resource_groups_for_r:
        return False
        
    for rg in resource_groups_for_r:
        group_capabilities = organizational_model['cap'].get(rg, [])
        for cap in group_capabilities:
            if (cap['case_type'] == case_type and
                cap['activity_type'] == activity_type and
                cap['time_type'] == time_type):
                return True
    return False

def generate_context_permission_map(df, organizational_model, activity_clusters):
    """Builds a map of (activity, case_type, time_type) -> list of allowed resources."""
    print("Generating context permission map...")
    context_permission_map = defaultdict(list)
    unique_contexts = df[['concept:name', 'case_type', 'time_type']].drop_duplicates()
    
    # Pre-compute allowed resources validation to speed up
    # However, implementing the exact logic from the original script:
    unique_resources = df['org:resource'].unique()
    
    for _, row in unique_contexts.iterrows():
        act, ct, tt = row['concept:name'], row['case_type'], row['time_type']
        allowed = []
        for res in unique_resources:
            if has_permission(res, ct, act, tt, organizational_model, activity_clusters):
                allowed.append(res)
        context_permission_map[(act, ct, tt)] = allowed
        
    return context_permission_map

def save_output(permission_map, activity_clusters, resource_groups, role_names, organizational_model, context_permission_map):
    """Saves the model to files."""
    print("Saving output files...")
    
    # 1. Full Python Pickle (handles all Python types including numpy)
    export_data = {
        'permission_map': dict(permission_map),
        'activity_clusters': activity_clusters.to_dict(),
        'resource_groups': resource_groups,
        'role_names': role_names,
        'organizational_model': organizational_model,
        'context_permission_map': {
            f"{act}|{ct}|{tt}": resources 
            for (act, ct, tt), resources in context_permission_map.items()
        },
    }

    with open('organizational_model.pkl', 'wb') as f:
        pickle.dump(export_data, f)
    
    print("Saved organizational_model.pkl")
        
    # 2. JSON Export (convert all numpy types to native Python)
    
    # Helper function to convert numpy types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_to_native(k): convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    # Convert organizational model
    json_om = {
        'RG': list(organizational_model['RG']),
        'mem': {str(k): list(v) for k, v in organizational_model['mem'].items()},
        'cap': {
            int(k): [
                {
                    'case_type': cap['case_type'],
                    'activity_type': int(cap['activity_type']),
                    'time_type': cap['time_type'],
                    'rel_stake': float(cap['rel_stake']),
                    'coverage': float(cap['coverage']),
                    'overall_score': float(cap['overall_score'])
                }
                for cap in v
            ]
            for k, v in organizational_model['cap'].items()
        }
    }
    
    # Build JSON export with proper type conversion
    export_json = {
        'permission_map': permission_map,
        'context_permission_map': {
            f"{act}|{ct}|{tt}": resources 
            for (act, ct, tt), resources in context_permission_map.items()
        },
        'activity_clusters': {str(k): int(v) for k, v in activity_clusters.items()},
        'resource_groups': {str(k): v for k, v in resource_groups.items()},
        'role_names': {str(k): v for k, v in role_names.items()},
        'organizational_model': json_om
    }
    
    # Pass the entire dictionary through your conversion helper first
    cleaned_export = convert_to_native(export_json)

    with open('organizational_model.json', 'w') as f:
        json.dump(cleaned_export, f, indent=2)
        
    print("Saved organizational_model.json")
    
    # 3. BONUS: Export context permissions as CSV for easy inspection
    context_df_data = []
    for (activity, case_type, time_type), resources in context_permission_map.items():
        context_df_data.append({
            'activity': activity,
            'case_type': case_type,
            'time_type': time_type,
            'num_resources': len(resources),
            'resources_sample': ', '.join(resources[:3]) + ('...' if len(resources) > 3 else '')
        })
    
    if context_df_data:
        context_df = pd.DataFrame(context_df_data)
        context_df.to_csv('context_permissions.csv', index=False)
        print(f"Saved context_permissions.csv ({len(context_df)} contexts)")
    
    # 4. BONUS: Export role summary
    role_summary = []
    for cluster_id, role_name in role_names.items():
        capabilities = organizational_model['cap'].get(cluster_id, [])
        role_summary.append({
            'cluster_id': cluster_id,
            'role_name': role_name,
            'num_resources': len(resource_groups[cluster_id]),
            'num_capabilities': len(capabilities),
            'resources_sample': ', '.join(resource_groups[cluster_id][:5]) + ('...' if len(resource_groups[cluster_id]) > 5 else '')
        })
    
    role_summary_df = pd.DataFrame(role_summary)
    role_summary_df.to_csv('role_summary.csv', index=False)
    print(f"Saved role_summary.csv")
    
    print("\nExport Summary:")
    print(f"   Simple permissions: {len(permission_map)} activities")
    print(f"   Context permissions: {len(context_permission_map)} contexts")
    print(f"   Resource groups: {len(resource_groups)} clusters")
    print(f"   Total resources: {len(organizational_model['mem'])} resources")

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_colored_dendrogram(Z, activity_clusters, activity_names):
    plt.figure(figsize=(15, 8))
    plt.title("Organizational Roles: Activity Clusters by Social Distance", fontsize=16)
    
    # Calculate threshold dynamically to match the number of clusters found
    # We want a threshold that cuts the dendrogram into the same number of clusters as 'activity_clusters'
    num_clusters = activity_clusters.nunique()
    
    if num_clusters > 1 and len(Z) >= num_clusters:
        # The merge at index -(k-1) creates k-1 clusters. We want to be below this distance.
        upper_bound = Z[-(num_clusters - 1), 2]
        # The merge at index -k creates k clusters. We want to be above this distance to color them.
        lower_bound = Z[-num_clusters, 2]
        
        # Use midpoint between the boundaries
        threshold = (upper_bound + lower_bound) / 2.0
    else:
        # Fallback
        threshold = 0.5 * max(Z[:, 2]) if len(Z) > 0 else 0 
    
    # Generate the dendrogram
    dend_data = dendrogram(
        Z,
        labels=activity_names,
        leaf_rotation=90,
        leaf_font_size=11,
        color_threshold=threshold,  # This applies the colors based on your clusters
        above_threshold_color='grey'
    )
    
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Role Boundary (Threshold {threshold})')
    plt.ylabel("Distance (1 - Pearson Correlation)", fontsize=12)
    plt.xlabel("Activities (Grouped by Resource Footprint)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('colored_activity_dendrogram.png', dpi=300)
    plt.show()

def plot_capability_heatmap(capabilities, role_names):
    # Flatten the capabilities dictionary into a list for DataFrame conversion
    heatmap_rows = []
    for rg_id, caps in capabilities.items():
        # Map the numeric cluster ID to your human-readable role names
        role_label = role_names.get(rg_id, f"Role {rg_id}")
        for cap in caps:
            heatmap_rows.append({
                'Role': role_label,
                'Context': f"{cap['case_type']} | {cap['time_type']}",
                'Score': cap['overall_score']
            })
    
    df_heat = pd.DataFrame(heatmap_rows)
    
    # Pivot the data: Roles on the Y-axis, Context (Case+Time) on the X-axis
    pivot_heat = df_heat.pivot(index='Role', columns='Context', values='Score')

    plt.figure(figsize=(16, 9))
    sns.heatmap(pivot_heat, annot=True, fmt=".2f", cmap="YlGnBu", 
                cbar_kws={'label': 'Capability Score ($\lambda$)'})
    
    plt.title("Role-Context Capability Map (Organizational Maturity)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Discovered Organizational Roles", fontsize=12)
    plt.xlabel("Execution Context (Complexity | Time of Day)", fontsize=12)
    plt.tight_layout()
    plt.savefig('capability_heatmap.png', dpi=300)
    plt.show()


def main():
    file_path = DEFAULT_XES_PATH
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Please check the path.")
        return

    # 1. Load Data
    df = load_event_log(file_path)
    
    # 2. Roles & Clustering
    activity_clusters, resource_allocation_df, Z = discover_roles(df)
    
    # 3. Resource Assignment
    resource_groups = assign_resources_to_clusters(activity_clusters, resource_allocation_df, df)
    
    # Role Names Mapping (Could be externalized)
    role_names = {
        1: "Core Process Handler",
        2: "Completion Specialist", 
        3: "Submission Handler",
        4: "Support & Communication Specialist",
        5: "Collections Specialist"
    }

    # 4. Context Definition
    df = define_execution_contexts(df, activity_clusters, resource_groups)
    
    # 5. Capabilities
    capabilities = calculate_capabilities(df, resource_groups, role_names)
    
    # 6. Build Model
    organizational_model = build_organizational_model(resource_groups, capabilities)
    
    # 7. Generate simple permission map for backward compatibility
    simple_permission_map = {}
    for activity, cluster_id in activity_clusters.items():
        simple_permission_map[activity] = resource_groups[cluster_id]

    # 8. Save
    save_output(simple_permission_map, activity_clusters, resource_groups, role_names, organizational_model)
    print("Process complete.")

if __name__ == "__main__":
    main()
