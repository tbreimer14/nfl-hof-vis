import pandas as pd
import altair as alt
import numpy as np
import os

# ==========================================
# 1. LOAD DATA (Or Generate Dummy Data)
# ==========================================

# File names
FILES = {
    'influences': 'influences.csv',
    'train_names': 'train_names.csv',
    'val_names': 'val_names.csv',
    'stats': 'career_stats.csv'
}

# Check if files exist, otherwise generate dummy data for demonstration
if not all(os.path.exists(f) for f in FILES.values()):
    print("Files not found. Generating dummy data for demonstration...")
    
    # Create dummy names
    train_names_list = [f'Train_Player_{i}' for i in range(50)]
    val_names_list = [f'Val_Player_{i}' for i in range(10)]
    
    # Dummy Influence Matrix (Val x Train)
    # Rows: Validation Samples, Cols: Training Samples
    influences_data = np.random.randn(len(val_names_list), len(train_names_list))
    pd.DataFrame(influences_data).to_csv(FILES['influences'], index=False, header=False)
    
    # Dummy Name Files
    pd.DataFrame(train_names_list).to_csv(FILES['train_names'], index=False, header=False)
    pd.DataFrame(val_names_list).to_csv(FILES['val_names'], index=False, header=False)
    
    # Dummy Stats
    stats_data = {
        'Name': train_names_list,
        'Points': np.random.randint(10, 30, size=len(train_names_list)),
        'Assists': np.random.randint(0, 10, size=len(train_names_list)),
        'Rebounds': np.random.randint(0, 15, size=len(train_names_list))
    }
    pd.DataFrame(stats_data).to_csv(FILES['stats'], index=False)
    print("Dummy data generated.")

# Load the Data
print("Loading data...")
# Assuming influences is a matrix without headers/index in the file
inf_matrix = pd.read_csv(FILES['influences'], header=None)
val_names = pd.read_csv(FILES['val_names'], header=None).iloc[:, 0].tolist()
train_names = pd.read_csv(FILES['train_names'], header=None).iloc[:, 0].tolist()
career_stats = pd.read_csv(FILES['stats'])

# ==========================================
# 2. DATA PROCESSING
# ==========================================
print("Processing data...")

# Assign names to the matrix
# Ensure dimensions match
if inf_matrix.shape != (len(val_names), len(train_names)):
    print(f"Warning: Matrix shape {inf_matrix.shape} does not match name counts ({len(val_names)}, {len(train_names)})")
    # Truncate to fit for safety in this skeleton
    inf_matrix = inf_matrix.iloc[:len(val_names), :len(train_names)]

inf_matrix.index = val_names
inf_matrix.columns = train_names

# 1. Melt the matrix to long format: [val_name, train_name, influence]
# This creates a row for every pair. 
df_long = inf_matrix.reset_index().melt(id_vars='index', var_name='train_name', value_name='influence')
df_long.rename(columns={'index': 'val_name'}, inplace=True)

# 2. Filter to Top 10 Influences per Validation Player
# We calculate absolute influence to find the "most influential" (positive or negative)
df_long['abs_influence'] = df_long['influence'].abs()

# Sort by validation player and influence, then take top 10
top_influences = df_long.sort_values(['val_name', 'abs_influence'], ascending=[True, False]) \
                        .groupby('val_name').head(10)

# 3. Merge with Career Stats
# We merge on 'train_name' == 'Name'
viz_data = top_influences.merge(career_stats, left_on='train_name', right_on='Name', how='left')

# 4. Melt the stats part for the detail view
# This makes it easy to list multiple stats (Points, Assists, etc.) in a generic table
stat_columns = [c for c in career_stats.columns if c != 'Name']
viz_data_long = viz_data.melt(
    id_vars=['val_name', 'train_name', 'influence'], 
    value_vars=stat_columns, 
    var_name='Stat_Name', 
    value_name='Stat_Value'
)

# ==========================================
# 3. ALTAIR VISUALIZATION
# ==========================================
print("Building visualization...")

# A. Create the Selector for the Validation Player (Dropdown)
# This binds to the 'val_name' column
val_dropdown = alt.binding_select(options=sorted(viz_data['val_name'].unique()), name="Select Validation Player: ")
selection_val = alt.selection_single(fields=['val_name'], bind=val_dropdown, init={'val_name': val_names[0]})

# B. Create the Selector for the Training Player (Click on Bar)
# This allows clicking a bar to filter the stats view
selection_train = alt.selection_single(fields=['train_name'], empty='none')

# C. Chart 1: Bar Graph of Top Influences
# We use max(influence) to ensure we get unique bars per player (since joining with stats duplicates rows)
bars = alt.Chart(viz_data).mark_bar().encode(
    x=alt.X('influence:Q', title='Influence Score'),
    y=alt.Y('train_name:N', sort='-x', title='Training Player'),
    color=alt.condition(selection_train, alt.value('steelblue'), alt.value('lightgray')),
    tooltip=['train_name', 'influence']
).add_selection(
    selection_val,
    selection_train
).transform_filter(
    selection_val
).properties(
    title="Top 10 Influencing Training Instances",
    width=400,
    height=300
)

# D. Chart 2: Stats Detail View
# Displays the stats for the selected training player
stats_text = alt.Chart(viz_data_long).mark_text(size=16).encode(
    y=alt.Y('Stat_Name:N', axis=alt.Axis(title='Statistic')),
    text='Stat_Value:Q',
    color=alt.value('black')
).transform_filter(
    selection_val  # Must match the validation player
).transform_filter(
    selection_train # Must match the clicked bar
).properties(
    title="Player Stats (Click a bar to view)",
    width=200,
    height=300
)

# Combine the charts side-by-side
final_chart = bars | stats_text

# Save to HTML
output_file = 'influence_viz.html'
final_chart.save(output_file)

print(f"Done! Open '{output_file}' in your browser.")