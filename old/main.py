import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from networkx import display
from pydantic import BaseModel
import altair as alt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pydvl.influence.torch import DirectInfluence
import torch.nn.functional as F

app = FastAPI()

# Allow your browser to make requests to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClickRequest(BaseModel):
    row_index: int
    column_name: str
    value: str

class InteractionRequest(BaseModel):
    train_id: int
    test_id: int
    influence: float

class TrainRequest(BaseModel):
    learning_rate: float
    epochs: int

class HOFPredictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HOFPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

class QBDataSet(Dataset):
    def __init__(self, X, y):
        # Ensure data is float32 for features and long for targets (standard for PyTorch)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

_data_cache = {}

def get_processed_data():
    """
    Loads data only if not already loaded. 
    Returns processed numpy arrays/series ready for training.
    """
    if "X_train" in _data_cache:
        return _data_cache

    try:
        df = pd.read_csv('career_stats.csv')
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="career_stats.csv not found")

    target = 'HOF'
    # Drop non-feature columns
    features_to_drop = ['QBrec', 'HOF', 'Name']
    
    # X is still a Pandas DataFrame here, so it has column names
    X = df.drop(columns=features_to_drop, errors='ignore')
    X = X.fillna(0)
    y = df[target]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get Names corresponding to the split
    train_names = df['Name'].loc[X_train.index]
    val_names = df['Name'].loc[X_val.index]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    _data_cache["X_train"] = X_train_scaled
    _data_cache["y_train"] = y_train
    _data_cache["X_val"] = X_val_scaled
    _data_cache["y_val"] = y_val
    _data_cache["train_names"] = train_names
    _data_cache["val_names"] = val_names
    _data_cache["input_dim"] = X_train.shape[1]
    _data_cache["scaler"] = scaler
    
    # CHANGE IS HERE: Use X.columns.tolist() instead of generating "Feat i"
    _data_cache["feature_names"] = X.columns.tolist()
    
    return _data_cache

@app.post("/train-and-visualize")
def train_model(req: TrainRequest):
    data = get_processed_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    train_dataset = QBDataSet(X_train, y_train.values)
    val_dataset = QBDataSet(X_val, y_val.values)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = HOFPredictor(input_dim=data["input_dim"], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=req.learning_rate)

    for epoch in range(req.epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        outputs = model(X_val_tensor)
        y_pred = outputs.argmax(dim=1).numpy()

    infl_model = DirectInfluence(model, nn.CrossEntropyLoss(), regularization=0.01)
    infl_model = infl_model.fit(train_loader)
    influences = infl_model.influences(val_dataset.X, val_dataset.y, train_dataset.X, train_dataset.y)

    influences = influences.cpu().numpy()
    _data_cache["influences"] = influences
    _data_cache["model"] = model

    val_df = pd.DataFrame({
        'Name': data["val_names"].values, 
        'Actual': y_val.values,
        'Predicted': y_pred
    })
    
    val_df = pd.DataFrame({
        'Name': data["val_names"].values, 
        'Actual': y_val.values,
        'Predicted': y_pred
    })
    
    val_df['RowIndex'] = val_df.index
    val_df['Correct'] = val_df['Actual'] == val_df['Predicted']

    melted_df = val_df.melt(
    id_vars=['RowIndex'], 
    value_vars=['Name', 'Actual', 'Predicted'],
    var_name='ColumnName',
    value_name='ValueAsString'   
    )

    final_df = pd.merge(melted_df, val_df, on='RowIndex')
    alt.data_transformers.disable_max_rows()

    base = alt.Chart(final_df).encode(
        y=alt.Y('RowIndex:O', title=None),
        x=alt.X('ColumnName:O', title=None, sort=['Name', 'Actual', 'Predicted'])
    )

    rects = base.mark_rect().encode(
        color=alt.Color('Correct:N', 
                        scale=alt.Scale(domain=[True, False], range=["#00ffcc", "#ff0000"]),
                        legend=alt.Legend(title="Prediction Correct?")),
        tooltip=['Name', 'Actual', 'Predicted', 'Correct']
    )

    text = base.mark_text().encode(
        text='ValueAsString:N',
        color=alt.value('black')
    )

    chart = (rects + text).properties(
        title="Validation Results",
        width=400
    )

    return chart.to_dict()


class InfluenceRequest(BaseModel):
    row_index: int

@app.post("/get-influence")
def get_influence(req: InfluenceRequest):
    if "influences" not in _data_cache:
        raise HTTPException(status_code=500, detail="Influence model not trained yet")
    
    _data_cache['last_clicked_val_index'] = req.row_index
    
    influences = _data_cache["influences"]
    train_names = _data_cache["train_names"].values
    
    row_influences = influences[req.row_index]
    
    top_pos_indices = np.argsort(row_influences)[-5:]
    top_neg_indices = np.argsort(row_influences)[:5]
    combined_indices = np.concatenate([top_neg_indices, top_pos_indices])

    combined_values = row_influences[combined_indices]
    combined_names = train_names[combined_indices]
    
    df = pd.DataFrame({
        'TrainIndex': combined_indices,
        'Name': combined_names,
        'Influence': combined_values,
        'Type': ['Opponent' if x < 0 else 'Proponent' for x in combined_values]
    })

    base = alt.Chart(df).encode(
        # Sort x-axis by Influence value descending
        x=alt.X('Name:N', sort=alt.EncodingSortField(field="Influence", order="descending"), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Influence:Q'),
        tooltip=['Name', 'Influence', 'TrainIndex', 'Type']
    )

    bars = base.mark_bar().encode(
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Proponent', 'Opponent'], range=['green', 'red']))
    )

    chart = bars.properties(
        title=f"Top Influencers for Validation Point #{req.row_index}",
        width=600,
        height=300
    )
    return chart.to_dict()

class TrainClickRequest(BaseModel):
    train_index: int

@app.post("/log-train-click")
def log_train_click(req: TrainClickRequest):
    if "last_clicked_val_index" not in _data_cache:
         raise HTTPException(status_code=400, detail="No validation player selected.")

    val_idx = _data_cache["last_clicked_val_index"]
    train_idx = req.train_index
    
    val_name = str(_data_cache["val_names"].values[val_idx])
    train_name = str(_data_cache["train_names"].values[train_idx])
    
    scaler = _data_cache["scaler"]
    val_vec_norm = _data_cache["X_val"][val_idx]
    train_vec_norm = _data_cache["X_train"][train_idx]
    
    val_vec_raw = scaler.inverse_transform(val_vec_norm.reshape(1, -1))[0]
    train_vec_raw = scaler.inverse_transform(train_vec_norm.reshape(1, -1))[0]
    
    val_hof = "Yes" if _data_cache["y_val"].iloc[val_idx] == 1 else "No"
    train_hof = "Yes" if _data_cache["y_train"].iloc[train_idx] == 1 else "No"

    stats_data = []
    features = _data_cache["feature_names"]
    
    stats_data.append({"Stat": "HOF Status", "Column": val_name, "Value": val_hof, "IsDiff": False, "IsHOF": True, "NumericDiff": 0})
    stats_data.append({"Stat": "HOF Status", "Column": train_name, "Value": train_hof, "IsDiff": False, "IsHOF": True, "NumericDiff": 0})
    stats_data.append({"Stat": "HOF Status", "Column": "Diff", "Value": "-", "IsDiff": True, "IsHOF": True, "NumericDiff": 0})

    for i, feat in enumerate(features):
        v_val = float(val_vec_raw[i])
        t_val = float(train_vec_raw[i])
        diff = t_val - v_val
        
        stats_data.append({"Stat": feat, "Column": val_name, "Value": f"{v_val:,.1f}", "IsDiff": False, "IsHOF": False, "NumericDiff": 0})
        stats_data.append({"Stat": feat, "Column": train_name, "Value": f"{t_val:,.1f}", "IsDiff": False, "IsHOF": False, "NumericDiff": 0})
        stats_data.append({"Stat": feat, "Column": "Diff", "Value": f"{diff:+,.1f}", "IsDiff": True, "NumericDiff": diff, "IsHOF": False})

    df = pd.DataFrame(stats_data)

    # Altair Chart
    base = alt.Chart(df).encode(
        y=alt.Y('Stat:O', sort=None, title=None),
        x=alt.X('Column:O', title=None, sort=[val_name, train_name, "Diff"])
    )

    # Layer A: Heatmap for Numeric Diffs
    diff_bg = base.mark_rect().encode(
        color=alt.Color('NumericDiff:Q', scale=alt.Scale(scheme="redyellowgreen", domainMid=0), legend=None)
    ).transform_filter(
        (alt.datum.IsDiff == True) & (alt.datum.IsHOF == False)
    )

    # Layer B: Grey for HOF rows
    hof_bg = base.mark_rect().encode(
        color=alt.value('#f0f0f0')
    ).transform_filter(
        alt.datum.IsHOF == True
    )

    # Text Labels
    text = base.mark_text().encode(
        text='Value:N',
        color=alt.value('black')
    )

    # Combine them: Diff Background + HOF Background + Text
    chart = (diff_bg + hof_bg + text).properties(
        title=f"Comparison: {val_name} vs {train_name}",
        width=400
    )

    return chart.to_dict()


async def main():
    result = train_model(TrainRequest(learning_rate=0.001, epochs=20)) 
    print(f"Received result: {result}")
    

if __name__ == "__main__":
    asyncio.run(main())