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

app = FastAPI()

# Allow your browser to make requests to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    X = df.drop(columns=features_to_drop, errors='ignore')
    X = X.fillna(0)
    y = df[target]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get Names corresponding to the split (Preserve indices initially)
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
    
    return _data_cache

@app.post("/train-and-visualize")
def train_model(req: TrainRequest):
    data = get_processed_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    train_dataset = QBDataSet(X_train, y_train.values)
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

    val_df = pd.DataFrame({
        'Name': data["val_names"].values, 
        'Actual': y_val.values,
        'Predicted': y_pred
    })
    
    # 1. Create the base DataFrame
    val_df = pd.DataFrame({
        'Name': data["val_names"].values, 
        'Actual': y_val.values,
        'Predicted': y_pred
    })
    
    val_df['RowIndex'] = val_df.index
    val_df['Correct'] = val_df['Actual'] == val_df['Predicted']

    # 2. MELT (Simplified)
    # Only keep 'RowIndex' as the identifier to avoid the overlap bug.
    melted_df = val_df.melt(
        id_vars=['RowIndex'], 
        value_vars=['Name', 'Actual', 'Predicted'],
        var_name='ColumnName',
        value_name='ValueAsString' 
    )

    # 3. MERGE (The "Vlookup" Step)
    # We bring the original context (Correct, Name, Actual, Predicted) back 
    # by joining on 'RowIndex'.
    final_df = pd.merge(melted_df, val_df, on='RowIndex')

    # Debug Prints
    print(f"Rows in val_df: {len(val_df)}")
    print(f"Rows in final_df: {len(final_df)}") # Should be 37 * 3 = 111

    # 4. Create Chart
    alt.data_transformers.disable_max_rows()

    base = alt.Chart(final_df).encode(
        y=alt.Y('RowIndex:O', title=None),
        # We use the new merged columns for sorting
        x=alt.X('ColumnName:O', title=None, sort=['Name', 'Actual', 'Predicted'])
    )

    rects = base.mark_rect().encode(
        # We can now access 'Correct' safely because we merged it back in
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


@app.post("/log-interaction")
async def log_interaction(interaction: InteractionRequest):
    print(f"User clicked cell: Train={interaction.train_id}, Test={interaction.test_id}, Val={interaction.influence:.4f}")
    return {"status": "logged"}


async def main():
    # Call and wait for the result
    result = train_model(TrainRequest(learning_rate=0.001, epochs=20)) 
    print(f"Received result: {result}")
    

if __name__ == "__main__":
    asyncio.run(main())