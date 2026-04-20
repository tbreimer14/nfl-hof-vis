import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import shap
from pydvl.influence.torch import DirectInfluence
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Dict, Optional
from scipy.special import expit
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Mount static files (like your CSS or JS if they are in a folder)
# app.mount("/static", StaticFiles(directory="static"), name="static")

app = FastAPI()

# Mount static files (like your CSS or JS if they are in a folder)
#app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_data = {}

class WhatIfRequest(BaseModel):
    overrides: Dict[str, float] = {}

class TorchLogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)

@app.on_event("startup")
def load_and_train():
    print("Loading data and training models...")
    df = pd.read_csv('data/data.csv')
    
    p_cols = [c for c in df.columns if c.startswith('playoff_')]
    df[p_cols] = df[p_cols].fillna(0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    X = df.drop(columns=['Player', 'HOF'])
    y = df['HOF']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)
    
    sklearn_model = LogisticRegression(C=1.0, random_state=57, max_iter=1000)
    sklearn_model.fit(X_train_scaled, y_train)
    
    explainer = shap.LinearExplainer(sklearn_model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_values.data = X_test.values 

    torch_model = TorchLogisticRegression(X_train_scaled.shape[1])
    with torch.no_grad():
        torch_model.linear.weight.copy_(torch.tensor(sklearn_model.coef_, dtype=torch.float32))
        torch_model.linear.bias.copy_(torch.tensor(sklearn_model.intercept_, dtype=torch.float32))

    X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    influence_model = DirectInfluence(model=torch_model, loss=F.binary_cross_entropy_with_logits, regularization=1.0)
    influence_model.fit(train_loader)
    influence_matrix_tensor = influence_model.influences(X_test_t, y_test_t, X_train_t, y_train_t)

    # Save data for endpoints
    model_data['X_train'] = X_train  # Needed for scatter/KDE raw feature values
    model_data['y_train'] = y_train
    model_data['X_test'] = X_test
    model_data['y_test'] = y_test
    model_data['test_players'] = df.loc[X_test.index, 'Player'].values
    model_data['train_players'] = df.loc[X_train.index, 'Player'].values
    model_data['predictions'] = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    model_data['shap_values'] = shap_values
    model_data['influence_matrix'] = influence_matrix_tensor.numpy()
    model_data['feature_names'] = X_train.columns.tolist()

    # Save objects needed for on-the-fly SHAP recalculation
    model_data['scaler'] = scaler
    model_data['explainer'] = explainer
    
    # Precompute feature ranges so the frontend sliders know their min/max
    ranges = {}
    for col in X_train.columns:
        ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "step": 0.1 if X_train[col].dtype == float else 1.0
        }
    model_data['feature_ranges'] = ranges

    print("Startup complete. Ready to serve requests.")

@app.get("/api/test_results")
def get_test_results():
    results = []
    for i, player in enumerate(model_data['test_players']):
        results.append({
            "index": i,
            "player": player,
            "actual": int(model_data['y_test'].iloc[i]),
            "predicted": round(float(model_data['predictions'][i]), 3)
        })
    return results

@app.post("/api/shap/{player_index}")
def get_shap_values(player_index: int, req: WhatIfRequest):
    raw_stats = model_data['X_test'].iloc[player_index].to_dict()
    for f, v in req.overrides.items():
        if f in raw_stats:
            raw_stats[f] = v
            
    df_raw = pd.DataFrame([raw_stats])
    scaled_stats = model_data['scaler'].transform(df_raw)
    shap_vals = model_data['explainer'](scaled_stats).values[0]
    expected_value = float(model_data['explainer'].expected_value)
    if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
        
    features = model_data['feature_names']
    
    items = list(zip(features, shap_vals, df_raw.iloc[0].values))
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Split into Top 10 and "The Rest"
    top_items = items[:10]
    other_items = items[10:] 
    
    top_features = [x[0] for x in top_items]
    all_features = [x[0] for x in items]
    
    # --- PROBABILITY CONVERSION FOR WATERFALL ---
    waterfall_data = []
    current_log_odds = expected_value
    
    # Base Value anchor
    base_prob = float(expit(expected_value))
    waterfall_data.append({
        "feature": "Base Value",
        "start": 0, "end": base_prob,
        "color": "#888", "is_base": 1, "order": 0,
        "arrow": "" # No arrow needed for the baseline
    })
    
    # 1. Add the Top 10 Features
    for i, (f, sv, raw_val) in enumerate(top_items):
        start_prob = float(expit(current_log_odds))
        current_log_odds += sv
        end_prob = float(expit(current_log_odds))
        
        waterfall_data.append({
            "feature": f"{f} ({raw_val:.1f})",
            "start": start_prob,
            "end": end_prob,
            "color": "#d4af37" if sv > 0 else "#999999",
            "arrow": "▶" if sv > 0 else ("◀" if sv < 0 else ""),
            "is_base": 0, "order": i + 1
        })
        
    # 2. Catch-all for "Other Features" 
    if other_items:
        start_prob = float(expit(current_log_odds))
        other_sv_sum = sum(x[1] for x in other_items)
        current_log_odds += other_sv_sum
        end_prob = float(expit(current_log_odds))
        
        waterfall_data.append({
            "feature": f"Other {len(other_items)} Features",
            "start": start_prob,
            "end": end_prob,
            "color": "#d4af37" if other_sv_sum > 0 else "#999999",
            "arrow": "▶" if other_sv_sum > 0 else ("◀" if other_sv_sum < 0 else ""),
            "is_base": 0, "order": len(top_items) + 1
        })
        
    init_probability = float(model_data['predictions'][player_index])
    
    # Now this will perfectly match the true model output!
    final_probability = float(expit(current_log_odds))
    
    # Build Global Strip Plot Data (Kept in Log-Odds for visual accuracy)
    top_indices = [features.index(f) for f in top_features]
    global_shap = model_data['shap_values'].values[:, top_indices]
    
    strip_data = []
    for i in range(global_shap.shape[0]):
        for j, f in enumerate(top_features):
            strip_data.append({
                "feature": f,
                "shap_value": float(global_shap[i, j]),
                "type": "background"
            })
            
    for j, f in enumerate(top_features):
        strip_data.append({
            "feature": f,
            "shap_value": float(shap_vals[features.index(f)]),
            "type": "current_player"
        })

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": f"SHAP What-If Analysis for {model_data['test_players'][player_index]}",
        "vconcat": [
            {
                "title": "Waterfall: Cumulative Feature Impact (Probability)",
                "width": 600, "height": 250,
                "data": {"values": waterfall_data},
                # Calculate the exact center of the bar for perfect text placement
                "transform": [
                    {"calculate": "(datum.start + datum.end) / 2", "as": "midpoint"}
                ],
                # Y-axis applies to both layers, so we put it at the parent level
                "encoding": {
                    "y": {"field": "feature", "type": "nominal", "sort": {"field": "order"}, "title": None}
                },
                "layer": [
                    {
                        # Layer 1: The colored bars
                        "mark": {"type": "bar", "tooltip": True},
                        "encoding": {
                            "x": {
                                "field": "start", "type": "quantitative", 
                                "title": "Prediction Probability",
                                "axis": {"format": "%"}
                            },
                            "x2": {"field": "end"},
                            "color": {"field": "color", "type": "nominal", "scale": None}
                        }
                    },
                    {
                        # Layer 2: The white directional arrows
                        "mark": {
                            "type": "text", 
                            "align": "center", 
                            "baseline": "middle", 
                            "color": "white", 
                            "fontWeight": "bold", 
                            "fontSize": 14
                        },
                        "encoding": {
                            "x": {"field": "midpoint", "type": "quantitative"},
                            "text": {"field": "arrow", "type": "nominal"}
                        }
                    }
                ]
            },
            {
                "title": "Global Distribution (Top 10 Features - Log Odds)",
                "width": 600, "height": 250,
                "data": {"values": strip_data},
                "transform": [{"calculate": "random()", "as": "jitter"}],
                "layer": [
                    {
                        "transform": [{"filter": "datum.type == 'background'"}],
                        "mark": {"type": "circle", "opacity": 0.15, "size": 60},
                        "encoding": {
                            "y": {"field": "feature", "type": "nominal", "sort": top_features, "title": None},
                            "x": {"field": "shap_value", "type": "quantitative", "title": "SHAP Value (Log-Odds Impact)"},
                            "yOffset": {"field": "jitter", "type": "quantitative"},
                            "color": {"value": "#1f77b4"}
                        }
                    },
                    {
                        "transform": [{"filter": "datum.type == 'current_player'"}],
                        "mark": {"type": "point", "filled": True, "size": 150, "stroke": "black", "strokeWidth": 1.5},
                        "encoding": {
                            "y": {"field": "feature", "type": "nominal", "sort": top_features},
                            "x": {"field": "shap_value", "type": "quantitative"},
                            "color": {"value": "#ff7f0e"}
                        }
                    }
                ]
            }
        ],
        "config": {"axis": {"labelFontSize": 11, "titleFontSize": 13}}
    }
    
    return {
        "spec": vega_spec, 
        "ranges": model_data['feature_ranges'],
        "current_stats": raw_stats,
        "top_features": top_features,
        "all_features": all_features,
        "final_probability": final_probability,
        "initial_probability": init_probability
    }

@app.get("/api/influence/{player_index}")
def get_influence_values(player_index: int):
    # --- 1. INTERPRETATION FIX ---
    raw_influences = model_data['influence_matrix'][player_index]
    test_actual = float(model_data['y_test'].iloc[player_index])
    
    # If test point is 0, a positive influence (loss reducer) pushed the prediction down.
    # Negating it ensures positive ALWAYS means "Pushes towards HOF (1)".
    if test_actual == 0.0:
        influences = -raw_influences
    else:
        influences = raw_influences

    X_tr = model_data['X_train']
    y_tr = model_data['y_train']
    train_players = model_data['train_players']
    features = model_data['feature_names']
    
    inf_with_index = [(i, inf) for i, inf in enumerate(influences)]
    inf_with_index.sort(key=lambda x: abs(x[1]), reverse=True)
    top_15_indices = set([idx for idx, _ in inf_with_index[:15]])
    
    data = []
    for i, inf in enumerate(influences):
        is_hof = "HOF" if float(y_tr.iloc[i]) == 1.0 else "Not HOF"
        row = {
            "train_player": train_players[i],
            "influence": float(inf),
            "is_top_15": i in top_15_indices,
            "HOF_Status": is_hof,
            "Influence_Type": "Proponent" if inf > 0 else "Opponent"
        }
        for f in features:
            row[f] = float(X_tr.iloc[i][f])
        data.append(row)

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data},
        "params": [
            {
                "name": "SelectedFeature", "value": features[0],
                "bind": {"input": "select", "options": features, "name": "Feature (X-Axis/Dist): "}
            },
            {
                "name": "PlayerSelection",
                "select": {"type": "point", "fields": ["train_player"]},
                # 1. EXPLICIT ROUTING: Only attach click listeners to these two views
                "views": ["bar_view", "scatter_view"] 
            }
        ],
        "transform": [
            {"calculate": "datum[SelectedFeature]", "as": "DynamicFeature"}
        ],
        "vconcat": [
            {
                # 2. NAME TAG: Identify this as the bar_view
                "name": "bar_view", 
                "title": "Top 15 Most Influential Training Instances",
                "width": 800, "height": 200,
                "transform": [{"filter": "datum.is_top_15"}],
                "mark": {"type": "bar", "cursor": "pointer"},
                "encoding": {
                    "x": {"field": "influence", "type": "quantitative", "title": "Influence (Negative = Towards 0, Positive = Towards 1)"},
                    "y": {"field": "train_player", "type": "nominal", "sort": "-x", "title": "Training Player"},
                    "color": {
                        "condition": {"param": "PlayerSelection", "empty": False, "value": "#e83e8c"},
                        "field": "HOF_Status", "type": "nominal", "title": "HOF Status",
                        "scale": {"domain": ["HOF", "Not HOF"], "range": ["#d4af37", "#999999"]},
                        "legend": {"orient": "right", "title": "True Label"}
                    },
                    "opacity": {"condition": {"param": "PlayerSelection", "empty": True, "value": 1.0}, "value": 0.2}
                }
            },
            {
                "hconcat": [
                    {
                        # 3. NAME TAG: Identify this as the scatter_view
                        "name": "scatter_view", 
                        "title": "Influence Distribution across Feature",
                        "width": 350, "height": 300,
                        "mark": {"type": "circle", "size": 120, "stroke": "white", "strokeWidth": 0.5, "cursor": "pointer"},
                        "encoding": {
                            "x": {"field": "DynamicFeature", "type": "quantitative", "axis": {"title": {"expr": "SelectedFeature"}}},
                            "y": {"field": "influence", "type": "quantitative", "title": "Influence"},
                            "color": {
                                "condition": {"param": "PlayerSelection", "empty": False, "value": "#e83e8c"},
                                "field": "HOF_Status", "type": "nominal",
                                "scale": {"domain": ["HOF", "Not HOF"], "range": ["#d4af37", "#999999"]},
                                "legend": None
                            },
                            "opacity": {"condition": {"param": "PlayerSelection", "empty": True, "value": 0.8}, "value": 0.1},
                            "tooltip": [
                                {"field": "train_player", "title": "Player"}, 
                                {"field": "influence", "title": "Influence", "format": ".4f"}, 
                                {"field": "DynamicFeature", "title": {"expr": "SelectedFeature"}},
                                {"field": "HOF_Status", "title": "Actual Status"}
                            ]
                        }
                    },
                    {
                        # 4. NO NAME TAG: The boxplot is skipped, avoiding the duplicate signal crash!
                        "title": "Feature Distribution (HOF Status)",
                        "width": 350, "height": 300,
                        "mark": {"type": "boxplot", "extent": "min-max"},
                        "encoding": {
                            "x": {"field": "DynamicFeature", "type": "quantitative", "axis": {"title": {"expr": "SelectedFeature"}}},
                            "y": {"field": "HOF_Status", "type": "nominal", "title": "HOF Status"},
                            "color": {
                                "field": "HOF_Status", "type": "nominal", "legend": None,
                                "scale": {"domain": ["HOF", "Not HOF"], "range": ["#d4af37", "#999999"]}
                            }
                        }
                    }
                ]
            }
        ],
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 14}, "title": {"fontSize": 16, "anchor": "middle"}}
    }
    

    return vega_spec

@app.get("/api/compare/{player_index}/{train_player_name}")
def compare_players(player_index: int, train_player_name: str):
    X_test = model_data['X_test']
    X_train = model_data['X_train']
    test_players = model_data['test_players']
    train_players = model_data['train_players']
    features = model_data['feature_names']

    # 1. Get the Test Player's name
    test_player_name = test_players[player_index]

    # 2. Find the index of the Training Player
    try:
        # train_players is a numpy array, we convert to list to find the index easily
        train_index = list(train_players).index(train_player_name)
    except ValueError:
        return {"error": f"Training player {train_player_name} not found."}

    # 3. Extract the raw (unscaled) stats for both players
    test_stats = X_test.iloc[player_index]
    train_stats = X_train.iloc[train_index]

    # 4. Build the comparison payload
    comparison_data = []
    for f in features:
        test_val = float(test_stats[f])
        train_val = float(train_stats[f])
        diff = train_val - test_val
        
        comparison_data.append({
            "feature": f,
            "test_val": test_val,
            "train_val": train_val,
            "diff": diff
        })

    return {
        "test_player": test_player_name,
        "train_player": train_player_name,
        "stats": comparison_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)