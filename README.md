# 🚑 Smart Emergency Routing System

AI-powered routing system that optimizes emergency vehicle paths under dynamic traffic using:
- **GNN (Graph Neural Networks)**
- **Reinforcement Learning (Q-learning)**
- **Ensemble ML (LSTM + GRU + Random Forest)**

## ✨ Features
- Real-time traffic simulation + optional live API (TomTom)
- Multiple routing modes: Normal, Emergency, GNN-based, RL-based
- Nearest hospital selection
- OSMnx + NetworkX road graph

## 🧠 Architecture
1. Load road graph (OSMnx)
2. Simulate / fetch traffic
3. Predict congestion (LSTM + GRU + RF ensemble)
4. Adjust graph weights (GNN)
5. Route selection (Dijkstra / RL policy)

## 🛠 Tech Stack
Python, Flask, OSMnx, NetworkX, PyTorch (GNN), scikit-learn

## ▶️ Run
```bash
pip install -r requirements.txt
python app.py
