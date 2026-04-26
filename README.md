# 🚑 Smart Emergency Routing System

An AI-powered emergency vehicle routing system that optimizes paths in real time under dynamic traffic conditions using Graph Neural Networks, Reinforcement Learning, and Ensemble Machine Learning.

---

## 🖥️ Demo

> Real-time routing on live road graphs with traffic-aware path selection and nearest hospital detection.

---

## ✨ Features

- 🗺️ **Live Road Graph** — Loads real city road networks using OSMnx + NetworkX
- 🚦 **Traffic Simulation** — Real-time traffic simulation with optional TomTom live API integration
- 🤖 **Congestion Prediction** — LSTM + GRU + Random Forest ensemble predicts traffic congestion
- 🧠 **GNN Weight Adjustment** — Graph Neural Network dynamically adjusts road weights based on predicted congestion
- 🚗 **Multiple Routing Modes** — Normal, Emergency, GNN-based, and RL-based routing
- 🏥 **Hospital Finder** — Automatically selects the nearest available hospital
- 📊 **RL Policy** — Q-learning agent learns optimal routing strategies over time

---

## 🧠 System Architecture

```
1. Load Road Graph        →  OSMnx fetches real city road network
2. Traffic Layer          →  Simulate or fetch live traffic data
3. Congestion Prediction  →  LSTM + GRU + Random Forest ensemble
4. GNN Adjustment         →  Reweight graph edges based on congestion
5. Route Selection        →  Dijkstra (normal) / RL Policy (emergency)
6. Output                 →  Optimal path + nearest hospital displayed
```

---

## 🛠️ Tech Stack

| Component            | Technology                          |
|----------------------|-------------------------------------|
| Language             | Python 3.x                          |
| Web Framework        | Flask                               |
| Road Network         | OSMnx, NetworkX                     |
| Deep Learning        | PyTorch (GNN, LSTM, GRU)            |
| Machine Learning     | scikit-learn (Random Forest)        |
| Reinforcement Learning | Q-Learning (custom implementation)|
| Live Traffic API     | TomTom Traffic API (optional)       |
| Frontend             | HTML, CSS, JavaScript               |

---

## 📁 Project Structure

```
Smart-Emergency-Routing-System/
├── app.py               # Flask web server and API endpoints
├── routing.py           # Core routing logic (Dijkstra + RL)
├── hospital.py          # Nearest hospital selection
├── live_traffic.py      # Live traffic API integration
├── map_loader.py        # OSMnx road graph loader
├── utils.py             # Helper utilities
├── gnn/                 # Graph Neural Network model
├── rl/                  # Reinforcement Learning (Q-learning) agent
├── models/              # Trained ML models
├── simulation/          # Traffic simulation engine
├── training/            # Model training scripts
├── templates/           # HTML frontend templates
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/chehak655/Smart-Emergency-Routing-System.git
cd Smart-Emergency-Routing-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure TomTom API
To use live traffic data, add your TomTom API key in `live_traffic.py`:
```python
API_KEY = "your_tomtom_api_key"
```

### 4. Run the application
```bash
python app.py
```

Then open your browser at `http://localhost:5000`

---

## 🔬 ML Models Used

| Model         | Purpose                              |
|---------------|--------------------------------------|
| LSTM          | Sequential traffic pattern learning  |
| GRU           | Short-term congestion prediction     |
| Random Forest | Ensemble congestion classification   |
| GNN           | Dynamic road graph weight adjustment |
| Q-Learning    | RL-based optimal route policy        |

---

## 👩‍💻 Author

**Chehak** — [github.com/chehak655](https://github.com/chehak655)
**Sakshi** - [github.com/sakshisingh3141](https://github.com/sakshisingh3141)
