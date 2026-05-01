# Smart Plant Watering Scheduler 🌿

An AI-based garden management system built entirely in pure Python 
(no external ML libraries) that predicts plant watering needs and 
optimizes the watering sequence.

## Features
- **Perceptron Classifier** — trained from scratch to predict whether 
  each plant needs water based on soil moisture, last watered time, 
  and plant type
- **Simulated Annealing Optimizer** — finds the most efficient watering 
  order minimizing total distance walked and missed plants
- **Interactive GUI** — place plants on a garden map, visualize 
  predictions, and watch the SA optimization step by step
- **Excel Export** — save full garden data with watering order to .xlsx

## Tech Stack
Python · Tkinter · Pure stdlib only (no numpy, pandas, or sklearn)

## How to Run
1. Clone the repo
2. Place `Data.xlsx` in the project folder
3. Run: python plant_watering_scheduler.py

## Project Structure
```
plant_project/
├── plant_watering_scheduler.py   # main application
├── Data.xlsx                     # training data
└── README.md
```

## Dataset
100 labeled plant samples with features:
- `soil_moisture` (0–100)
- `last_watered` (hours, 0–48)  
- `plant_type` (0=Cactus, 1=Flower, 2=Herb)
- `needs_water` (0 or 1)

## Algorithm Details

**Perceptron:**
- 3 input features → binary output
- Custom learning rule: `w = w + lr × (y - ŷ) × x`
- Trained for 50 epochs with learning rate 0.1

**Simulated Annealing:**
- Cost = `plants_missed + total_distance + extra_watering`
- Swap-based neighborhood search
- Geometric cooling: `T = T × cooling_rate`
