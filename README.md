# Sentiment Analysis Pipeline

## Setup

### Option 1: Python venv
```python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Conda
```python
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt
```

## Train
```python
python src/train.py --data data/train.csv --out models/sentiment.joblib
```

## Predict
```python
python src/predict.py "I absolutely loved it" "That was awful"
```
Output format: label  probability  text
Example:
```python
1    0.982    I absolutely loved it
0    0.015    That was awful
```

