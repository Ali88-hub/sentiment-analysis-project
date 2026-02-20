FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# Train model during build
RUN mkdir -p models && \
    python src/train.py \
      --data data/sentiments.csv \
      --out models/sentiment.joblib

ENTRYPOINT ["python", "src/predict.py"]
CMD ["I absolutely loved it", "That was awful"]
