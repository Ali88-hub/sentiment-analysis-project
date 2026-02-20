# Layer 1: Use an official Python image as a base image
FROM python:3.11-slim

# Layer 2: Set the working directory inside the container
WORKDIR /app

# Layers 3 & 4: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layers 5 & 6: Copy local project files into the container
COPY src/ src/
COPY data/ data/

# Layer 7: Train the model during build (creates models/)
RUN python src/train.py \
    --data data/sentiments.csv \
    --out models/sentiment.joblib

# Layer 8: Fixed command (never changes)
ENTRYPOINT ["python", "src/predict.py"]

# Layer 9: Default arguments
CMD ["I absolutely loved it", "That was awful"]
