FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy your script and dependencies
COPY predict.py .
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run prediction script by default (can be overridden at runtime)
ENTRYPOINT ["python", "predict.py"]
