# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "aisummarizer:app", "--host", "0.0.0.0", "--port", "8000"]
