# ai-practice
FastAPI OpenAI Summarizer
This project is a lightweight web service built with FastAPI that uses the OpenAI API to summarize text. It provides a simple API endpoint to send a block of text and receive a three-bullet-point summary in response.

Features
FastAPI Framework: A modern, high-performance web framework for building APIs.

OpenAI Integration: Communicates with the OpenAI API for text summarization.

Single-Endpoint Functionality: A dedicated endpoint for chat-based summarization.

Structured Request/Response: Uses Pydantic models for clear and validated API data.

Requirements
To run this project, you need to have Python 3.8 or higher installed. The dependencies are managed using pip and are listed in the requirements.txt file.

Setup and Installation
Clone the repository:

git clone https://github.com/rumanhassan/ai-practice.git
cd your-repository-name

Create and activate a virtual environment:
It is highly recommended to use a virtual environment to manage dependencies for your project.

# Create the virtual environment
`python -m venv venv`

# Activate the virtual environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

Install dependencies:
Use pip to install the required packages from the requirements.txt file.

`pip install -r requirements.txt`

Set up your OpenAI API key:
The application requires an OpenAI API key to make calls to the OpenAI service. Store your key as an environment variable named OPENAI_API_KEY.

# On macOS and Linux:
`export OPENAI_API_KEY="YOUR_API_KEY_HERE"`

# On Windows (Command Prompt):
`set OPENAI_API_KEY="YOUR_API_KEY_HERE"`

Run the application:
Start the FastAPI server.

fastapi dev aisummarizer.py

The API will be available at http://127.0.0.1:8000.

Deployment with Docker

1. Build the Docker Image
You will first need to create a Dockerfile to build your application image.

Dockerfile:
```
# Use a slim Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "aisummarizer:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build the image from the project's root directory:

`docker build -t fastapi-app .`


2. Run the Docker Container
To run the container, you need to pass the API key as an environment variable. Since your key is already in your .zshrc file, you can reference it directly.

`docker run -d -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY fastapi-app`


-d: Runs the container in detached mode (in the background).

-p 8000:8000: Publishes the container's port to your local machine.

-e OPENAI_API_KEY=$OPENAI_API_KEY: Passes your API key from your shell's environment variable into the container.