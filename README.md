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
python -m venv venv

# Activate the virtual environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

Install dependencies:
Use pip to install the required packages from the requirements.txt file.

pip install -r requirements.txt

Set up your OpenAI API key:
The application requires an OpenAI API key to make calls to the OpenAI service. Store your key as an environment variable named OPENAI_API_KEY.

# On macOS and Linux:
export OPENAI_API_KEY="YOUR_API_KEY_HERE"

# On Windows (Command Prompt):
set OPENAI_API_KEY="YOUR_API_KEY_HERE"

Run the application:
Start the FastAPI server.

fastapi dev aisummarizer.py

The API will be available at http://127.0.0.1:8000.