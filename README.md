# Multimodal RAG Pipeline with FastAPI Backend

This project implements a Multimodal Retrieval Augmented Generation (RAG) system that can process PDFs containing text, tables, and images. The system uses FastAPI for the backend API and integrates with Google's Gemini model for advanced multimodal processing.

## Features

- PDF Processing with support for:
  - Text extraction and chunking
  - Table structure recognition
  - Image extraction
- Multimodal RAG capabilities using:
  - Google Gemini 2.5 Flash for text and image processing
  - Hugging Face embeddings (BAAI/bge-small-en-v1.5)
  - Chroma vector store for efficient retrieval
- Secure API with JWT authentication
- Structured response handling for texts, tables, and images

## Pipeline Description

The system implements a sophisticated multimodal RAG pipeline that:

1. Processes PDF documents by extracting and separating:
   - Text content with intelligent chunking
   - Tables with structure preservation
   - Images with base64 encoding
2. Generates summaries for each content type using Gemini model
3. Stores content in a multi-vector retrieval system using Chroma
4. Provides context-aware responses using both text and visual elements

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini model)
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manas0109/flowautomate.git
cd flowautomate
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt

```

4. Create a `.env` file in the root directory with the following variables:
```
GOOGLE_API_KEY=your_google_api_key_here
JWT_SECRET_KEY=your_secret_key_here
ADMIN_API_KEY=your_admin_api_key_here
```

## Running the Backend

1. Start the FastAPI server:
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

2. The API will be available at `http://localhost:8000`
3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

1. **Authentication**
   - POST `/auth/login`: Get JWT token using API key

2. **Document Processing**
   - POST `/ingest`: Upload and process PDF documents
   - POST `/query`: Query the processed documents

3. **Health Check**
   - GET `/health`: Check API status

## Usage Example

1. First, authenticate to get a JWT token:
```bash
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"api_key": "your_admin_api_key"}'
```

2. Use the token to upload a PDF:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "Authorization: Bearer your_jwt_token" \
     -F "file=@your_document.pdf"
```

3. Query the processed document:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Authorization: Bearer your_jwt_token" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the main findings in the document?"}'
```

## Security Notes

- The API uses JWT tokens for authentication
- API keys are required for initial authentication
- All sensitive information should be stored in the `.env` file
- The `.env` file should never be committed to version control


