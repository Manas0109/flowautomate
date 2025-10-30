import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import jwt
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
from final_pipeline import MultimodalRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
port = os.getenv("PORT", 8000)
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
API_KEYS = {
    "admin": os.getenv("ADMIN_API_KEY")
}

# Security utilities
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Pydantic models
class LoginRequest(BaseModel):
    api_key: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    message: str
    num_texts: int
    num_tables: int
    num_images: int



# FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="API for multimodal document processing and question answering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
rag_system = MultimodalRAG(GOOGLE_API_KEY)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Multimodal RAG API is running"}

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate with API key and get JWT token.
    """
    if request.api_key not in API_KEYS.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Find username for the API key
    username = None
    for user, key in API_KEYS.items():
        if key == request.api_key:
            username = user
            break
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    print("username",username)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    current_user: str = Depends(verify_token)
):
    """
    Ingest a PDF file and process it for RAG.
    Requires authentication.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process PDF
        num_texts, num_tables, num_images = rag_system.process_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        logger.info(f"Successfully processed PDF by user: {current_user}")
        
        return IngestResponse(
            message="PDF successfully processed and ingested",
            num_texts=num_texts,
            num_tables=num_tables,
            num_images=num_images
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    current_user: str = Depends(verify_token)
):
    """
    Query the processed documents.
    Requires authentication.
    """
    try:
        answer = rag_system.query(request.question)
        
        logger.info(f"Successfully answered query by user: {current_user}")
        
        return QueryResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)