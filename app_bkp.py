import asyncio
import sys
import platform

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import time
import base64
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="URL Preprocessing API for Phishing Detection")

# ---- Request & Response Models ----

class URLRequest(BaseModel):
    url: HttpUrl

class URLMetadata(BaseModel):
    original_url: HttpUrl
    final_url: HttpUrl
    status_code: Optional[int]
    load_time: float  # seconds
    page_title: Optional[str]
    text_content: Optional[str]
    headers: Dict[str, str]
    domain_info: Dict[str, str]

# ---- Core Scraping Function ----

def scrape_url(url: str) -> URLMetadata:
    """
    Uses requests to fetch URL and extracts:
    - Final URL (after redirects)
    - HTTP status code, load time
    - Page title and extracted text (via BeautifulSoup)
    - HTTP headers
    - Domain information
    """
    try:
        logger.info(f"Fetching URL: {url}")
        start_time = time.time()
        
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request with a timeout
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        load_time = time.time() - start_time
        
        # Get the final URL after redirects
        final_url = response.url
        status_code = response.status_code
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else None
        
        # Extract text content
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Get domain information
        parsed_url = urlparse(final_url)
        domain_info = {
            'domain': parsed_url.netloc,
            'scheme': parsed_url.scheme,
            'path': parsed_url.path,
            'query': parsed_url.query,
            'fragment': parsed_url.fragment
        }
        
        logger.info(f"Successfully processed URL: {final_url}")
        
        return URLMetadata(
            original_url=url,
            final_url=final_url,
            status_code=status_code,
            load_time=load_time,
            page_title=title,
            text_content=text_content,
            headers=dict(response.headers),
            domain_info=domain_info
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---- API Endpoints ----

@app.post("/preprocess", response_model=URLMetadata)
def preprocess_url(request: URLRequest):
    """
    Preprocesses the URL by:
      - Fetching the webpage
      - Extracting and returning various metadata required for phishing detection.
    
    The output JSON includes final URL after redirects, page content,
    HTTP headers, domain information, and load time.
    """
    try:
        logger.info(f"Received request to preprocess URL: {request.url}")
        result = scrape_url(str(request.url))
        return result
    except Exception as e:
        logger.error(f"Error in preprocess_url: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
