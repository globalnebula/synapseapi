from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Any
import time
import logging
import requests
import ssl
import socket
import json
from datetime import datetime
import httpx
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
GOOGLE_SAFE_BROWSING_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")

app = FastAPI(title="URL Preprocessing API for Phishing Detection")

# ---- Request & Response Models ----

class URLRequest(BaseModel):
    url: HttpUrl
    check_threat_intel: bool = Field(default=True, description="Whether to check threat intelligence APIs")

class BatchURLRequest(BaseModel):
    urls: List[HttpUrl]
    check_threat_intel: bool = Field(default=True, description="Whether to check threat intelligence APIs")

class SSLCertInfo(BaseModel):
    issuer: Dict[str, str]
    subject: Dict[str, str]
    version: int
    not_before: str
    not_after: str
    serial_number: str
    is_valid: bool

class ThreatIntelInfo(BaseModel):
    is_malicious: bool
    threat_types: List[str]
    confidence_score: float
    last_updated: str
    sources: List[str]
    security_checks: Dict[str, bool]

class URLMetadata(BaseModel):
    original_url: HttpUrl
    final_url: HttpUrl
    status_code: Optional[int]
    load_time: float  # seconds
    page_title: Optional[str]
    text_content: Optional[str]
    headers: Dict[str, str]
    domain_info: Dict[str, str]
    ssl_info: Optional[SSLCertInfo]
    threat_intel: Optional[ThreatIntelInfo]

# ---- SSL Certificate Functions ----

def get_ssl_cert_info(hostname: str) -> Optional[SSLCertInfo]:
    """Get SSL certificate information for a domain."""
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                
                # Convert issuer and subject to dictionaries
                issuer_dict = {}
                if isinstance(cert['issuer'], tuple):
                    for key, value in cert['issuer']:
                        issuer_dict[key] = value
                else:
                    # Handle single-item format
                    issuer_dict = dict(cert['issuer'])
                
                subject_dict = {}
                if isinstance(cert['subject'], tuple):
                    for key, value in cert['subject']:
                        subject_dict[key] = value
                else:
                    # Handle single-item format
                    subject_dict = dict(cert['subject'])
                
                # Parse dates
                not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                
                return SSLCertInfo(
                    issuer=issuer_dict,
                    subject=subject_dict,
                    version=cert['version'],
                    not_before=not_before.isoformat(),
                    not_after=not_after.isoformat(),
                    serial_number=hex(cert['serialNumber']),
                    is_valid=datetime.now() < not_after
                )
    except Exception as e:
        logger.error(f"Error getting SSL cert info: {str(e)}")
        return None

# ---- Threat Intelligence Functions ----

async def check_google_safe_browsing(url: str) -> Dict[str, Any]:
    """Check URL against Google Safe Browsing API."""
    if not GOOGLE_SAFE_BROWSING_API_KEY:
        logger.warning("Google Safe Browsing API key not configured")
        return None
        
    try:
        safe_browsing_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        payload = {
            "client": {
                "clientId": "phishing-detection-api",
                "clientVersion": "1.0.0"
            },
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}]
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{safe_browsing_url}?key={GOOGLE_SAFE_BROWSING_API_KEY}", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Google Safe Browsing API error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error checking Google Safe Browsing: {str(e)}")
        return None

def perform_security_checks(url: str, headers: Dict[str, str], ssl_info: Optional[SSLCertInfo]) -> Dict[str, bool]:
    """Perform various free security checks on the URL."""
    checks = {
        "has_ssl": False,
        "valid_ssl": False,
        "has_security_headers": False,
        "has_content_security_policy": False,
        "has_xss_protection": False,
        "has_frame_protection": False
    }
    
    # Check SSL
    if ssl_info:
        checks["has_ssl"] = True
        checks["valid_ssl"] = ssl_info.is_valid
    
    # Check security headers
    security_headers = {
        "Content-Security-Policy": "has_content_security_policy",
        "Content-Security-Policy-Report-Only": "has_content_security_policy",
        "X-XSS-Protection": "has_xss_protection",
        "X-Frame-Options": "has_frame_protection"
    }
    
    for header, check_key in security_headers.items():
        if header.lower() in [h.lower() for h in headers.keys()]:
            checks[check_key] = True
    
    checks["has_security_headers"] = any(checks[key] for key in ["has_content_security_policy", "has_xss_protection", "has_frame_protection"])
    
    return checks

def parse_threat_intel_results(google_sb_result: Dict, security_checks: Dict[str, bool]) -> ThreatIntelInfo:
    """Parse and combine results from threat intelligence APIs and security checks."""
    threat_types = []
    confidence_score = 0.0
    sources = []
    
    if google_sb_result:
        sources.append("Google Safe Browsing")
        if "matches" in google_sb_result:
            for match in google_sb_result["matches"]:
                threat_types.extend(match.get("threatType", []))
                confidence_score = max(confidence_score, 0.8)
    
    security_score = sum(1 for check in security_checks.values() if check) / len(security_checks)
    
    if not threat_types:
        confidence_score = max(confidence_score, 1.0 - security_score)
    
    return ThreatIntelInfo(
        is_malicious=len(threat_types) > 0 or (not threat_types and security_score < 0.3),
        threat_types=list(set(threat_types)),
        confidence_score=confidence_score,
        last_updated=datetime.now().isoformat(),
        sources=sources,
        security_checks=security_checks
    )

# ---- Core Scraping Function ----

async def scrape_url(url: str, check_threat_intel: bool = True) -> URLMetadata:
    """
    Uses requests to fetch URL and extracts:
    - Final URL (after redirects)
    - HTTP status code, load time
    - Page title and extracted text (via BeautifulSoup)
    - HTTP headers
    - Domain information
    - SSL certificate information
    - Threat intelligence data
    """
    try:
        logger.info(f"Fetching URL: {url}")
        start_time = time.time()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(verify=True) as client:
            response = await client.get(url, headers=headers, timeout=30, follow_redirects=True)
            load_time = time.time() - start_time
            
            final_url = str(response.url)
            status_code = response.status_code
            
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else None
            text_content = soup.get_text(separator='\n', strip=True)
            
            parsed_url = urlparse(final_url)
            domain_info = {
                'domain': parsed_url.netloc,
                'scheme': parsed_url.scheme,
                'path': parsed_url.path,
                'query': parsed_url.query,
                'fragment': parsed_url.fragment
            }
            
            ssl_info = get_ssl_cert_info(parsed_url.netloc)
            
            threat_intel = None
            if check_threat_intel:
                google_sb_result = await check_google_safe_browsing(url)
                security_checks = perform_security_checks(url, dict(response.headers), ssl_info)
                threat_intel = parse_threat_intel_results(google_sb_result, security_checks)
            
            logger.info(f"Successfully processed URL: {final_url}")
            
            return URLMetadata(
                original_url=url,
                final_url=final_url,
                status_code=status_code,
                load_time=load_time,
                page_title=title,
                text_content=text_content,
                headers=dict(response.headers),
                domain_info=domain_info,
                ssl_info=ssl_info,
                threat_intel=threat_intel
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---- API Endpoints ----

@app.post("/preprocess", response_model=URLMetadata)
async def preprocess_url(request: URLRequest):
    """
    Preprocesses the URL by:
      - Fetching the webpage
      - Extracting and returning various metadata required for phishing detection.
      - Checking SSL certificate information
      - Checking threat intelligence databases
    
    The output JSON includes final URL after redirects, page content,
    HTTP headers, domain information, SSL certificate info, and threat intelligence data.
    """
    try:
        logger.info(f"Received request to preprocess URL: {request.url}")
        result = await scrape_url(
            str(request.url),
            request.check_threat_intel
        )
        return result
    except Exception as e:
        logger.error(f"Error in preprocess_url: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/batch", response_model=List[URLMetadata])
async def preprocess_urls_batch(request: BatchURLRequest):
    """
    Preprocesses multiple URLs in parallel.
    """
    try:
        logger.info(f"Received batch request for {len(request.urls)} URLs")
        tasks = [
            scrape_url(
                str(url),
                request.check_threat_intel
            )
            for url in request.urls
        ]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        logger.error(f"Error in preprocess_urls_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preprocess")
async def preprocess_url_get(url: str, check_threat_intel: bool = True):
    """
    GET endpoint for testing URL preprocessing.
    
    Parameters:
    - url: The URL to check (must be URL encoded)
    - check_threat_intel: Whether to check threat intelligence APIs (default: True)
    
    Example:
    /preprocess?url=https%3A%2F%2Fexample.com&check_threat_intel=true
    """
    try:
        logger.info(f"Received GET request to preprocess URL: {url}")
        result = await scrape_url(
            url,
            check_threat_intel
        )
        return result
    except Exception as e:
        logger.error(f"Error in preprocess_url_get: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preprocess/batch")
async def preprocess_urls_batch_get(urls: str, check_threat_intel: bool = True):
    """
    GET endpoint for testing batch URL preprocessing.
    
    Parameters:
    - urls: Comma-separated list of URLs to check (must be URL encoded)
    - check_threat_intel: Whether to check threat intelligence APIs (default: True)
    
    Example:
    /preprocess/batch?urls=https%3A%2F%2Fexample.com%2Chttps%3A%2F%2Fexample.org&check_threat_intel=true
    """
    try:
        url_list = [url.strip() for url in urls.split(",")]
        logger.info(f"Received GET batch request for {len(url_list)} URLs")
        tasks = [
            scrape_url(
                url,
                check_threat_intel
            )
            for url in url_list
        ]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        logger.error(f"Error in preprocess_urls_batch_get: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information and usage examples."""
    return {
        "name": "URL Preprocessing API for Phishing Detection",
        "version": "1.0.0",
        "description": "An API for preprocessing URLs and detecting phishing attempts using security checks and threat intelligence",
        "endpoints": {
            "/preprocess": {
                "methods": ["GET", "POST"],
                "description": "Process a single URL",
                "examples": {
                    "GET": "/preprocess?url=https%3A%2F%2Fexample.com",
                    "POST": "Use Swagger UI or send POST request with JSON body"
                }
            },
            "/preprocess/batch": {
                "methods": ["GET", "POST"],
                "description": "Process multiple URLs in parallel",
                "examples": {
                    "GET": "/preprocess/batch?urls=https%3A%2F%2Fexample.com%2Chttps%3A%2F%2Fexample.org",
                    "POST": "Use Swagger UI or send POST request with JSON body"
                }
            },
            "/docs": "API documentation (Swagger UI)"
        },
        "usage": {
            "single_url": {
                "GET": "curl 'http://localhost:8000/preprocess?url=https%3A%2F%2Fexample.com'",
                "POST": "curl -X POST 'http://localhost:8000/preprocess' -H 'Content-Type: application/json' -d '{\"url\":\"https://example.com\"}'"
            },
            "batch_urls": {
                "GET": "curl 'http://localhost:8000/preprocess/batch?urls=https%3A%2F%2Fexample.com%2Chttps%3A%2F%2Fexample.org'",
                "POST": "curl -X POST 'http://localhost:8000/preprocess/batch' -H 'Content-Type: application/json' -d '{\"urls\":[\"https://example.com\",\"https://example.org\"]}'"
            }
        }
    }