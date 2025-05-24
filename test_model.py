import asyncio
import logging
import torch
from model import PhishingDetector, PhishingScoreFusion
from app import scrape_url
from typing import Dict, List
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingModelTester:
    def __init__(self, model_path: str = 'models/phishing_detector.pt'):
        """Initialize the model tester with trained models."""
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Using untrained model.")
        
        self.detector = PhishingDetector(model_path)
        self.fusion_model = PhishingScoreFusion(
            gnn_dim=4,  # Match input feature dimension
            llm_dim=256,
            hidden_dim=256,
            dropout=0.1,
            temporal_decay=0.1
        )
        
        # Load fusion model state if available
        try:
            checkpoint = torch.load(model_path)
            if 'fusion_state_dict' in checkpoint:
                self.fusion_model.load_state_dict(checkpoint['fusion_state_dict'])
                logger.info("Loaded fusion model state successfully")
        except Exception as e:
            logger.warning(f"Could not load fusion model state: {str(e)}")
            logger.info("Using untrained fusion model")
    
    async def test_url(self, url: str) -> Dict:
        """Test a single URL using the model."""
        try:
            # Get URL metadata using app.py
            metadata = await scrape_url(url, check_threat_intel=True)
            
            # Prepare URL data
            url_data = {
                'url': url,
                'metadata': metadata,
                'verified_at': datetime.now().isoformat(),
                'is_phishing': None  # We don't know the ground truth
            }
            
            # Get GNN output
            gnn_output = self.detector.preprocess_url(url_data)
            
            # Get LLM output (placeholder - replace with actual LLM output)
            llm_output = {
                'embeddings': torch.randn(128, 256),  # Match GNN output dimensions
                'attention_mask': torch.ones(128, 128)
            }
            
            # Prepare metadata for fusion
            fusion_metadata = {
                'timestamps': [url_data['verified_at']]
            }
            
            # Get fusion model prediction
            with torch.no_grad():
                fusion_output = self.fusion_model(gnn_output, llm_output, fusion_metadata)
                final_score = fusion_output['final_score'].item()
            
            # Get GNN-only prediction
            gnn_score = self.detector.predict(url_data)
            
            # Get threat intelligence score if available
            threat_score = None
            if metadata and 'threat_intel' in metadata:
                threat_score = metadata['threat_intel'].get('confidence_score')
            
            # Get security checks if available
            security_checks = None
            if metadata and 'threat_intel' in metadata:
                security_checks = metadata['threat_intel'].get('security_checks')
            
            return {
                'url': url,
                'gnn_score': gnn_score,
                'fusion_score': final_score,
                'threat_score': threat_score,
                'security_checks': security_checks,
                'metadata': metadata,
                'is_phishing': final_score > 0.5,
                'confidence': abs(final_score - 0.5) * 2  # Convert to 0-1 range
            }
            
        except Exception as e:
            logger.error(f"Error testing URL {url}: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def test_urls(self, urls: List[str]) -> List[Dict]:
        """Test multiple URLs using the model."""
        results = []
        for url in urls:
            result = await self.test_url(url)
            results.append(result)
        return results

async def main():
    # Example URLs to test
    test_urls = [
        "https://www.google.com",  # Benign
        "https://www.paypal.com",  # Benign
        "https://paypal-secure-verify.com",  # Likely phishing
        "https://www.microsoft.com",  # Benign
        "https://microsoft-account-verify.com"  # Likely phishing
    ]
    
    # Initialize tester
    tester = PhishingModelTester()
    
    # Test URLs
    results = await tester.test_urls(test_urls)
    
    # Print results
    print("\nTest Results:")
    print("-" * 80)
    for result in results:
        if 'error' in result:
            print(f"\nURL: {result['url']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nURL: {result['url']}")
            print(f"GNN Score: {result['gnn_score']:.4f}")
            print(f"Fusion Score: {result['fusion_score']:.4f}")
            print(f"Prediction: {'Phishing' if result['is_phishing'] else 'Benign'}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            # Print relevant metadata
            if result['metadata']:
                print("\nRelevant Metadata:")
                if 'domain_info' in result['metadata']:
                    print(f"Domain: {result['metadata']['domain_info'].get('domain', 'N/A')}")
                if 'threat_intel' in result['metadata']:
                    print(f"Threat Score: {result['metadata']['threat_intel'].get('score', 'N/A')}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 