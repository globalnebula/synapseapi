import json
import logging
import os
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import PhishingDetector
import pandas as pd
import requests
from tqdm import tqdm
import networkx as nx
from datetime import datetime
import asyncio
from app import scrape_url
import csv
from urllib.parse import urlparse
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

class PhishingDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_phishtank_public_data() -> List[Dict]:
    """Load phishing URLs from PhishTank's public dataset."""
    logger.info("Loading data from PhishTank public dataset...")
    
    # PhishTank public dataset URL
    url = "https://data.phishtank.com/data/online-valid.csv"
    
    try:
        # Download the CSV file
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse CSV data
        csv_data = response.text.splitlines()
        reader = csv.DictReader(csv_data)
        
        # Process data
        processed_data = []
        for entry in reader:
            processed_data.append({
                'url': entry['url'],
                'is_phishing': True,
                'verified_at': entry['submission_time'],
                'target': entry.get('target', 'unknown')
            })
        
        logger.info(f"Loaded {len(processed_data)} phishing URLs from PhishTank public dataset")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error loading PhishTank public data: {str(e)}")
        return []

def load_phishtank_public_json() -> List[Dict]:
    """Load phishing URLs from PhishTank's public JSON dataset."""
    logger.info("Loading data from PhishTank public JSON dataset...")
    
    # PhishTank public JSON dataset URL
    url = "https://data.phishtank.com/data/online-valid.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Process data
        processed_data = []
        for entry in data:
            processed_data.append({
                'url': entry['url'],
                'is_phishing': True,
                'verified_at': entry['submission_time'],
                'target': entry.get('target', 'unknown')
            })
        
        logger.info(f"Loaded {len(processed_data)} phishing URLs from PhishTank public JSON dataset")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error loading PhishTank public JSON data: {str(e)}")
        return []

def load_benign_urls(limit: int = 1000) -> List[Dict]:
    """Load benign URLs from various sources."""
    logger.info("Loading benign URLs...")
    
    # List of trusted domains to sample from
    trusted_domains = [
        'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
        'facebook.com', 'twitter.com', 'linkedin.com', 'github.com',
        'wikipedia.org', 'reddit.com', 'youtube.com', 'netflix.com',
        'spotify.com', 'dropbox.com', 'slack.com', 'zoom.us',
        'mozilla.org', 'adobe.com', 'wordpress.com', 'medium.com',
        'quora.com', 'stackoverflow.com', 'gitlab.com', 'bitbucket.org',
        'atlassian.com', 'trello.com', 'notion.so', 'figma.com',
        'behance.net', 'dribbble.com', 'producthunt.com', 'techcrunch.com'
    ]
    
    # Create URLs with different paths and parameters
    paths = ['', '/about', '/contact', '/products', '/services', '/blog', 
             '/help', '/support', '/docs', '/api', '/download', '/pricing']
    params = ['', '?ref=home', '?source=main', '?lang=en', '?utm_source=direct']
    
    benign_data = []
    
    for domain in trusted_domains:
        for path in paths:
            for param in params:
                url = f"https://{domain}{path}{param}"
                benign_data.append({
                    'url': url,
                    'is_phishing': False,
                    'verified_at': datetime.now().isoformat(),
                    'target': 'benign'
                })
    
    # Add some subdomains
    subdomains = ['www', 'blog', 'docs', 'api', 'support', 'help', 'store']
    for domain in trusted_domains:
        for subdomain in subdomains:
            url = f"https://{subdomain}.{domain}"
            benign_data.append({
                'url': url,
                'is_phishing': False,
                'verified_at': datetime.now().isoformat(),
                'target': 'benign'
            })
    
    # Limit the data
    benign_data = benign_data[:limit]
    logger.info(f"Loaded {len(benign_data)} benign URLs")
    return benign_data

def generate_synthetic_phishing_urls(limit: int = 500) -> List[Dict]:
    """Generate synthetic phishing URLs for additional training data."""
    logger.info("Generating synthetic phishing URLs...")
    
    # Common phishing targets
    targets = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 
              'netflix', 'spotify', 'bank', 'pay', 'login', 'verify']
    
    # Common TLDs used in phishing
    tlds = ['.com', '.net', '.org', '.info', '.xyz', '.online', '.site', '.web']
    
    # Common subdomains used in phishing
    subdomains = ['secure', 'account', 'login', 'verify', 'update', 'support', 
                 'service', 'help', 'customer', 'payment']
    
    synthetic_data = []
    
    for _ in range(limit):
        # Generate random components
        target = random.choice(targets)
        tld = random.choice(tlds)
        subdomain = random.choice(subdomains)
        
        # Generate random string for domain
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        
        # Create URL
        url = f"https://{subdomain}-{target}-{random_str}{tld}"
        
        synthetic_data.append({
            'url': url,
            'is_phishing': True,
            'verified_at': datetime.now().isoformat(),
            'target': target
        })
    
    logger.info(f"Generated {len(synthetic_data)} synthetic phishing URLs")
    return synthetic_data

async def process_urls(urls: List[Dict]) -> List[Dict]:
    """Process URLs to get metadata."""
    logger.info("Processing URLs to get metadata...")
    processed_data = []
    
    for url_data in tqdm(urls):
        try:
            # Get URL metadata
            metadata = await scrape_url(url_data['url'], check_threat_intel=True, use_graph_model=False)
            
            # Add to processed data
            processed_data.append({
                **url_data,
                'metadata': metadata
            })
            
        except Exception as e:
            logger.error(f"Error processing URL {url_data['url']}: {str(e)}")
    
    return processed_data

def build_knowledge_graph(data: List[Dict], output_path: str):
    """Build and save the knowledge graph."""
    logger.info("Building knowledge graph...")
    graph = nx.MultiDiGraph()
    
    # Add domain connections
    for entry in data:
        if 'metadata' in entry and 'domain_info' in entry['metadata']:
            domain = entry['metadata']['domain_info']['domain']
            
            # Add domain node if not exists
            if domain not in graph:
                graph.add_node(domain, type='domain')
            
            # Add connections based on target
            if entry.get('target') and entry['target'] != 'benign':
                graph.add_edge(domain, entry['target'], type='targets')
    
    # Save graph
    try:
        nx.write_graphml(graph, output_path)
        logger.info(f"Saved knowledge graph to {output_path}")
    except Exception as e:
        logger.error(f"Error saving knowledge graph: {str(e)}")

def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    model_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    early_stopping_patience: int = 5
):
    """Train the phishing detection model with early stopping."""
    logger.info("Training model...")
    
    # Initialize detector
    detector = PhishingDetector()
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        detector.model.train()
        total_loss = 0
        
        # Process training data
        for url_data in tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}"):
            try:
                # Preprocess URL
                x, edge_index, edge_attr = detector.preprocess_url(url_data)
                
                # Get target
                target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                
                # Forward pass
                output = detector.model(x, edge_index, edge_attr)
                loss = torch.nn.functional.cross_entropy(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Error processing URL {url_data.get('url', 'unknown')}: {str(e)}")
                continue
        
        # Validation
        detector.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for url_data in val_data:
                try:
                    # Preprocess URL
                    x, edge_index, edge_attr = detector.preprocess_url(url_data)
                    
                    # Get target
                    target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                    
                    # Forward pass
                    output = detector.model(x, edge_index, edge_attr)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    val_loss += loss.item()
                    
                    # Get predictions
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += 1
                except Exception as e:
                    logger.error(f"Error validating URL {url_data.get('url', 'unknown')}: {str(e)}")
                    continue
        
        # Calculate metrics
        avg_train_loss = total_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        val_accuracy = correct / total if total > 0 else 0
        
        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )
        
        # Save checkpoint
        checkpoint_path = f"{model_path}.checkpoint"
        detector.save_model(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping check
        early_stopping(avg_val_loss, detector.model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            detector.model.load_state_dict(early_stopping.best_model)
            break
    
    # Save final model
    detector.save_model(model_path)
    logger.info(f"Saved final model to {model_path}")

async def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data from multiple sources
    phishing_data = []
    
    # Try loading from PhishTank public datasets
    phishing_data.extend(load_phishtank_public_data())
    phishing_data.extend(load_phishtank_public_json())
    
    # If we don't have enough data, generate synthetic data
    if len(phishing_data) < 1000:
        synthetic_data = generate_synthetic_phishing_urls(limit=1000 - len(phishing_data))
        phishing_data.extend(synthetic_data)
    
    # Load benign URLs
    benign_data = load_benign_urls(limit=len(phishing_data))
    
    # Combine and shuffle data
    all_data = phishing_data + benign_data
    random.shuffle(all_data)
    
    # Process URLs
    processed_data = await process_urls(all_data)
    
    # Split data
    train_data, val_data = train_test_split(
        processed_data,
        test_size=0.2,
        random_state=42
    )
    
    # Build knowledge graph
    build_knowledge_graph(
        processed_data,
        'data/knowledge_graph.graphml'
    )
    
    # Train model with early stopping
    train_model(
        train_data=train_data,
        val_data=val_data,
        model_path='models/phishing_detector.pt',
        epochs=100,
        batch_size=32,
        early_stopping_patience=5
    )

if __name__ == "__main__":
    asyncio.run(main()) 