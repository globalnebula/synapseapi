import json
import logging
import os
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import PhishingDetector, PhishingScoreFusion
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
            metadata = await scrape_url(url_data['url'], check_threat_intel=True)
            
            # Add to processed data
            processed_data.append({
                **url_data,
                'metadata': metadata
            })
            
        except Exception as e:
            logger.error(f"Error processing URL {url_data['url']}: {str(e)}")
            # Add the URL data even if metadata extraction fails
            processed_data.append({
                **url_data,
                'metadata': None
            })
    
    if not processed_data:
        logger.error("No URLs were successfully processed!")
        raise ValueError("No valid data available for training")
    
    return processed_data

def build_knowledge_graph(data: List[Dict], output_path: str):
    """Build and save the knowledge graph."""
    logger.info("Building knowledge graph...")
    graph = nx.MultiDiGraph()
    
    # Add domain connections
    for entry in data:
        try:
            # Skip if metadata is None
            if not entry.get('metadata'):
                continue
                
            domain_info = entry['metadata'].get('domain_info', {})
            if not domain_info:
                continue
                
            domain = domain_info.get('domain')
            if not domain:
                continue
            
            # Add domain node if not exists
            if domain not in graph:
                graph.add_node(domain, type='domain')
            
            # Add connections based on target
            if entry.get('target') and entry['target'] != 'benign':
                graph.add_edge(domain, entry['target'], type='targets')
                
        except Exception as e:
            logger.error(f"Error processing entry for knowledge graph: {str(e)}")
            continue
    
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
    
    # Initialize detector and fusion model
    detector = PhishingDetector()
    fusion_model = PhishingScoreFusion(
        gnn_dim=4,  # Match input feature dimension
        llm_dim=256,
        hidden_dim=256,
        dropout=0.1,
        temporal_decay=0.1
    )
    
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Set up optimizers
    detector_optimizer = torch.optim.Adam(detector.model.parameters(), lr=lr)
    fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        detector.model.train()
        fusion_model.train()
        total_loss = 0
        processed_samples = 0
        
        # Process training data
        for url_data in tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}"):
            try:
                # Skip if metadata is None
                if url_data.get('metadata') is None:
                    continue
                
                # Get GNN output
                gnn_output = detector.preprocess_url(url_data)
                
                # Get LLM output (placeholder - replace with actual LLM output)
                llm_output = {
                    'embeddings': torch.randn(128, 256),  # Match GNN output dimensions
                    'attention_mask': torch.ones(128, 128)
                }
                
                # Prepare metadata
                metadata = {
                    'timestamps': [url_data.get('verified_at', datetime.now().isoformat())]
                }
                
                # Get target
                target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                
                # Forward pass through fusion model
                fusion_output = fusion_model(gnn_output, llm_output, metadata)
                final_score = fusion_output['final_score']
                
                # Calculate loss
                loss = torch.nn.functional.binary_cross_entropy(final_score, target)
                
                # Backward pass
                detector_optimizer.zero_grad()
                fusion_optimizer.zero_grad()
                loss.backward()
                detector_optimizer.step()
                fusion_optimizer.step()
                
                total_loss += loss.item()
                processed_samples += 1
                
            except Exception as e:
                logger.error(f"Error processing URL {url_data.get('url', 'unknown')}: {str(e)}")
                continue
        
        if processed_samples == 0:
            logger.error("No valid samples processed in this epoch!")
            continue
        
        # Validation
        detector.model.eval()
        fusion_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for url_data in val_data:
                try:
                    # Skip if metadata is None
                    if url_data.get('metadata') is None:
                        continue
                    
                    # Get GNN output
                    gnn_output = detector.preprocess_url(url_data)
                    
                    # Get LLM output
                    llm_output = {
                        'embeddings': torch.randn(128, 256),  # Match GNN output dimensions
                        'attention_mask': torch.ones(128, 128)
                    }
                    
                    # Prepare metadata
                    metadata = {
                        'timestamps': [url_data.get('verified_at', datetime.now().isoformat())]
                    }
                    
                    # Get target
                    target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                    
                    # Forward pass
                    fusion_output = fusion_model(gnn_output, llm_output, metadata)
                    final_score = fusion_output['final_score']
                    
                    # Calculate loss
                    loss = torch.nn.functional.binary_cross_entropy(final_score, target)
                    val_loss += loss.item()
                    
                    # Get predictions
                    pred = (final_score > 0.5).float()
                    correct += (pred == target).sum().item()
                    total += 1
                    
                except Exception as e:
                    logger.error(f"Error validating URL {url_data.get('url', 'unknown')}: {str(e)}")
                    continue
        
        if total == 0:
            logger.error("No valid samples in validation set!")
            continue
        
        # Calculate metrics
        avg_train_loss = total_loss / processed_samples
        avg_val_loss = val_loss / total
        val_accuracy = correct / total
        
        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Processed Samples: {processed_samples}/{len(train_data)}"
        )
        
        # Save checkpoint
        checkpoint_path = f"{model_path}.checkpoint"
        torch.save({
            'detector_state_dict': detector.model.state_dict(),
            'fusion_state_dict': fusion_model.state_dict(),
            'detector_optimizer': detector_optimizer.state_dict(),
            'fusion_optimizer': fusion_optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping check
        early_stopping(avg_val_loss, detector.model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            detector.model.load_state_dict(early_stopping.best_model)
            break
    
    # Save final model
    torch.save({
        'detector_state_dict': detector.model.state_dict(),
        'fusion_state_dict': fusion_model.state_dict()
    }, model_path)
    logger.info(f"Saved final model to {model_path}")

def sample_balanced_dataset(data: List[Dict], max_samples: int = 5000) -> List[Dict]:
    """
    Sample a balanced subset of the dataset, ensuring good representation of different types.
    
    Args:
        data: List of URL data dictionaries
        max_samples: Maximum number of samples to return (default: 5000)
    
    Returns:
        List of sampled URL data dictionaries
    """
    logger.info(f"Sampling balanced dataset from {len(data)} URLs...")
    
    # Group URLs by target
    target_groups = {}
    for entry in data:
        target = entry.get('target', 'unknown')
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(entry)
    
    # Calculate samples per target
    samples_per_target = max_samples // len(target_groups)
    sampled_data = []
    
    # Sample from each target group
    for target, urls in target_groups.items():
        if len(urls) > samples_per_target:
            # If we have more URLs than needed, sample intelligently
            # Prioritize recently verified URLs
            sorted_urls = sorted(urls, key=lambda x: x.get('verified_at', ''), reverse=True)
            sampled_urls = sorted_urls[:samples_per_target]
        else:
            # If we have fewer URLs than needed, use all of them
            sampled_urls = urls
        
        sampled_data.extend(sampled_urls)
        logger.info(f"Sampled {len(sampled_urls)} URLs for target '{target}'")
    
    # Shuffle the sampled data
    random.shuffle(sampled_data)
    
    logger.info(f"Final sampled dataset size: {len(sampled_data)} URLs")
    return sampled_data

async def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data from multiple sources
    phishing_data = []
    
    # Try loading from PhishTank public datasets
    phishtank_data = []
    phishtank_data.extend(load_phishtank_public_data())
    phishtank_data.extend(load_phishtank_public_json())
    
    # Sample a balanced subset from PhishTank data
    if phishtank_data:
        phishing_data.extend(sample_balanced_dataset(phishtank_data, max_samples=5000))
    
    # If we don't have enough data, generate synthetic data
    if len(phishing_data) < 1000:
        synthetic_data = generate_synthetic_phishing_urls(limit=1000 - len(phishing_data))
        phishing_data.extend(synthetic_data)
    
    # Load benign URLs - match the number of phishing URLs
    benign_data = load_benign_urls(limit=len(phishing_data))
    
    # Combine and shuffle data
    all_data = phishing_data + benign_data
    random.shuffle(all_data)
    
    logger.info(f"Total dataset size: {len(all_data)} URLs ({len(phishing_data)} phishing, {len(benign_data)} benign)")
    
    # Process URLs
    try:
        processed_data = await process_urls(all_data)
        
        if not processed_data:
            raise ValueError("No data was successfully processed")
        
        # Split data
        train_data, val_data = train_test_split(
            processed_data,
            test_size=0.2,
            random_state=42
        )
        
        logger.info(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")
        
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
    except Exception as e:
        logger.error(f"Error during training process: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 