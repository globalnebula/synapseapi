import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import logging
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

logger = logging.getLogger(__name__)

class URLGraphBuilder:
    """Builds and manages the URL knowledge graph."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.scaler = StandardScaler()
    
    def add_url(self, url_data: Dict):
        """Add a URL and its metadata to the graph."""
        if 'metadata' not in url_data or 'domain_info' not in url_data['metadata']:
            return
        
        domain = url_data['metadata']['domain_info']['domain']
        
        # Add domain node if not exists
        if domain not in self.graph:
            self.graph.add_node(domain, type='domain')
        
        # Add connections based on target
        if url_data.get('target') and url_data['target'] != 'benign':
            self.graph.add_edge(domain, url_data['target'], type='targets')
    
    def get_node_features(self, node: str) -> torch.Tensor:
        """Extract features for a node in the graph."""
        # Basic features
        features = []
        
        # Node type (domain vs target)
        node_type = self.graph.nodes[node].get('type', 'unknown')
        features.append(1.0 if node_type == 'domain' else 0.0)
        
        # Node degree
        features.append(self.graph.degree(node))
        
        # Number of incoming edges
        features.append(self.graph.in_degree(node))
        
        # Number of outgoing edges
        features.append(self.graph.out_degree(node))
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float)
    
    def get_edge_features(self, edge: Tuple[str, str, str]) -> torch.Tensor:
        """Extract features for an edge in the graph."""
        # Basic features
        features = []
        
        # Edge type
        edge_type = self.graph.edges[edge].get('type', 'unknown')
        features.append(1.0 if edge_type == 'targets' else 0.0)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float)
    
    def build_graph_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tensors for graph neural network input."""
        # Get node features
        node_features = []
        for node in self.graph.nodes():
            features = self.get_node_features(node)
            node_features.append(features)
        
        # Stack node features
        x = torch.stack(node_features)
        
        # Get edge indices and features
        edge_indices = []
        edge_features = []
        for edge in self.graph.edges(data=True):
            # Get node indices
            source_idx = list(self.graph.nodes()).index(edge[0])
            target_idx = list(self.graph.nodes()).index(edge[1])
            
            # Add edge indices
            edge_indices.append([source_idx, target_idx])
            
            # Get edge features
            features = self.get_edge_features(edge)
            edge_features.append(features)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.stack(edge_features)
        
        return x, edge_index, edge_attr

class PhishingGAT(nn.Module):
    """Graph Attention Network for phishing detection."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x

class TextEncoder:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode text into BERT embeddings."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

class PhishingDetector:
    """Main phishing detection model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.graph_builder = URLGraphBuilder()
        self.model = PhishingGAT(
            in_channels=4,  # Number of node features
            hidden_channels=64,
            out_channels=2,  # Binary classification
            heads=4
        )
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_url(self, url_data: Dict) -> torch.Tensor:
        """Preprocess URL data for model input."""
        # Add URL to graph
        self.graph_builder.add_url(url_data)
        
        # Get graph tensors
        x, edge_index, edge_attr = self.graph_builder.build_graph_tensors()
        
        return x, edge_index, edge_attr
    
    def predict(self, url_data: Dict) -> float:
        """Predict phishing probability for a URL."""
        # Preprocess URL
        x, edge_index, edge_attr = self.preprocess_url(url_data)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)
            probs = F.softmax(output, dim=1)
            
            # Get probability of phishing class
            phishing_prob = probs[0, 1].item()
        
        return phishing_prob
    
    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32
    ):
        """Train the model."""
        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Process training data
            for url_data in train_data:
                # Preprocess URL
                x, edge_index, edge_attr = self.preprocess_url(url_data)
                
                # Get target
                target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                
                # Forward pass
                output = self.model(x, edge_index, edge_attr)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for url_data in val_data:
                    # Preprocess URL
                    x, edge_index, edge_attr = self.preprocess_url(url_data)
                    
                    # Get target
                    target = torch.tensor([1.0 if url_data['is_phishing'] else 0.0])
                    
                    # Forward pass
                    output = self.model(x, edge_index, edge_attr)
                    loss = F.cross_entropy(output, target)
                    val_loss += loss.item()
                    
                    # Get predictions
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {total_loss/len(train_data):.4f} | "
                    f"Val Loss: {val_loss/len(val_data):.4f} | "
                    f"Val Acc: {correct/total:.4f}"
                )
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'graph_builder': self.graph_builder
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.graph_builder = checkpoint['graph_builder'] 