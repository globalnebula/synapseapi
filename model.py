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
from datetime import datetime

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

class TemporalWeighting(nn.Module):
    """Implements temporal weighting for threat knowledge."""
    
    def __init__(self, decay_factor: float = 0.1):
        super().__init__()
        self.decay_factor = decay_factor
    
    def calculate_time_weight(self, timestamp: str) -> float:
        try:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            time_diff = (datetime.now() - timestamp).days
            weight = np.exp(-self.decay_factor * time_diff)
            return float(weight)
        except Exception as e:
            logger.error(f"Error calculating time weight: {str(e)}")
            return 0.5
    
    def forward(self, edge_weights: torch.Tensor, timestamps: list) -> torch.Tensor:
        time_weights = torch.tensor([
            self.calculate_time_weight(ts) for ts in timestamps
        ], device=edge_weights.device)
        
        weighted_edges = edge_weights * time_weights
        return weighted_edges

class PSSA(nn.Module):
    """Phishing Score Semantic Alignment module."""
    
    def __init__(
        self,
        gnn_dim: int,
        llm_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gnn_projection = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.llm_projection = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        gnn_embeddings: torch.Tensor,
        llm_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project embeddings to same dimension
        gnn_proj = self.gnn_projection(gnn_embeddings)  # [128, hidden_dim]
        llm_proj = self.llm_projection(llm_embeddings)  # [128, hidden_dim]
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (gnn_proj.size(0), gnn_proj.size(0)),
                device=gnn_proj.device
            )
        
        # Reshape for attention
        gnn_proj = gnn_proj.unsqueeze(0)  # [1, 128, hidden_dim]
        llm_proj = llm_proj.unsqueeze(0)  # [1, 128, hidden_dim]
        
        # Apply attention
        attended_gnn, attention_weights = self.attention(
            gnn_proj,
            llm_proj,
            llm_proj,
            key_padding_mask=attention_mask
        )
        
        # Reshape back
        attended_gnn = attended_gnn.squeeze(0)  # [128, hidden_dim]
        llm_proj = llm_proj.squeeze(0)  # [128, hidden_dim]
        
        # Concatenate and fuse
        combined = torch.cat([attended_gnn, llm_proj], dim=-1)  # [128, hidden_dim*2]
        final_score = self.fusion(combined)  # [128, 1]
        
        # Take mean over sequence dimension
        final_score = final_score.mean(dim=0)  # [1]
        
        return final_score, attention_weights

class PhishingScoreFusion(nn.Module):
    """Combines GNN and LLM outputs with temporal weighting and PSSA."""
    
    def __init__(
        self,
        gnn_dim: int = 4,  # Match input feature dimension
        llm_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temporal_decay: float = 0.1
    ):
        super().__init__()
        
        self.temporal_weighting = TemporalWeighting(decay_factor=temporal_decay)
        self.pssa = PSSA(
            gnn_dim=gnn_dim,
            llm_dim=llm_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Add projection layer for GNN embeddings
        self.gnn_projection = nn.Linear(gnn_dim, hidden_dim)
    
    def forward(
        self,
        gnn_output: Dict[str, torch.Tensor],
        llm_output: Dict[str, torch.Tensor],
        metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        # Project GNN embeddings to match dimensions
        gnn_embeddings = self.gnn_projection(gnn_output['embeddings'])  # [128, hidden_dim]
        
        # Apply temporal weighting
        temporal_weights = self.temporal_weighting(
            gnn_output['edge_weights'],
            metadata['timestamps']
        )
        
        # Apply temporal weights to embeddings
        gnn_embeddings = gnn_embeddings * temporal_weights.unsqueeze(-1)  # [128, hidden_dim]
        
        # Get final score through PSSA
        final_score, attention_weights = self.pssa(
            gnn_embeddings,
            llm_output['embeddings'],
            llm_output.get('attention_mask')
        )
        
        return {
            'final_score': final_score,
            'attention_weights': attention_weights,
            'temporal_weights': temporal_weights
        }

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
    
    def preprocess_url(self, url_data: Dict) -> Dict[str, torch.Tensor]:
        """Preprocess URL data for model input."""
        try:
            # Add URL to graph
            self.graph_builder.add_url(url_data)
            
            # Get graph tensors
            x, edge_index, edge_attr = self.graph_builder.build_graph_tensors()
            
            # If graph is empty, create a minimal valid graph
            if x.size(0) == 0:
                # Create a single node with default features
                x = torch.zeros((1, 4), dtype=torch.float)  # 4 features per node
                edge_index = torch.zeros((2, 0), dtype=torch.long)  # No edges
                edge_attr = torch.zeros((0, 1), dtype=torch.float)  # No edge features
            
            # Ensure x has the correct shape for the model
            if x.size(0) == 1:
                # Duplicate the node features to match expected dimensions
                x = x.repeat(128, 1)  # Repeat to get 128 nodes
                edge_index = torch.zeros((2, 0), dtype=torch.long)  # No edges
                edge_attr = torch.zeros((0, 1), dtype=torch.float)  # No edge features
            
            return {
                'embeddings': x,
                'edge_index': edge_index,
                'edge_weights': torch.ones(x.size(0), dtype=torch.float)  # Default weights
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing URL: {str(e)}")
            # Return minimal valid tensors with correct dimensions
            return {
                'embeddings': torch.zeros((128, 4), dtype=torch.float),  # 128 nodes with 4 features
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_weights': torch.ones(128, dtype=torch.float)
            }
    
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
        try:
            checkpoint = torch.load(path)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'detector_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['detector_state_dict'])
            else:
                # If no state dict found, try loading the entire checkpoint
                self.model.load_state_dict(checkpoint)
            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Initialize with default weights
            logger.info("Initializing model with default weights")
            self.model = PhishingGAT(
                in_channels=4,  # Number of node features
                hidden_channels=64,
                out_channels=2,  # Binary classification
                heads=4
            ) 