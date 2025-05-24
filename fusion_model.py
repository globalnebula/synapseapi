import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalWeighting(nn.Module):
    """Implements temporal weighting for threat knowledge."""
    
    def __init__(self, decay_factor: float = 0.1):
        """
        Initialize temporal weighting module.
        
        Args:
            decay_factor: Controls how quickly weights decay over time (default: 0.1)
        """
        super().__init__()
        self.decay_factor = decay_factor
    
    def calculate_time_weight(self, timestamp: str) -> float:
        """
        Calculate weight based on timestamp.
        
        Args:
            timestamp: ISO format timestamp string
            
        Returns:
            float: Weight between 0 and 1
        """
        try:
            # Convert timestamp to datetime
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Calculate time difference in days
            time_diff = (datetime.now() - timestamp).days
            
            # Apply exponential decay
            weight = np.exp(-self.decay_factor * time_diff)
            
            return float(weight)
        except Exception as e:
            logger.error(f"Error calculating time weight: {str(e)}")
            return 0.5  # Default weight if calculation fails
    
    def forward(self, edge_weights: torch.Tensor, timestamps: list) -> torch.Tensor:
        """
        Apply temporal weighting to edge weights.
        
        Args:
            edge_weights: Original edge weights tensor
            timestamps: List of timestamps for each edge
            
        Returns:
            torch.Tensor: Temporally weighted edge weights
        """
        # Calculate time weights
        time_weights = torch.tensor([
            self.calculate_time_weight(ts) for ts in timestamps
        ], device=edge_weights.device)
        
        # Apply weights
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
        """
        Initialize PSSA module.
        
        Args:
            gnn_dim: Dimension of GNN embeddings
            llm_dim: Dimension of LLM embeddings
            hidden_dim: Hidden dimension for alignment (default: 256)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Alignment layers
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
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Fusion layer
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
        """
        Forward pass of PSSA.
        
        Args:
            gnn_embeddings: GNN output embeddings
            llm_embeddings: LLM output embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Final phishing score
                - Attention weights for interpretability
        """
        # Project embeddings to same dimension
        gnn_proj = self.gnn_projection(gnn_embeddings)
        llm_proj = self.llm_projection(llm_embeddings)
        
        # Apply attention
        if attention_mask is None:
            attention_mask = torch.ones(
                (gnn_proj.size(0), gnn_proj.size(0)),
                device=gnn_proj.device
            )
        
        # Reshape for attention
        gnn_proj = gnn_proj.unsqueeze(0)  # Add sequence dimension
        llm_proj = llm_proj.unsqueeze(0)
        
        # Apply cross-attention
        attended_gnn, attention_weights = self.attention(
            gnn_proj,
            llm_proj,
            llm_proj,
            key_padding_mask=attention_mask
        )
        
        # Remove sequence dimension
        attended_gnn = attended_gnn.squeeze(0)
        
        # Concatenate and fuse
        combined = torch.cat([attended_gnn, llm_proj.squeeze(0)], dim=-1)
        final_score = self.fusion(combined)
        
        return final_score, attention_weights

class PhishingScoreFusion(nn.Module):
    """Combines GNN and LLM outputs with temporal weighting and PSSA."""
    
    def __init__(
        self,
        gnn_dim: int,
        llm_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temporal_decay: float = 0.1
    ):
        """
        Initialize fusion model.
        
        Args:
            gnn_dim: Dimension of GNN embeddings
            llm_dim: Dimension of LLM embeddings
            hidden_dim: Hidden dimension for PSSA (default: 256)
            dropout: Dropout rate (default: 0.1)
            temporal_decay: Decay factor for temporal weighting (default: 0.1)
        """
        super().__init__()
        
        self.temporal_weighting = TemporalWeighting(decay_factor=temporal_decay)
        self.pssa = PSSA(
            gnn_dim=gnn_dim,
            llm_dim=llm_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        gnn_output: Dict[str, torch.Tensor],
        llm_output: Dict[str, torch.Tensor],
        metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the fusion model.
        
        Args:
            gnn_output: Dictionary containing GNN outputs
                - embeddings: GNN node embeddings
                - edge_weights: Edge weights from GNN
            llm_output: Dictionary containing LLM outputs
                - embeddings: LLM embeddings
                - attention_mask: Optional attention mask
            metadata: Dictionary containing metadata
                - timestamps: List of timestamps for edges
                - other metadata as needed
        
        Returns:
            Dict containing:
                - final_score: Combined phishing score
                - attention_weights: Attention weights for interpretability
                - temporal_weights: Temporal weights applied
        """
        # Apply temporal weighting to edge weights
        temporal_weights = self.temporal_weighting(
            gnn_output['edge_weights'],
            metadata['timestamps']
        )
        
        # Update GNN embeddings with temporal weights
        gnn_embeddings = gnn_output['embeddings'] * temporal_weights.unsqueeze(-1)
        
        # Apply PSSA
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

def get_fusion_score(
    gnn_output: Dict[str, torch.Tensor],
    llm_output: Dict[str, torch.Tensor],
    metadata: Dict,
    model_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Get final phishing score using the fusion model.
    
    Args:
        gnn_output: GNN model outputs
        llm_output: LLM model outputs
        metadata: URL metadata including timestamps
        model_path: Optional path to saved model weights
        
    Returns:
        Dict containing:
            - phishing_score: Final combined score
            - confidence: Confidence in the prediction
            - temporal_weight: Applied temporal weight
    """
    # Initialize model
    model = PhishingScoreFusion(
        gnn_dim=gnn_output['embeddings'].size(-1),
        llm_dim=llm_output['embeddings'].size(-1)
    )
    
    # Load weights if provided
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    # Get prediction
    with torch.no_grad():
        output = model(gnn_output, llm_output, metadata)
    
    # Calculate confidence based on attention weights
    confidence = torch.mean(output['attention_weights']).item()
    
    return {
        'phishing_score': output['final_score'].item(),
        'confidence': confidence,
        'temporal_weight': torch.mean(output['temporal_weights']).item()
    }

if __name__ == "__main__":
    # Example usage
    gnn_output = {
        'embeddings': torch.randn(10, 128),  # 10 nodes, 128-dim embeddings
        'edge_weights': torch.randn(10)  # 10 edges
    }
    
    llm_output = {
        'embeddings': torch.randn(10, 256),  # 10 tokens, 256-dim embeddings
        'attention_mask': torch.ones(10, 10)  # 10x10 attention mask
    }
    
    metadata = {
        'timestamps': [datetime.now().isoformat() for _ in range(10)]
    }
    
    score = get_fusion_score(gnn_output, llm_output, metadata)
    print(f"Phishing Score: {score['phishing_score']:.4f}")
    print(f"Confidence: {score['confidence']:.4f}")
    print(f"Temporal Weight: {score['temporal_weight']:.4f}") 