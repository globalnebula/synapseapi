# SynapseAPI: Advanced Phishing Detection System

## Overview
SynapseAPI is a state-of-the-art phishing detection system that combines Graph Neural Networks (GNN), Large Language Models (LLM), and temporal analysis to provide robust protection against sophisticated phishing attacks. The system analyses URLs and their associated metadata to detect potential phishing attempts with high accuracy.

## Live Demo
Test the system at: [https://aisynapse.space](https://aisynapse.space)

## Key Features

### 1. Multi-Modal Analysis
- **Graph Neural Network (GNN)**: Analyses URL structure and relationships
- **Large Language Model (LLM)**: Processes textual content and semantic patterns
- **Temporal Analysis**: Considers time-based patterns and historical data

### 2. Advanced Architecture
- **PhishingGAT**: Graph Attention Network for URL analysis
- **PSSA (Phishing Score Semantic Alignment)**: Aligns GNN and LLM outputs
- **Temporal Weighting**: Weights predictions based on temporal relevance

### 3. Real-time Processing
- Fast URL pre-processing and analysis
- Batch processing capabilities
- Asynchronous API endpoints

### 4. Security Features
- SSL certificate validation
- Threat intelligence integration
- Security header analysis
- Domain reputation checking

## Technical Architecture

### Model Components

1. **URL Graph Builder**
   - Constructs knowledge graphs from URL data
   - Extracts domain relationships
   - Generates node and edge features
   - Node Features:
     - Domain type (legitimate vs suspicious)
     - SSL certificate status
     - Domain age
     - Registration information
     - IP address information
     - Geographic location
     - ASN details
   - Edge Features:
     - Connection type (redirect, subdomain, etc.)
     - Temporal relationship
     - Trust score
     - Redirect chain length
     - Domain similarity score

2. **PhishingGAT**
   - Graph Attention Network implementation
   - Multi-head attention mechanism (4 heads)
   - Hierarchical feature extraction
   - Architecture Details:
     ```
     Input Layer (4 features)
     ↓
     GAT Layer 1 (64 hidden units, 4 heads)
     ↓
     ReLU + Dropout (0.2)
     ↓
     GAT Layer 2 (64 hidden units, 4 heads)
     ↓
     ReLU + Dropout (0.2)
     ↓
     GAT Layer 3 (2 output units, 1 head)
     ↓
     Softmax
     ```
   - Attention Mechanism:
     - Query: Current node features
     - Key: Neighbor node features
     - Value: Weighted neighbor features
     - Attention Score: Softmax(QK^T/√d)

3. **PSSA Module**
   - Semantic alignment between GNN and LLM outputs
   - Attention-based fusion
   - Confidence scoring
   - Architecture Details:
     ```
     GNN Embeddings (128, 4) → Projection → (128, 256)
     LLM Embeddings (128, 256)
     ↓
     Multi-head Attention (4 heads)
     ↓
     Concatenation
     ↓
     Fusion Network (512 → 256 → 1)
     ↓
     Sigmoid
     ```
   - Attention Computation:
     ```
     Attention(Q, K, V) = softmax(QK^T/√d)V
     where d is the dimension of the key vectors
     ```

4. **Temporal Weighting**
   - Time-based decay factors
   - Historical pattern analysis
   - Dynamic weight adjustment
   - Formula: weight = exp(-decay_factor * time_diff)
   - Decay Factors:
     - URL age: 0.1
     - Certificate age: 0.05
     - Domain age: 0.02
     - Historical updates: 0.15

## Model Training

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Batch Size | 32 | Number of samples per batch |
| Epochs | 100 | Maximum training epochs |
| Early Stopping | 5 | Patience for early stopping |
| Dropout Rate | 0.1 | Dropout probability |
| Hidden Dim | 256 | Hidden layer dimension |
| GNN Heads | 4 | Number of attention heads |
| Temporal Decay | 0.1 | Decay factor for temporal weighting |
| Weight Decay | 1e-5 | L2 regularization |
| Gradient Clip | 1.0 | Maximum gradient norm |
| Warmup Steps | 1000 | Learning rate warmup |

### Training Process
1. **Data Preparation**
   - URL collection from PhishTank
   - Benign URL sampling
   - Metadata extraction
   - Graph construction
   - Feature normalization
   - Data augmentation

2. **Training Loop**
   - Forward pass through GNN
   - LLM output generation
   - PSSA fusion
   - Loss calculation (Binary Cross Entropy)
   - Backpropagation
   - Parameter updates
   - Gradient clipping
   - Learning rate scheduling

3. **Validation**
   - Cross-validation on test set
   - Performance metrics calculation
   - Model checkpointing
   - Early stopping
   - Hyperparameter tuning

## Performance Analysis

### Attack Type Performance

| Attack Type | Precision | Recall | F1-Score | Average Detection Time |
|-------------|-----------|---------|-----------|----------------------|
| Brand Impersonation | 95.2% | 96.8% | 96.0% | 0.8s |
| Credential Theft | 94.5% | 95.2% | 94.8% | 0.9s |
| Malware Distribution | 92.8% | 93.5% | 93.1% | 1.1s |
| Payment Fraud | 93.5% | 94.2% | 93.8% | 0.85s |
| Social Engineering | 91.5% | 92.8% | 92.1% | 0.95s |

### Feature Importance
1. **URL Structure Features** (35%)
   - Domain similarity
   - Path patterns
   - Query parameters
   - URL length
   - Special characters
   - TLD analysis

2. **SSL/TLS Features** (25%)
   - Certificate validity
   - Issuer reputation
   - Protocol version
   - Cipher suite
   - Certificate chain
   - HSTS status

3. **Domain Features** (20%)
   - Registration age
   - WHOIS information
   - DNS records
   - Nameserver analysis
   - Domain reputation
   - IP history

4. **Content Features** (15%)
   - Page similarity
   - Text patterns
   - Image analysis
   - JavaScript analysis
   - Form analysis
   - Meta tags

5. **Temporal Features** (5%)
   - URL age
   - Update frequency
   - Historical patterns
   - Certificate updates
   - DNS changes
   - Content modifications

### Performance Metrics by URL Type

| URL Type | Accuracy | False Positive Rate | Processing Time | Memory Usage |
|----------|----------|---------------------|-----------------|--------------|
| Short URLs | 93.5% | 2.1% | 0.8s | 128MB |
| Long URLs | 94.2% | 1.8% | 0.9s | 132MB |
| Redirect Chains | 92.8% | 2.5% | 1.2s | 145MB |
| Subdomains | 95.1% | 1.5% | 0.7s | 125MB |
| International Domains | 93.2% | 2.3% | 1.0s | 135MB |

## Installation

1. Clone the repository:
```