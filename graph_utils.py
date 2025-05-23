import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class URLGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_types = {
            'url': 0,
            'header': 1,
            'content': 2,
            'security': 3,
            'domain': 4
        }
        
    def _create_node_id(self, node_type: str, value: str) -> str:
        """Create a unique node ID."""
        return f"{node_type}:{value}"
    
    def _add_url_node(self, url_data: Dict) -> str:
        """Add URL node and its attributes."""
        url_id = self._create_node_id('url', url_data['original_url'])
        self.graph.add_node(
            url_id,
            type='url',
            url=url_data['original_url'],
            final_url=url_data['final_url'],
            status_code=url_data['status_code'],
            load_time=url_data['load_time']
        )
        return url_id
    
    def _add_header_nodes(self, url_id: str, headers: Dict[str, str]) -> List[str]:
        """Add header nodes and connect to URL node."""
        header_ids = []
        for header, value in headers.items():
            header_id = self._create_node_id('header', f"{header}:{value}")
            self.graph.add_node(
                header_id,
                type='header',
                name=header,
                value=value
            )
            self.graph.add_edge(url_id, header_id, type='has_header')
            header_ids.append(header_id)
        return header_ids
    
    def _add_content_node(self, url_id: str, content_data: Dict) -> str:
        """Add content node and connect to URL node."""
        content_id = self._create_node_id('content', url_id)
        self.graph.add_node(
            content_id,
            type='content',
            title=content_data.get('page_title'),
            text_content=content_data.get('text_content')
        )
        self.graph.add_edge(url_id, content_id, type='has_content')
        return content_id
    
    def _add_security_nodes(self, url_id: str, security_data: Dict) -> List[str]:
        """Add security nodes and connect to URL node."""
        security_ids = []
        
        # Add SSL info
        if security_data.get('ssl_info'):
            ssl_id = self._create_node_id('security', f"ssl:{url_id}")
            self.graph.add_node(
                ssl_id,
                type='security',
                security_type='ssl',
                **security_data['ssl_info']
            )
            self.graph.add_edge(url_id, ssl_id, type='has_ssl')
            security_ids.append(ssl_id)
        
        # Add threat intel
        if security_data.get('threat_intel'):
            threat_id = self._create_node_id('security', f"threat:{url_id}")
            self.graph.add_node(
                threat_id,
                type='security',
                security_type='threat',
                **security_data['threat_intel']
            )
            self.graph.add_edge(url_id, threat_id, type='has_threat')
            security_ids.append(threat_id)
        
        return security_ids
    
    def _add_domain_node(self, url_id: str, domain_info: Dict) -> str:
        """Add domain node and connect to URL node."""
        domain_id = self._create_node_id('domain', domain_info['domain'])
        self.graph.add_node(
            domain_id,
            type='domain',
            **domain_info
        )
        self.graph.add_edge(url_id, domain_id, type='belongs_to')
        return domain_id
    
    def build_graph(self, url_data: Dict) -> nx.MultiDiGraph:
        """Build a graph from URL metadata."""
        # Add URL node
        url_id = self._add_url_node(url_data)
        
        # Add header nodes
        self._add_header_nodes(url_id, url_data['headers'])
        
        # Add content node
        self._add_content_node(url_id, {
            'page_title': url_data['page_title'],
            'text_content': url_data['text_content']
        })
        
        # Add security nodes
        self._add_security_nodes(url_id, {
            'ssl_info': url_data['ssl_info'],
            'threat_intel': url_data['threat_intel']
        })
        
        # Add domain node
        self._add_domain_node(url_id, url_data['domain_info'])
        
        return self.graph
    
    def to_pyg_data(self) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        # Get node features
        node_features = []
        node_mapping = {}
        
        for i, (node_id, node_data) in enumerate(self.graph.nodes(data=True)):
            node_mapping[node_id] = i
            # Create feature vector based on node type
            features = torch.zeros(len(self.node_types))
            features[self.node_types[node_data['type']]] = 1.0
            node_features.append(features)
        
        # Get edge indices and features
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            # Add edge type as feature
            edge_attr.append(torch.tensor([1.0]))  # Placeholder for edge features
        
        return Data(
            x=torch.stack(node_features),
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.stack(edge_attr) if edge_attr else None
        )

class GlobalKnowledgeGraph:
    def __init__(self, cache_file: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.cache_file = cache_file
        if cache_file:
            self.load_cache()
    
    def load_cache(self):
        """Load graph from cache file."""
        try:
            self.graph = nx.read_graphml(self.cache_file)
            logger.info(f"Loaded knowledge graph from {self.cache_file}")
        except Exception as e:
            logger.warning(f"Could not load cache: {str(e)}")
    
    def save_cache(self):
        """Save graph to cache file."""
        if self.cache_file:
            try:
                nx.write_graphml(self.graph, self.cache_file)
                logger.info(f"Saved knowledge graph to {self.cache_file}")
            except Exception as e:
                logger.error(f"Could not save cache: {str(e)}")
    
    def add_domain_connection(self, domain1: str, domain2: str, connection_type: str):
        """Add a connection between two domains."""
        self.graph.add_edge(domain1, domain2, type=connection_type)
    
    def get_domain_connections(self, domain: str) -> List[Tuple[str, str]]:
        """Get all connections for a domain."""
        return [(u, v, d['type']) for u, v, d in self.graph.edges(domain, data=True)]
    
    def merge_with_url_graph(self, url_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Merge URL graph with global knowledge graph."""
        merged_graph = url_graph.copy()
        
        # Find domain nodes in URL graph
        domain_nodes = [n for n, d in url_graph.nodes(data=True) if d['type'] == 'domain']
        
        # Add connections from global graph
        for domain_node in domain_nodes:
            domain = url_graph.nodes[domain_node]['domain']
            connections = self.get_domain_connections(domain)
            
            for _, connected_domain, conn_type in connections:
                # Add connected domain if not exists
                if connected_domain not in merged_graph:
                    merged_graph.add_node(
                        connected_domain,
                        type='domain',
                        domain=connected_domain
                    )
                
                # Add connection
                merged_graph.add_edge(
                    domain_node,
                    connected_domain,
                    type=f'global_{conn_type}'
                )
        
        return merged_graph 