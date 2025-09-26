import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

class VASOTModel(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=256, num_prototypes=None):
        super(VASOTModel, self).__init__()
        
        # MLP Encoder - Transform 2048 features to embedding space
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.action_prototypes = None  # Will be initialized later
    
    def discover_and_initialize_prototypes(self, train_embeddings, method='kmeans', 
                                         k_range=(5, 20), selection_criterion='silhouette'):
        """
        Discover optimal number of action prototypes and initialize them
        
        Args:
            train_embeddings: Encoded training features
            method: 'kmeans', 'gmm', etc.
            k_range: Range of possible prototype numbers to try
            selection_criterion: 'silhouette', 'elbow', 'gap_statistic'
        """
        if method == 'kmeans':
            best_k = self._find_optimal_k(train_embeddings, k_range, selection_criterion)
            
            # Initialize K-means with discovered number of clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            kmeans.fit(train_embeddings.detach().cpu().numpy())
            
            # Set prototypes as learnable parameters
            self.num_prototypes = best_k
            self.action_prototypes = nn.Parameter(
                torch.from_numpy(kmeans.cluster_centers_).float()
            )
            
            print(f"Discovered {best_k} action prototypes using {selection_criterion}")
    
    def _find_optimal_k(self, embeddings, k_range, criterion):
        """Find optimal number of clusters using various criteria"""
        from sklearn.metrics import silhouette_score
        
        embeddings_np = embeddings.detach().cpu().numpy()
        scores = []
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)
            
            if criterion == 'silhouette':
                score = silhouette_score(embeddings_np, labels)
                scores.append(score)
            # Add other criteria as needed
        
        # Return k with best score
        best_idx = np.argmax(scores)
        return k_range[0] + best_idx
    
    def encode_features(self, features):
        """Transform raw features to embedding space"""
        return self.feature_encoder(features)
    
    def forward(self, window_features):
        """Forward pass through the model"""
        embeddings = self.encode_features(window_features)
        return embeddings