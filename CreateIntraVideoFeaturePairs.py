import numpy as np
from typing import List, Dict, Tuple


'''
THIS FILE IS USED TO CREATE FEATURE PAIRS FOR THE TRAINING DATASET of frames WITHIN THE SAME ACTION AND WITHIN THE NEXT ACTION OF SAME VIDEO
USEFUL for Temporal Coherence and Action Consistency. 

Note- DOESN'T Compare INTER Video frames. Only Intra-video frames.

'''


class FeaturePairBuilder:
    def __init__(self, dataset: List[Dict]):
        """
        dataset: list of dicts with keys:
          - 'video_id': str
          - 'video_feature': np.ndarray of shape (T, D)
          - 'video_label': np.ndarray of shape (T,)
        """
        self.dataset = dataset

    def sliding_window_pairs(
        self,
        window_size: int = 16,
        stride: int = 4,
        overlap_threshold: float = 0.5,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Build pairs of overlapping windows; label = 1 if overlap_ratio > threshold else 0.
        Returns:
          - pairs: list of dicts containing window data and metadata
          - labels: list[int] (0/1)
        """
        pairs: List[Dict] = []
        labels: List[int] = []

        for video in self.dataset:
            features = video['video_feature']
            labels_arr = video['video_label']

            T = len(features)
            if T < window_size:
                continue

            for i in range(0, T - window_size + 1, stride):
                w1_feat = features[i:i + window_size]
                w1_lbl = labels_arr[i:i + window_size]

                # Subsequent windows within the next window_size range
                j_start = i + 1
                j_end = min(i + window_size, T - window_size + 1)
                for j in range(j_start, j_end):
                    w2_feat = features[j:j + window_size]
                    w2_lbl = labels_arr[j:j + window_size]

                    overlap = len(set(range(i, i + window_size)) & set(range(j, j + window_size)))
                    overlap_ratio = overlap / float(window_size)

                    pairs.append({
                        'window1': w1_feat,
                        'window2': w2_feat,
                        'window1_labels': w1_lbl,
                        'window2_labels': w2_lbl,
                        'overlap_ratio': overlap_ratio,
                        'video_id': video['video_id'],
                        'start_idx1': i,
                        'start_idx2': j,
                    })
                    labels.append(1 if overlap_ratio > overlap_threshold else 0)

        return pairs, labels

    def unsupervised_temporal_pairs(
        self,
        window_size: int = 8,
        step: int = None,
        similarity_threshold: float = 0.7,
    ) -> List[Dict]:
        """
        Create pairs based on feature similarity, not ground truth labels. UNSUPERVISED.
        """
        if step is None:
            step = max(1, window_size // 2)

        pairs = []
        
        for video in self.dataset:
            features = video['video_feature']
            T = len(features)
            
            if T < window_size * 2:
                continue

            # Create all possible windows
            windows = []
            for i in range(0, T - window_size + 1, step):
                window_feat = features[i:i + window_size]
                window_repr = np.mean(window_feat, axis=0)  # Average representation
                windows.append({
                    'features': window_feat,
                    'representation': window_repr,
                    'start_idx': i
                })

            # Create pairs based on feature similarity
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    w1, w2 = windows[i], windows[j]
                    
                    # Compute cosine similarity
                    w1_norm = np.linalg.norm(w1['representation'])
                    w2_norm = np.linalg.norm(w2['representation'])
                    
                    if w1_norm > 0 and w2_norm > 0:  # Avoid division by zero
                        similarity = np.dot(w1['representation'], w2['representation']) / (w1_norm * w2_norm)
                        
                        # Only create pairs if similarity is high (likely same action)
                        if similarity > similarity_threshold:
                            pairs.append({
                                'anchor': w1['features'],
                                'positive': w2['features'],
                                'similarity': similarity,
                                'video_id': video['video_id'],
                                'anchor_start': w1['start_idx'],
                                'positive_start': w2['start_idx'],
                            })
        
        return pairs

# COMMENTED OUT - Uses ground truth labels (not truly self-supervised)
'''
def action_consistency_pairs(
    self,
    window_size: int = 8,
    step: int = None,
) -> List[Dict]:
    """
    Build positive pairs within the same action segment (consistency).
    Returns list of dicts with 'anchor' and 'positive' windows (same action).
    USES GROUND TRUTH LABELS - NOT TRULY SELF-SUPERVISED!
    """
    if step is None:
        step = max(1, window_size // 2)

    pairs: List[Dict] = []

    for video in self.dataset:
        features = video['video_feature']
        labels_arr = video['video_label']  # <-- USES GROUND TRUTH!
        T = len(labels_arr)
        if T < window_size:
            continue

        # Iterate segments of constant label
        current_action = int(labels_arr[0])  # <-- USES GROUND TRUTH!
        segment_start = 0

        for i in range(1, T + 1):
            end_of_sequence = (i == T)
            label_changed = (not end_of_sequence and int(labels_arr[i]) != current_action)  # <-- USES GROUND TRUTH!
			'''