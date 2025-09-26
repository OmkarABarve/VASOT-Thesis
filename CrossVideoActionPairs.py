import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class UnsupervisedCrossVideoFeaturePairs:
    def __init__(self, dataset: List[Dict]):
        """
        dataset: list of dicts with keys:
          - 'video_id': str
          - 'video_feature': np.ndarray of shape (T, D)
          - 'video_label': np.ndarray of shape (T,) [NOT USED!]
        """
        self.dataset = dataset

    def visual_similarity_cross_video_pairs(
        self,
        window_size: int = 16,
        similarity_threshold: float = 0.75,
        max_pairs_per_video: int = 20,
        negative_ratio: float = 0.3,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Create cross-video pairs based on VISUAL SIMILARITY only (no labels).
        
        Video S2_Cheese: Window showing "cutting bread"
Video S4_Coffee: Window showing "cutting vegetables" 
→ High visual similarity → Positive pair (both cutting actions)

Video S2_Tea: Window showing "pouring water"
Video S3_Hotdog: Window showing "taking ingredients"
→ Low visual similarity → Negative pair (different actions)
        
        """
        pairs = []
        labels = []
        
        # Extract all windows from all videos
        all_windows = []
        for video in self.dataset:
            features = video['video_feature']
            video_id = video['video_id']
            T = len(features)
            
            if T < window_size:
                continue
                
            # Create windows with stride
            for i in range(0, T - window_size + 1, window_size // 2):
                window_feat = features[i:i + window_size]
                
                # Compute window representation (mean pooling)
                window_repr = np.mean(window_feat, axis=0)  # Shape: (2048,)
                
                all_windows.append({
                    'video_id': video_id,
                    'features': window_feat,
                    'representation': window_repr,
                    'start_idx': i,
                })
        
        # Compute pairwise similarities between ALL windows
        representations = np.array([w['representation'] for w in all_windows])
        similarities = cosine_similarity(representations)
        
        # Create pairs based on similarity
        num_windows = len(all_windows)
        created_pairs = 0
        target_pairs = max_pairs_per_video * len(self.dataset)
        
        for i in range(num_windows):
            for j in range(i + 1, num_windows):
                window1 = all_windows[i]
                window2 = all_windows[j]
                
                # Skip if same video (we want cross-video pairs)
                if window1['video_id'] == window2['video_id']:
                    continue
                    
                similarity = similarities[i, j]
                
                # Create positive pairs (high similarity)
                if similarity > similarity_threshold:
                    pairs.append({
                        'window1': window1['features'],
                        'window2': window2['features'],
                        'video_id1': window1['video_id'],
                        'video_id2': window2['video_id'],
                        'start_idx1': window1['start_idx'],
                        'start_idx2': window2['start_idx'],
                        'similarity': similarity,
                        'pair_type': 'cross_video_positive'
                    })
                    labels.append(1)  # Similar
                    created_pairs += 1
                
                # Create negative pairs (low similarity)
                elif similarity < (1.0 - similarity_threshold) and np.random.random() < negative_ratio:
                    pairs.append({
                        'window1': window1['features'],
                        'window2': window2['features'],
                        'video_id1': window1['video_id'],
                        'video_id2': window2['video_id'],
                        'start_idx1': window1['start_idx'],
                        'start_idx2': window2['start_idx'],
                        'similarity': similarity,
                        'pair_type': 'cross_video_negative'
                    })
                    labels.append(0)  # Dissimilar
                    created_pairs += 1
                
                if created_pairs >= target_pairs:
                    break
            
            if created_pairs >= target_pairs:
                break
        
        return pairs, labels

    def temporal_correspondence_pairs(
        self,
        window_size: int = 16,
        pairs_per_activity: int = 30,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Create cross-video pairs based on temporal position within same activity.
        Assumption: Similar temporal positions likely contain similar actions.


        Pairs windows from same activity at similar temporal positions
Assumes: "30% into Cheese-making should be similar across different people"
Structure-based: Uses video names and temporal position
Example:
        """
        pairs = []
        labels = []
        
        # Group videos by activity type (from video_id)
        activity_videos = {}
        for video in self.dataset:
            video_id = video['video_id']
            # Extract activity from video name (e.g., "S2_Cheese_C1" -> "Cheese")
            activity = video_id.split('_')[1]
            
            if activity not in activity_videos:
                activity_videos[activity] = []
            activity_videos[activity].append(video)
        
        # Create pairs within same activity (different subjects)
        for activity, videos in activity_videos.items():
            if len(videos) < 2:
                continue
                
            # For each pair of videos in same activity
            for i in range(len(videos)):
                for j in range(i + 1, len(videos)):
                    video1, video2 = videos[i], videos[j]
                    
                    # Skip if same subject
                    subject1 = video1['video_id'].split('_')[0]
                    subject2 = video2['video_id'].split('_')[0]
                    if subject1 == subject2:
                        continue
                    
                    # Create pairs at similar temporal positions
                    T1, T2 = len(video1['video_feature']), len(video2['video_feature'])
                    min_T = min(T1, T2)
                    
                    if min_T < window_size:
                        continue
                    
                    # Sample corresponding temporal positions
                    num_pairs = min(pairs_per_activity, (min_T - window_size) // 4)
                    for k in range(num_pairs):
                        # Same relative position in both videos
                        pos1 = int(k * (T1 - window_size) / num_pairs)
                        pos2 = int(k * (T2 - window_size) / num_pairs)
                        
                        pairs.append({
                            'window1': video1['video_feature'][pos1:pos1 + window_size],
                            'window2': video2['video_feature'][pos2:pos2 + window_size],
                            'video_id1': video1['video_id'],
                            'video_id2': video2['video_id'],
                            'start_idx1': pos1,
                            'start_idx2': pos2,
                            'activity': activity,
                            'temporal_position': k / num_pairs,
                            'pair_type': 'temporal_correspondence'
                        })
                        labels.append(1)  # Same temporal position = positive
        
        return pairs, labels