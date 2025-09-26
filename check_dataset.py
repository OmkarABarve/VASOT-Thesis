# test_dataset.py
import os
import numpy as np

class SimpleGTEALoader:
    def __init__(self, data_dir=r"C:\Users\Admin\Desktop\VASOT-Thesis\gtea"):
        self.data_dir = data_dir
        self.load_mapping()
    
    def load_mapping(self):
        """Load action mapping from mapping.txt"""
        mapping_path = os.path.join(self.data_dir, "mapping.txt")
        with open(mapping_path, "r") as f:
            actions = f.read().splitlines()
        
        self.actions_dict = {}
        for a in actions:
            parts = a.split()
            if len(parts) >= 2:
                self.actions_dict[parts[1]] = int(parts[0])
    
    def load_split(self, split_name, config_name="split1"):
        """Load train or test split"""
        split_file = os.path.join(self.data_dir, "splits", f"{split_name}.{config_name}.bundle")
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()
        
        data = []
        for line in lines:
            vid = line[:-4]  # Remove .txt extension
            
            # Load features
            feature_path = os.path.join(self.data_dir, "features", f"{vid}.npy")
            feature = np.load(feature_path).T.astype(np.float32)
            
            # Load ground truth
            gt_path = os.path.join(self.data_dir, "groundTruth", line)
            with open(gt_path, "r") as fgt:
                content = fgt.read().splitlines()
            
            # Create labels
            label = np.zeros(min(feature.shape[0], len(content)), dtype=np.int32)
            for i in range(len(label)):
                if content[i] in self.actions_dict:
                    label[i] = self.actions_dict[content[i]]
            
            data.append({
                "video_id": vid,
                "video_feature": feature,
                "video_label": label
            })
        
        return data

# Test the loader
if __name__ == "__main__":
    try:
        loader = SimpleGTEALoader()
        
        # Load train and test data
        train_data = loader.load_split("train", "split1")
        test_data = loader.load_split("test", "split1")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Test first example
        example = train_data[0]
        print(f"Video ID: {example['video_id']}")
        print(f"Feature shape: {example['video_feature'].shape}")
        print(f"Label length: {len(example['video_label'])}")
        print(f"Unique labels: {np.unique(example['video_label'])}")
        
        # Test all splits
        print("\n--- Testing all splits ---")
        for split in ["split1", "split2", "split3", "split4"]:
            train = loader.load_split("train", split)
            test = loader.load_split("test", split)
            print(f"{split}: Train={len(train)}, Test={len(test)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
