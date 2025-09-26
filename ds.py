import datasets
from datasets import load_dataset

print("=== Library Information ===")


print("\n=== Dataset Information ===")
try:
    dataset = load_dataset(
        r"C:\Users\Admin\Desktop\VASOT-Thesis\gteaload.py",
        name="split1",
        trust_remote_code=True
    )
    
    print(f"Dataset version: {dataset['train'].info.version}")
    print(f"Dataset description: {dataset['train'].info.description}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
