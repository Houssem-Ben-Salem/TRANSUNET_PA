import numpy as np
import sys
import os

def inspect_weights(weights_path):
    """
    Inspect the structure of the weights file to help debug loading issues.
    """
    print(f"Checking if file exists: {os.path.exists(weights_path)}")
    try:
        weights = np.load(weights_path, allow_pickle=True)
        
        # Check if it's a .npz file (has 'files' attribute) or .npy (needs .item())
        if hasattr(weights, 'files'):
            print(f"Found .npz file with keys: {weights.files}")
            # Print a few sample keys for better understanding
            for key in list(weights.keys())[:10]:  # Print first 10 keys
                print(f"Sample key: {key}")
        else:
            # Likely a .npy file with dict
            weights_dict = weights.item()
            if isinstance(weights_dict, dict):
                print(f"Found .npy file with dictionary containing {len(weights_dict)} keys")
                # Print some sample keys
                sample_keys = list(weights_dict.keys())[:10]  # First 10 keys
                print("Sample keys:")
                for key in sample_keys:
                    print(f"  {key}")
                
                # Specifically look for attention-related keys to diagnose the issue
                attention_keys = [k for k in weights_dict.keys() if 'attention' in k.lower() or 'multihead' in k.lower()]
                if attention_keys:
                    print("\nFound attention-related keys:")
                    for key in attention_keys[:5]:  # First 5 attention keys
                        print(f"  {key}")
            else:
                print(f"File contains data of type {type(weights_dict)}, not a dictionary")
    except Exception as e:
        print(f"Error loading weights file: {str(e)}")
        
if __name__ == "__main__":
    # Replace with your actual weights path from config
    weights_path = "./model/imagenet21k_R50+ViT-B_16.npz"
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    
    print(f"Inspecting weights file: {weights_path}")
    inspect_weights(weights_path)