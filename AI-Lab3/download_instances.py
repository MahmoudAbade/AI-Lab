import requests
import os

def download_cvrp_instances():
    """Download required CVRP instances"""
    
    base_url = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/"
    
    instances = {
        # Beginner (n ≤ 30)
        "P-n16-k8.vrp": "P/P-n16-k8.vrp",
        "E-n22-k4.vrp": "E/E-n22-k4.vrp",
        
        # Intermediate (30 < n ≤ 80)  
        "A-n32-k5.vrp": "A/A-n32-k5.vrp",
        "B-n45-k6.vrp": "B/B-n45-k6.vrp", 
        "A-n80-k10.vrp": "A/A-n80-k10.vrp",
        
        # Advanced (n > 80)
        "X-n101-k25.vrp": "X/X-n101-k25.vrp",
        "M-n200-k17.vrp": "M/M-n200-k17.vrp"
    }
    
    os.makedirs("instances", exist_ok=True)
    
    for filename, path in instances.items():
        print(f"Downloading {filename}...")
        try:
            url = base_url + path
            response = requests.get(url)
            response.raise_for_status()
            
            with open(f"instances/{filename}", 'w') as f:
                f.write(response.text)
            print(f"  ✓ Downloaded {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_cvrp_instances()