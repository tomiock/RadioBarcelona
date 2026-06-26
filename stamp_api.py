from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # IMPORT CORS
import faiss
import json
import torch
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np
import re

# Import the model blueprint your teammate provided
from vae_model import ConvVAE 

app = FastAPI()

# --- FIX 1: ENABLE CORS ---
# This allows your frontend (Port 3000) to talk to this backend (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (change to "http://localhost:3000" in production)
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# 1. Load FAISS Index and Metadata
print("Loading FAISS index...")
index = faiss.read_index('visual_index.faiss')

metadata = []
with open('metadata.jsonl', 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

# 2. Load VAE Model
print("Loading VAE Model...")

checkpoint = torch.load('vae_best.pt', map_location=torch.device('cpu'))

if 'config' in checkpoint and 'model_state' in checkpoint:
    config = checkpoint['config']
    print(f"✅ Found config inside the model file: {config}")
    latent_dim = config.get('latent_dim', 1024)
    image_size = config.get('image_size', 128)
    in_channels = config.get('in_channels', 1)
    model_state = checkpoint['model_state']
else:
    print("⚠️ Dictionary format unknown, attempting default 1024")
    latent_dim = 1024
    image_size = 128
    in_channels = 1
    model_state = checkpoint

model = ConvVAE(image_size=image_size, latent_dim=latent_dim, in_channels=in_channels)
model.load_state_dict(model_state)
print(f"✅ Successfully loaded VAE with latent_dim={latent_dim}")

model.eval()

preprocess = T.Compose([
    T.Resize((128, 128)),
    T.Grayscale(num_output_channels=1), 
    T.ToTensor()
])

# --- FIX 2: RENAME ENDPOINT TO MATCH FRONTEND ---
@app.post("/search_upload")
async def search_stamp(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    tensor_img = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        mu, logvar = model.encode(tensor_img)
        embedding = mu.numpy().astype('float32') 
    
    D, I = index.search(embedding, k=5)
    
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    
    doc_ids = []
    for res in results:
        # Get the massive string from the JSONL (e.g., "radio_barcelona_...1925...")
        raw_id = res.get('document_id', '') or res.get('image', '')
        
        # Use Regex to extract the 4-digit year (e.g., 1925, 1953)
        match = re.search(r'(19\d{2})', raw_id)
        if match:
            year = match.group(1)
            # Map it to your prototype's simple ID format
            clean_id = f"{year}_1" 
            
            # Add it to the list if it's not already there
            if clean_id not in doc_ids:
                doc_ids.append(clean_id)
                
    return {"results": doc_ids}

@app.get("/get_stamp/{doc_id}")
async def get_stamp(doc_id: str):
    # doc_id comes in as something like "1925_1"
    # We look for an entry in our metadata that contains this ID
    # Note: We replace the '_' with '-' if needed to match your JSONL naming convention
    search_term = doc_id.replace("_", "-")
    
    for entry in metadata:
        # Check if the document_id in the JSONL contains our clean file ID
        if search_term in entry.get('document_id', ''):
            return {
                "found": True,
                "bbox": entry.get('bbox'),
                "crop_path": entry.get('crop_path')
            }
            
    return {"found": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
