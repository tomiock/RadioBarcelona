# RadioBarcelona

## KPIs
- processing of all or a very large quantity of the dataset (>50%)
- test set with small quantity of pages (5-10) that can be used for qualitative results
- test set with small quantity of pages (5-10) to check for OCR quality (quantitative metric)
- test set with small quantity of pages (5-10) to check for object recognition metrics (60% mAP[50])

## Requirements
- web interface with decent performance over a large quantity of pages (>5k)
- open source pipeline to replicate or adapt to other types of historical documents

# Project Structure

## WP 1 Dataset Preparation
#### Task 1.1 Qualitative description of dataset
- Idenfity how to the pages from different years differ
#### Task 1.2 Quantitative description of dataset
- Count the total number of pages and the necessary storage

## WP 2 Data Processing
#### Task 2.1 Testing of Suitable Initial Pipeline
- Identication of candidates
- Test end evaluate outputs (vibecheck)
#### Task 2.2 Definition and Development of Pipeline
#### Task 2.3 Preparation of long-running execution
- Specifiying the formats used
- Performance metrics (pages/s)
#### Task 2.4 Data Processing
- Run process over dataset
- Test with developed metrics

## WP 3 Evaluation
#### Task 3.1 Development of Test Sets (Ground Truth)
- Clear definition of target data (e.g. pages with "sellos")
- Manual selection of data based on targets (remove them from database)
- Development of the datasets via manual/AI annotations
#### Task 3.2 Development of Metrics and Evaluation Pipeline
- Development of metrics with mock data
- Development of evaluation pipeline (processes test set and compare with ground truth)

## WP 4 Final Deliverable
#### Task 4.1 Development of Webinterface according to defined formats
- Allow the displaying of all generated data

#### Task 4.2 Stress-Testing of Webinterface
- Test with 5k pages


# Product Usage
## Requirements
Before running the software, ensure you have the following installed on your system:

- Node.js (v14 or higher) - For the web server and data processing.

- Python 3.8+ - For the FAISS visual stamp search API.

- Elasticsearch (v8.x recommended) - Running locally on port 9200.

- Ollama - Running locally on port 11434 with the bge-m3 model installed.

  - To install the model, run: ollama pull bge-m3

## Directory Structure
For the application to run successfully, your prototype data must be organized in the root of the project as follows:
```
├── seg_split_A_package/
│   ├── original_images/       # The raw vintage scans (.png)
│   ├── layer_separation/
│   │   ├── blue/              # Typewritten layers (_blue.png)
│   │   └── red/               # Handwritten layers (_red.png)
│   └── pagexml_tesseract/     # The OCR bounding box data (.xml)
├── visual_index.faiss         # FAISS vector database for stamps
├── metadata.jsonl             # Stamp metadata 
├── vae_best.pt                # Trained VAE model weights
└── vae_model.py               # VAE PyTorch architecture blueprint
```
## Installation
### 1. Node.js Dependencies

Open your terminal in the project root and install the required Node packages:
```
npm install express cors @elastic/elasticsearch xml2js
```
### 2. Python Dependencies

Install the required Python libraries for the computer vision API:
```
pip install fastapi uvicorn faiss-cpu torch torchvision Pillow numpy python-multipart
```
## Data Preparation Pipeline
Before running the server for the first time, you must parse the raw XML data and load it into your Elasticsearch database.

Extract the Text & Bounding Boxes:
This reads the pagexml_tesseract folder and generates a data.js file for the frontend.
```
node build_data.js
```
Index the Data (Conceptual & Exact Search):
This sends the text to Ollama to create AI embeddings, and saves everything into Elasticsearch. (Note: Ollama and Elasticsearch must be running before you execute this).
```
node index_data.js
```
## Running the Application

To use the full Tri-Modal search engine, you need to spin up the two local servers.

Terminal 1: Start the Visual Stamp Search API (Python)
```
uvicorn stamp_api:app --reload --port 8000
```
Terminal 2: Start the Main Search Server (Node.js)
```
node server.js
```
## Usage

Once both servers are running, open your web browser and navigate to:

-> http://localhost:3000/in.html

Search Modes:

- Exact Match (Text): Select "Exact Match" in the dropdown to search for specific words using Elasticsearch BM25.

- Conceptual Search (Semantics): Select "Conceptual" to search for broad ideas (e.g., "politics", "war"). The AI will find text blocks that share that meaning.

- Visual Stamp Search: Click the Paperclip Icon in the search bar to upload a local crop of a stamp. The FAISS engine will instantly find and display visually similar documents in the archive.

Viewer Mode:

Click on any document in the grid to open the Projector Layer UI. You can toggle search highlights on/off, or switch to "Separated Layers" to view the isolated handwritten (red) and typewritten (blue) extractions side-by-side. Click on any highlighted red stamp to query the database for similar stamps.
