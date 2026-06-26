const express = require('express');
const cors = require('cors');
const { Client } = require('@elastic/elasticsearch');
const path = require('path');

const app = express();
app.use(cors()); 
app.use(express.static(__dirname));

const client = new Client({ node: 'http://localhost:9200' });

// Bring back the AI connection for the search bar
async function getEmbedding(text) {
    const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'bge-m3', 
            prompt: text
        })
    });
    const data = await response.json();
    return data.embedding;
}

app.get('/search', async (req, res) => {
    const query = req.query.q || "";
    const page = parseInt(req.query.page) || 1; 
    const month = req.query.month || "";
    const year = req.query.year || "";
    const searchType = req.query.type || "exact"; 
    const pageSize = 10; 

    try {
        let filterConditions = [];
        if (month) filterConditions.push({ term: { month: month } });
        if (year) filterConditions.push({ term: { year: year } });

        // Fetch up to 100 chunks so we can group them properly in the backend
        let esBody = {
            from: 0,
            size: 10000 
        };

        if (query.trim() !== "") {
            if (searchType === "conceptual") {
                const queryVector = await getEmbedding(query);
                esBody.knn = {
                    field: 'text_embedding',
                    query_vector: queryVector,
                    k: 100,
                    num_candidates: 200,
                    filter: filterConditions 
                };
            } else {
                esBody.query = {
                    bool: { must: { match: { text: query } }, filter: filterConditions }
                };
            }
        } else {
            esBody.query = {
                bool: { must: { match_all: {} }, filter: filterConditions }
            };
        }

        const esResult = await client.search({ index: 'radio_barcelona_chunks', body: esBody });
        const hits = esResult.hits.hits;

        if (hits.length === 0) {
            return res.json({ hits: [], total: 0, totalPages: 0, currentPage: 1 });
        }

        // --- THE FIX: DYNAMIC RELATIVE THRESHOLD ---
        // Get the score of the absolute best match
        const topScore = hits[0]._score; 

        // Set a drop-off percentage (e.g., keep only chunks that score within 85% of the best match)
        const SCORE_TOLERANCE = 0.85; 
        const dynamicThreshold = topScore * SCORE_TOLERANCE;

        const uniqueDocuments = new Map();

        hits.forEach(hit => {
            // If conceptual, apply the dynamic cap!
            if (searchType === "conceptual" && query.trim() !== "") {
                // If this chunk's score is way worse than the top result, ignore it
                if (hit._score < dynamicThreshold) {
                    return; 
                }
            }

            const fileId = hit._source.file_id;

            if (!uniqueDocuments.has(fileId)) {
                uniqueDocuments.set(fileId, {
                    id: fileId,
                    score: hit._score,
                    matched_chunks: [hit._source] 
                });
            } else {
                uniqueDocuments.get(fileId).matched_chunks.push(hit._source); 
            }
        });

        const finalResults = Array.from(uniqueDocuments.values());

        // --- THE FIX: Manual Pagination for Unique Documents ---
        const totalDocuments = finalResults.length;
        const fromSkip = (page - 1) * pageSize;
        const paginatedResults = finalResults.slice(fromSkip, fromSkip + pageSize);

        res.json({ 
            hits: paginatedResults, 
            total: totalDocuments, 
            totalPages: Math.ceil(totalDocuments / pageSize), 
            currentPage: page
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Search failed' });
    }
});

app.listen(3000, () => {
    console.log('🚀 Hybrid Search Server running on http://localhost:3000');
});

