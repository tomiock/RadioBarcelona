const { Client } = require('@elastic/elasticsearch');
const fs = require('fs');

const client = new Client({ node: 'http://localhost:9200' });

async function getEmbedding(text) {
    const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'bge-m3', prompt: text })
    });
    const data = await response.json();
    return data.embedding;
}

async function run() {
    console.log("🚀 Connecting to Elasticsearch for Final Indexing...");

    const rawData = fs.readFileSync('./data.js', 'utf8');
    const jsonString = rawData.replace('const filesData = ', '').trim().replace(/;$/, '');
    const filesData = JSON.parse(jsonString);

    const indexName = 'radio_barcelona_chunks';

    try {
        await client.indices.delete({ index: indexName });
        console.log(`🗑️  Deleted old index: ${indexName}`);
    } catch (err) {}

    await client.indices.create({
        index: indexName,
        body: {
            mappings: {
                properties: {
                    chunk_id: { type: 'keyword' },
                    file_id: { type: 'keyword' },
                    text: { type: 'text' },
                    imageWidth: { type: 'integer' },  // STRICTURE ADDED
                    imageHeight: { type: 'integer' }, // STRICTURE ADDED
                    bbox: { type: 'object' },
                    text_embedding: { type: 'dense_vector', dims: 1024, index: true, similarity: 'cosine' }
                }
            }
        }
    });

    for (const doc of filesData) {
        for (const chunk of doc.chunks) {
            console.log(`🧠 Generating embedding for: ${chunk.chunk_id}...`);
            const vector = await getEmbedding(chunk.text);

            await client.index({
                index: indexName,
                id: chunk.chunk_id, 
                body: {
                    chunk_id: chunk.chunk_id,
                    file_id: doc.id,
                    title: doc.title,
                    month: doc.month,
                    year: doc.year,
                    imageWidth: doc.imageWidth,   // SAVING DIMENSIONS
                    imageHeight: doc.imageHeight, // SAVING DIMENSIONS
                    text: chunk.text,
                    bbox: chunk.bbox,
                    text_embedding: vector 
                }
            });
            console.log(`📤 Uploaded: ${chunk.chunk_id}`);
        }
    }

    await client.indices.refresh({ index: indexName });
    console.log("🎉 All chunks successfully embedded and loaded!");
}

run().catch(console.error);
