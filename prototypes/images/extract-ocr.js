const { createWorker } = require('tesseract.js');
const { imageSize } = require('image-size');
const fs = require('fs');

// 1. Point this to the ORIGINAL JPG with the solid background
const imagePath = './file1/page1_original.jpg';

async function extractTextToPercentages(imagePath) {
    console.log(`Processing: ${imagePath}...`);

    const imageBuffer = fs.readFileSync(imagePath);
    const dimensions = imageSize(imageBuffer);
    const imgWidth = dimensions.width;
    const imgHeight = dimensions.height;

    // 2. Changed 'eng' to 'spa' to properly read Spanish accents and ñ
    const worker = await createWorker('spa', 1, {
        logger: m => console.log(m.status + " " + Math.round(m.progress * 100) + "%")
    });

    const result = await worker.recognize(imageBuffer);
    await worker.terminate();

    const words = result?.data?.words || [];

    if (words.length === 0) {
        console.log("\n⚠️ No words were detected in this image.");
        console.log("Raw extracted text was:\n", result?.data?.text || "[Empty]");
        return;
    }

    const ocrData = [];

    words.forEach(word => {
        const { x0, y0, x1, y1 } = word.bbox;

        const left = (x0 / imgWidth) * 100;
        const top = (y0 / imgHeight) * 100;
        const width = ((x1 - x0) / imgWidth) * 100;
        const height = ((y1 - y0) / imgHeight) * 100;

        if (word.text.trim().length > 0) {
            ocrData.push({
                text: word.text,
                top: parseFloat(top.toFixed(2)),
                         left: parseFloat(left.toFixed(2)),
                         height: parseFloat(height.toFixed(2)),
                         width: parseFloat(width.toFixed(2))
            });
        }
    });

    console.log("\n--- COPY AND PASTE THIS INTO YOUR filesData ---");
    console.log("ocr: " + JSON.stringify(ocrData, null, 2));
}

extractTextToPercentages(imagePath).catch(console.error);
