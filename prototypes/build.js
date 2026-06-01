const { createWorker } = require('tesseract.js');
const { imageSize } = require('image-size');
const fs = require('fs');
const path = require('path');

const IMAGES_DIR = './images';
const OUTPUT_FILE = './data.js';

async function buildApp() {
    console.log('🚀 Starting the automated build process...\n');
    const filesData = {};

    const worker = await createWorker('spa', 1, {
        logger: m => process.stdout.write(`\r[OCR] ${m.status}: ${Math.round(m.progress * 100)}%`)
    });
    console.log('\n');

    const folders = fs.readdirSync(IMAGES_DIR, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

    for (const folder of folders) {
        console.log(`\n📂 Processing Document: ${folder}`);

        filesData[folder] = {
            id: folder,
            title: folder.toUpperCase(),
            thumb: '',
            pages: []
        };

        const folderPath = path.join(IMAGES_DIR, folder);
        const files = fs.readdirSync(folderPath);

        // Still using _original to count the number of pages
        const originals = files.filter(f => f.endsWith('_original.jpg')).sort();

        if (originals.length === 0) {
            console.log(`   ⚠️ No _original.jpg files found in ${folder}. Skipping.`);
            continue;
        }

        filesData[folder].thumb = `images/${folder}/${originals[0]}`;

        for (const original of originals) {
            const pagePrefix = original.split('_original.jpg')[0];

            const originalPath = path.join(folderPath, original);
            const l1Path = path.join(folderPath, `${pagePrefix}_layer1.png`);
            const l2Path = path.join(folderPath, `${pagePrefix}_layer2.png`);

            const ocrData = [];

            // Check if layer 1 exists before scanning
            if (fs.existsSync(l1Path)) {
                console.log(`   📄 Scanning: ${pagePrefix}_layer1.png`);

                // Read Original into the buffer instead of Layer 1 (Tesseract struggles with transparency)
                const imageBuffer = fs.readFileSync(originalPath);
                const dimensions = imageSize(imageBuffer);
                const imgWidth = dimensions.width;
                const imgHeight = dimensions.height;

                const result = await worker.recognize(imageBuffer, {}, { blocks: true });
                const blocks = result?.data?.blocks || [];
                const words = blocks.flatMap(b => b.paragraphs.flatMap(p => p.lines.flatMap(l => l.words)));

                words.forEach(word => {
                    const { x0, y0, x1, y1 } = word.bbox;
                    if (word.text.trim().length > 0) {
                        ocrData.push({
                            text: word.text,
                            top: parseFloat(((y0 / imgHeight) * 100).toFixed(2)),
                                     left: parseFloat(((x0 / imgWidth) * 100).toFixed(2)),
                                     height: parseFloat((((y1 - y0) / imgHeight) * 100).toFixed(2)),
                                     width: parseFloat((((x1 - x0) / imgWidth) * 100).toFixed(2))
                        });
                    }
                });
            } else {
                console.log(`   ⚠️ Skipping OCR: ${pagePrefix}_layer1.png not found.`);
            }

            filesData[folder].pages.push({
                original: `images/${folder}/${original}`,
                l1: fs.existsSync(l1Path) ? `images/${folder}/${pagePrefix}_layer1.png` : '',
                                         l2: fs.existsSync(l2Path) ? `images/${folder}/${pagePrefix}_layer2.png` : '',
                                         ocr: ocrData
            });
        }
    }

    await worker.terminate();

    console.log(`\n💾 Saving data to ${OUTPUT_FILE}...`);
    const jsContent = `const filesData = ${JSON.stringify(filesData, null, 4)};`;
    fs.writeFileSync(OUTPUT_FILE, jsContent);

    console.log(`✅ Build Complete! Open index.html in your browser.`);
}

buildApp().catch(console.error);
