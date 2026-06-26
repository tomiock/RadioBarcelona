const fs = require('fs');
const path = require('path');
const xml2js = require('xml2js');

const PACKAGE_DIR = './seg_split_A_package';
const XML_DIR = path.join(PACKAGE_DIR, 'pagexml_tesseract');
const finalData = [];

async function parsePageXML() {
    console.log("🚀 Starting PageXML extraction...");
    const files = fs.readdirSync(XML_DIR).filter(f => f.endsWith('.xml'));
    const parser = new xml2js.Parser();

    for (const file of files) {
        const baseName = file.replace('.xml', ''); // e.g., "1925_1"
        const yearMatch = baseName.match(/^(\d{4})/);
        const year = yearMatch ? yearMatch[1] : "Unknown";

        const xmlData = fs.readFileSync(path.join(XML_DIR, file), 'utf8');
        const result = await parser.parseStringPromise(xmlData);
        
        const page = result.PcGts.Page[0];
        const pageWidth = parseInt(page.$.imageWidth);
        const pageHeight = parseInt(page.$.imageHeight);
        
        let chunks = [];
        let chunkIndex = 0;

        // Extract TextRegions and TextLines
        if (page.TextRegion) {
            page.TextRegion.forEach(region => {
                if (region.TextLine) {
                    region.TextLine.forEach(line => {
                        if (!line.TextEquiv || !line.TextEquiv[0].Unicode || !line.TextEquiv[0].Unicode[0]) return;
                        
                        // --- START OF NEW ROBUST EXTRACTION ---
                        let text = "";
                        const rawUnicode = line.TextEquiv[0].Unicode[0];
                        
                        // xml2js sometimes parses plain text as strings, and sometimes as objects
                        if (typeof rawUnicode === 'string') {
                            text = rawUnicode;
                        } else if (rawUnicode && typeof rawUnicode._ === 'string') {
                            text = rawUnicode._;
                        }

                        text = text.trim();

                        // Count valid letters (English + Spanish + Catalan)
                        const letters = text.match(/[a-zA-ZáéíóúÁÉÍÓÚñÑçÇl·l]/g);
                        const letterCount = letters ? letters.length : 0;

                        // If the line is mostly noise, less than 3 chars, or has fewer than 2 letters, drop it
                        if (text.length < 3 || letterCount < 2) {
                            return; // Skip OCR junk (empty spaces, random numbers, punctuation)
                        }

                        // Parse the polygon coordinates string: "x1,y1 x2,y2 x3,y3 x4,y4"
                        const coordsStr = line.Coords[0].$.points;
                        const points = coordsStr.split(' ').map(p => {
                            const [x, y] = p.split(',').map(Number);
                            return {x, y};
                        });

                        // Calculate standard bounding box
                        const minX = Math.min(...points.map(p => p.x));
                        const minY = Math.min(...points.map(p => p.y));
                        const maxX = Math.max(...points.map(p => p.x));
                        const maxY = Math.max(...points.map(p => p.y));

                        chunks.push({
                            chunk_id: `${baseName}_c${chunkIndex++}`,
                            text: text,
                            bbox: {
                                left: minX,
                                top: minY,
                                width: maxX - minX,
                                height: maxY - minY
                            }
                        });
                    });
                }
            });
        }

        finalData.push({
            id: baseName,
            title: `Document ${baseName.replace('_', ' - ')}`,
            year: year,
            month: "", // Add if you have month data
            thumb: `${baseName}.png`, 
            imageWidth: pageWidth,   // Vital for scaling in the UI
            imageHeight: pageHeight, // Vital for scaling in the UI
            pages: 1,
            chunks: chunks
        });
        
        console.log(`✅ Parsed ${file}: Found ${chunks.length} lines of text.`);
    }

    const jsOutput = `const filesData = ${JSON.stringify(finalData, null, 2)};`;
    fs.writeFileSync('./data.js', jsOutput);
    console.log("💾 Saved data.js. Ready to run index_data.js!");
}

parsePageXML().catch(console.error);
