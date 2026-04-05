import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { randomUUID } from "node:crypto";
import "dotenv/config";

// Pinecone metadata cannot contain nested objects, so we keep only supported value types.
function sanitizeMetadata(metadata) {
    return Object.fromEntries(
        Object.entries(metadata).filter(([, value]) => {
            if (
                typeof value === "string" ||
                typeof value === "number" ||
                typeof value === "boolean"
            ) {
                return true;
            }

            return (
                Array.isArray(value) &&
                value.every((item) => typeof item === "string")
            );
        })
    );
}

export async function docEmbedding(url) {
    const loader = new CheerioWebBaseLoader(url);
    const parsedDocs = await loader.load();
    const docsWithMeta = parsedDocs
        .map((doc) => {
            // Preserve the original URL as the source for each loaded document.
            doc.metadata.source = url;
            return doc;
        })
        .filter((doc) => doc.pageContent?.trim().length > 0);

    if (docsWithMeta.length === 0) {
        throw new Error(
            `No text content was loaded from ${url}. The page may block scraping or render content client-side.`
        );
    }

    console.log(`Loaded ${docsWithMeta.length} document(s) from ${url}`);
    console.log(`First document length: ${docsWithMeta[0].pageContent.length} characters`);

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 4000,
        chunkOverlap: 200,
    });

    // Split long page content into overlapping chunks so retrieval keeps enough context.
    const allSplits = (await textSplitter.splitDocuments(docsWithMeta)).filter(
        (doc) => doc.pageContent?.trim().length > 0
    );

    if (allSplits.length === 0) {
        throw new Error(
            `Text was loaded from ${url}, but chunking produced 0 documents. Check the loaded content and splitter settings.`
        );
    }

    console.log(`Prepared ${allSplits.length} chunk(s) for embedding`);

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY,
    });

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY,
    });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

    // Generate one embedding vector per chunk before building Pinecone records.
    const vectors = await embeddings.embedDocuments(
        allSplits.map((doc) => doc.pageContent)
    );

    // Each Pinecone record needs a unique id, the vector values, and primitive-safe metadata.
    const records = vectors.map((values, index) => ({
        id: randomUUID(),
        values,
        metadata: {
            ...sanitizeMetadata(allSplits[index].metadata),
            text: allSplits[index].pageContent,
        },
    }));

    if (records.length === 0) {
        throw new Error("Embedding step produced 0 vectors, so nothing can be upserted.");
    }

    await pineconeIndex.upsert({
        records,
    });

    console.log("Finished embedding document");
}

const targetUrl = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/";

try {
    await docEmbedding(targetUrl);
} catch (error) {
    console.error("Embedding pipeline failed:", error.message);
    process.exitCode = 1;
}
