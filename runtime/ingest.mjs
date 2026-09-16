// ingest.mjs - Sovereign model weights ingestion, HuggingFace downloader, and 4KB DMA slab slicing engine.
//
// Zero external dependencies.
// Dark Calm Terminal styling. Strictly zero em-dashes.

import { createHash } from "node:crypto";
import {
  openSync,
  readSync,
  writeSync,
  closeSync,
  statSync,
  mkdirSync,
  existsSync,
  writeFileSync,
  readFileSync,
} from "node:fs";
import { join, dirname, basename } from "node:path";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  amber: "\x1b[38;2;245;158;11m",
  green: "\x1b[38;2;34;197;94m",
  gray: "\x1b[38;2;156;163;175m",
  hairline: "\x1b[38;2;75;85;99m",
};

const SECTOR_SIZE = 4096; // 4KB DMA sector boundary

/**
 * Align an integer offset up to the next sector boundary.
 * @param {number} offset
 * @param {number} alignment
 * @returns {number}
 */
export function alignUp(offset, alignment = SECTOR_SIZE) {
  return Math.ceil(offset / alignment) * alignment;
}

/**
 * Parse SafeTensors 8-byte header and JSON directory from a Buffer or file.
 * @param {Buffer} buffer
 * @returns {{ headerLen: number, metadata: Record<string, any>, tensors: Record<string, any>, rawDataBase: number }}
 */
export function parseSafeTensorsHeader(buffer) {
  if (buffer.length < 8) {
    throw new Error("Buffer too short to contain SafeTensors 8-byte header length");
  }

  const headerLen = Number(buffer.readBigUInt64LE(0));
  if (headerLen <= 0 || headerLen > 128 * 1024 * 1024) {
    throw new Error(`Invalid SafeTensors header length: ${headerLen}`);
  }

  if (buffer.length < 8 + headerLen) {
    throw new Error(`Buffer has ${buffer.length} bytes, but header requires ${8 + headerLen} bytes`);
  }

  const jsonStr = buffer.toString("utf8", 8, 8 + headerLen);
  let parsed;
  try {
    parsed = JSON.parse(jsonStr);
  } catch (err) {
    throw new Error(`Failed to parse SafeTensors JSON header: ${err.message}`);
  }

  const metadata = parsed.__metadata__ || {};
  const tensors = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (k !== "__metadata__") {
      tensors[k] = v;
    }
  }

  return {
    headerLen,
    metadata,
    tensors,
    rawDataBase: 8 + headerLen,
  };
}

/**
 * Parse SafeTensors sharded index manifest (model.safetensors.index.json).
 * @param {string | Buffer} content
 * @returns {{ totalSize: number, weightMap: Record<string, string>, shards: string[] }}
 */
export function parseSafeTensorsIndex(content) {
  const jsonStr = typeof content === "string" ? content : content.toString("utf8");
  let parsed;
  try {
    parsed = JSON.parse(jsonStr);
  } catch (err) {
    throw new Error(`Failed to parse SafeTensors index manifest: ${err.message}`);
  }

  const weightMap = parsed.weight_map || {};
  const totalSize = parsed.metadata?.total_size ? Number(parsed.metadata.total_size) : 0;

  const shardSet = new Set(Object.values(weightMap));
  const shards = Array.from(shardSet).sort();

  return {
    totalSize,
    weightMap,
    shards,
  };
}

/**
 * Inspect SafeTensors file header without reading entire file into memory.
 * @param {string} filePath
 * @returns {{ headerLen: number, metadata: Record<string, any>, tensors: Record<string, any>, rawDataBase: number, fileSize: number }}
 */
export function inspectSafeTensorsFile(filePath) {
  const stat = statSync(filePath);
  const fd = openSync(filePath, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));

    const jsonBuf = Buffer.alloc(headerLen);
    readSync(fd, jsonBuf, 0, headerLen, 8);
    const jsonStr = jsonBuf.toString("utf8");
    const parsed = JSON.parse(jsonStr);

    const metadata = parsed.__metadata__ || {};
    const tensors = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (k !== "__metadata__") {
        tensors[k] = v;
      }
    }

    return {
      headerLen,
      metadata,
      tensors,
      rawDataBase: 8 + headerLen,
      fileSize: stat.size,
    };
  } finally {
    closeSync(fd);
  }
}

/**
 * Determine if a tensor corresponds to a routed MoE expert weight.
 * @param {string} tensorName
 * @returns {{ isExpert: boolean, layerIdx: number, expertIdx: number, subName: string }}
 */
export function parseMoEExpertRole(tensorName) {
  const match = tensorName.match(/(?:layers?\.(\d+)).*?experts?\.(\d+)\.(.+)/i);
  if (match) {
    return {
      isExpert: true,
      layerIdx: parseInt(match[1], 10),
      expertIdx: parseInt(match[2], 10),
      subName: match[3],
    };
  }
  return {
    isExpert: false,
    layerIdx: -1,
    expertIdx: -1,
    subName: tensorName,
  };
}

/**
 * Slice and repack tensor weights from a single safetensors file into 4096-byte DMA aligned slabs.
 * @param {string} sourcePath Path to source safetensors file.
 * @param {string} outputDir Target output directory.
 * @param {{
 *   dryRun?: boolean,
 *   onProgress?: (progress: { current: number, total: number, tensorName: string }) => void
 * }} options
 */
export function sliceDmaAlignedSlabs(sourcePath, outputDir, options = {}) {
  const { dryRun = false, onProgress } = options;

  if (!existsSync(sourcePath)) {
    throw new Error(`Source model file not found: ${sourcePath}`);
  }

  const info = inspectSafeTensorsFile(sourcePath);
  const tensorNames = Object.keys(info.tensors);

  if (!existsSync(outputDir) && !dryRun) {
    mkdirSync(outputDir, { recursive: true });
  }

  const baseName = basename(sourcePath, ".safetensors");
  const alignedFilePath = join(outputDir, `${baseName}.dma.anchor`);
  const manifestPath = join(outputDir, `${baseName}.anchor.index.json`);

  let currentOffset = SECTOR_SIZE;
  const records = [];
  const hasher = createHash("sha256");

  for (let i = 0; i < tensorNames.length; i++) {
    const name = tensorNames[i];
    const desc = info.tensors[name];
    const rawBegin = desc.data_offsets[0];
    const rawEnd = desc.data_offsets[1];
    const rawSize = rawEnd - rawBegin;

    const alignedOffset = alignUp(currentOffset, SECTOR_SIZE);
    const paddingBefore = alignedOffset - currentOffset;
    const alignedSize = alignUp(rawSize, SECTOR_SIZE);

    const moeRole = parseMoEExpertRole(name);

    records.push({
      name,
      shardIdx: 0,
      dtype: desc.dtype,
      shape: desc.shape,
      sourceOffset: info.rawDataBase + rawBegin,
      rawSize,
      targetOffset: alignedOffset,
      alignedSize,
      paddingBefore,
      isExpert: moeRole.isExpert,
      layerIdx: moeRole.layerIdx,
      expertIdx: moeRole.expertIdx,
      aligned_4k: true,
    });

    currentOffset = alignedOffset + rawSize;
  }

  if (dryRun) {
    return {
      manifestPath,
      alignedFilePath,
      tensorCount: records.length,
      totalBytesWritten: currentOffset,
      airgapHash: "DRY_RUN_ATTESTATION_PENDING",
      records,
    };
  }

  const srcFd = openSync(sourcePath, "r");
  const dstFd = openSync(alignedFilePath, "w");

  try {
    const headerPad = Buffer.alloc(SECTOR_SIZE, 0);
    writeSync(dstFd, headerPad, 0, SECTOR_SIZE, 0);

    const copyChunkSize = 1024 * 1024;
    const copyBuf = Buffer.alloc(copyChunkSize);

    for (let i = 0; i < records.length; i++) {
      const rec = records[i];

      if (onProgress) {
        onProgress({ current: i + 1, total: records.length, tensorName: rec.name });
      }

      if (rec.paddingBefore > 0) {
        const padBuf = Buffer.alloc(rec.paddingBefore, 0);
        writeSync(dstFd, padBuf, 0, rec.paddingBefore, rec.targetOffset - rec.paddingBefore);
      }

      let bytesToCopy = rec.rawSize;
      let srcPos = rec.sourceOffset;
      let dstPos = rec.targetOffset;

      while (bytesToCopy > 0) {
        const toRead = Math.min(copyChunkSize, bytesToCopy);
        readSync(srcFd, copyBuf, 0, toRead, srcPos);
        writeSync(dstFd, copyBuf, 0, toRead, dstPos);

        hasher.update(copyBuf.subarray(0, toRead));
        srcPos += toRead;
        dstPos += toRead;
        bytesToCopy -= toRead;
      }
    }
  } finally {
    closeSync(srcFd);
    closeSync(dstFd);
  }

  const airgapHash = hasher.digest("hex");

  const manifest = {
    format: "vitna-anchor-dma-v1",
    model: baseName,
    sourceFile: basename(sourcePath),
    totalTensors: records.length,
    sectorSize: SECTOR_SIZE,
    airgapSha256: airgapHash,
    createdAt: new Date().toISOString(),
    tensors: records.map((r) => ({
      name: r.name,
      shardIdx: 0,
      dtype: r.dtype,
      shape: r.shape,
      offset: r.targetOffset,
      sizeBytes: r.rawSize,
      alignedSize: r.alignedSize,
      isExpert: r.isExpert,
      layerIdx: r.layerIdx,
      expertIdx: r.expertIdx,
      aligned_4k: true,
    })),
  };

  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

  return {
    manifestPath,
    alignedFilePath,
    tensorCount: records.length,
    totalBytesWritten: currentOffset,
    airgapHash,
    records,
  };
}

/**
 * Slice and repack tensor weights from multi-shard safetensors files into 4096-byte DMA aligned slabs.
 * Matches the C11 engine multi-shard expert store contract (shard_idx).
 *
 * @param {string[]} shardPaths Array of paths to source safetensors shard files.
 * @param {string} outputDir Target output directory.
 * @param {{
 *   modelName?: string,
 *   dryRun?: boolean,
 *   onProgress?: (progress: { shardIdx: number, totalShards: number, current: number, total: number, tensorName: string }) => void
 * }} options
 */
export function sliceShardedDmaSlabs(shardPaths, outputDir, options = {}) {
  const { modelName = "sharded_model", dryRun = false, onProgress } = options;

  if (!existsSync(outputDir) && !dryRun) {
    mkdirSync(outputDir, { recursive: true });
  }

  const manifestPath = join(outputDir, `${modelName}.anchor.index.json`);
  const shardSummaries = [];
  const allRecords = [];
  let totalBytesAllShards = 0;

  for (let shardIdx = 0; shardIdx < shardPaths.length; shardIdx++) {
    const shardPath = shardPaths[shardIdx];
    if (!existsSync(shardPath)) {
      throw new Error(`Shard file not found: ${shardPath}`);
    }

    const info = inspectSafeTensorsFile(shardPath);
    const tensorNames = Object.keys(info.tensors);
    const shardBaseName = basename(shardPath, ".safetensors");
    const alignedShardFile = join(outputDir, `${shardBaseName}.dma.anchor`);

    let currentOffset = SECTOR_SIZE;
    const shardRecords = [];
    const hasher = createHash("sha256");

    for (let i = 0; i < tensorNames.length; i++) {
      const name = tensorNames[i];
      const desc = info.tensors[name];
      const rawBegin = desc.data_offsets[0];
      const rawEnd = desc.data_offsets[1];
      const rawSize = rawEnd - rawBegin;

      const alignedOffset = alignUp(currentOffset, SECTOR_SIZE);
      const paddingBefore = alignedOffset - currentOffset;
      const alignedSize = alignUp(rawSize, SECTOR_SIZE);
      const moeRole = parseMoEExpertRole(name);

      const record = {
        name,
        shardIdx,
        dtype: desc.dtype,
        shape: desc.shape,
        sourceOffset: info.rawDataBase + rawBegin,
        rawSize,
        targetOffset: alignedOffset,
        alignedSize,
        paddingBefore,
        isExpert: moeRole.isExpert,
        layerIdx: moeRole.layerIdx,
        expertIdx: moeRole.expertIdx,
        aligned_4k: true,
      };

      shardRecords.push(record);
      allRecords.push(record);
      currentOffset = alignedOffset + rawSize;
    }

    totalBytesAllShards += currentOffset;

    if (!dryRun) {
      const srcFd = openSync(shardPath, "r");
      const dstFd = openSync(alignedShardFile, "w");

      try {
        const headerPad = Buffer.alloc(SECTOR_SIZE, 0);
        writeSync(dstFd, headerPad, 0, SECTOR_SIZE, 0);

        const copyChunkSize = 1024 * 1024;
        const copyBuf = Buffer.alloc(copyChunkSize);

        for (let i = 0; i < shardRecords.length; i++) {
          const rec = shardRecords[i];

          if (onProgress) {
            onProgress({
              shardIdx,
              totalShards: shardPaths.length,
              current: i + 1,
              total: shardRecords.length,
              tensorName: rec.name,
            });
          }

          if (rec.paddingBefore > 0) {
            const padBuf = Buffer.alloc(rec.paddingBefore, 0);
            writeSync(dstFd, padBuf, 0, rec.paddingBefore, rec.targetOffset - rec.paddingBefore);
          }

          let bytesToCopy = rec.rawSize;
          let srcPos = rec.sourceOffset;
          let dstPos = rec.targetOffset;

          while (bytesToCopy > 0) {
            const toRead = Math.min(copyChunkSize, bytesToCopy);
            readSync(srcFd, copyBuf, 0, toRead, srcPos);
            writeSync(dstFd, copyBuf, 0, toRead, dstPos);

            hasher.update(copyBuf.subarray(0, toRead));
            srcPos += toRead;
            dstPos += toRead;
            bytesToCopy -= toRead;
          }
        }
      } finally {
        closeSync(srcFd);
        closeSync(dstFd);
      }

      shardSummaries.push({
        shardIdx,
        fileName: basename(alignedShardFile),
        sourceShard: basename(shardPath),
        tensorCount: shardRecords.length,
        totalBytesWritten: currentOffset,
        airgapSha256: hasher.digest("hex"),
      });
    } else {
      shardSummaries.push({
        shardIdx,
        fileName: `${shardBaseName}.dma.anchor`,
        sourceShard: basename(shardPath),
        tensorCount: shardRecords.length,
        totalBytesWritten: currentOffset,
        airgapSha256: "DRY_RUN_ATTESTATION_PENDING",
      });
    }
  }

  if (dryRun) {
    return {
      manifestPath,
      totalShards: shardPaths.length,
      shards: shardSummaries,
      tensorCount: allRecords.length,
      totalBytesWritten: totalBytesAllShards,
      records: allRecords,
    };
  }

  const manifest = {
    format: "vitna-anchor-sharded-dma-v1",
    model: modelName,
    totalShards: shardPaths.length,
    sectorSize: SECTOR_SIZE,
    shards: shardSummaries,
    createdAt: new Date().toISOString(),
    tensors: allRecords.map((r) => ({
      name: r.name,
      shardIdx: r.shardIdx,
      dtype: r.dtype,
      shape: r.shape,
      offset: r.targetOffset,
      sizeBytes: r.rawSize,
      alignedSize: r.alignedSize,
      isExpert: r.isExpert,
      layerIdx: r.layerIdx,
      expertIdx: r.expertIdx,
      aligned_4k: true,
    })),
  };

  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

  return {
    manifestPath,
    totalShards: shardPaths.length,
    shards: shardSummaries,
    tensorCount: allRecords.length,
    totalBytesWritten: totalBytesAllShards,
    records: allRecords,
  };
}

/**
 * Verify that all tensor boundaries in an anchor manifest satisfy 4KB alignment.
 * Supports both single-shard and multi-shard manifests.
 * @param {string} manifestPath
 * @returns {{ valid: boolean, errors: string[], tensorCount: number }}
 */
export function verifyDmaAlignment(manifestPath) {
  if (!existsSync(manifestPath)) {
    return { valid: false, errors: [`Manifest not found: ${manifestPath}`], tensorCount: 0 };
  }

  const content = readFileSync(manifestPath, "utf8");
  const manifest = JSON.parse(content);
  const errors = [];

  const totalShards = manifest.totalShards || 1;

  for (const t of manifest.tensors) {
    if (t.offset % SECTOR_SIZE !== 0) {
      errors.push(`Tensor "${t.name}" offset ${t.offset} is not aligned to ${SECTOR_SIZE} bytes boundary`);
    }
    if (t.aligned_4k !== true) {
      errors.push(`Tensor "${t.name}" missing aligned_4k flag`);
    }
    if (typeof t.shardIdx !== "number" || t.shardIdx < 0 || t.shardIdx >= totalShards) {
      errors.push(`Tensor "${t.name}" invalid shardIdx ${t.shardIdx}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    tensorCount: manifest.tensors.length,
  };
}

/**
 * Check if a remote HTTP/HTTPS resource exists via HEAD or quick GET.
 * @param {string} url
 * @returns {Promise<boolean>}
 */
export function probeRemoteUrl(url) {
  return new Promise((resolve) => {
    const parsedUrl = new URL(url);
    const client = parsedUrl.protocol === "https:" ? httpsRequest : httpRequest;

    const req = client(
      url,
      { method: "HEAD", headers: { "User-Agent": "vitna-anchor/0.1.0" } },
      (res) => {
        if (res.statusCode >= 200 && res.statusCode < 400) {
          resolve(true);
        } else {
          resolve(false);
        }
      }
    );

    req.on("error", () => resolve(false));
    req.end();
  });
}

/**
 * Stream download a remote URL following redirects with air-gap hash computation.
 * @param {string} url
 * @param {string} destPath
 * @param {{ onProgress?: (downloaded: number, total: number) => void }} options
 * @returns {Promise<{ bytesWritten: number, sha256: string }>}
 */
export function downloadFile(url, destPath, options = {}) {
  const { onProgress } = options;

  return new Promise((resolve, reject) => {
    function execute(targetUrl, redirectCount = 0) {
      if (redirectCount > 10) {
        return reject(new Error("Too many HTTP redirects encountered"));
      }

      const parsedUrl = new URL(targetUrl);
      const isHttps = parsedUrl.protocol === "https:";
      const client = isHttps ? httpsRequest : httpRequest;

      const req = client(
        targetUrl,
        {
          method: "GET",
          headers: {
            "User-Agent": "vitna-anchor/0.1.0",
            Accept: "*/*",
          },
        },
        (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            const redirectUrl = new URL(res.headers.location, targetUrl).toString();
            return execute(redirectUrl, redirectCount + 1);
          }

          if (res.statusCode !== 200) {
            return reject(new Error(`HTTP ${res.statusCode} failed to download from ${targetUrl}`));
          }

          const totalBytes = parseInt(res.headers["content-length"] || "0", 10);
          let bytesWritten = 0;
          const hasher = createHash("sha256");

          const fd = openSync(destPath, "w");

          res.on("data", (chunk) => {
            writeSync(fd, chunk, 0, chunk.length);
            hasher.update(chunk);
            bytesWritten += chunk.length;
            if (onProgress) {
              onProgress(bytesWritten, totalBytes);
            }
          });

          res.on("end", () => {
            closeSync(fd);
            resolve({
              bytesWritten,
              sha256: hasher.digest("hex"),
            });
          });

          res.on("error", (err) => {
            closeSync(fd);
            reject(err);
          });
        }
      );

      req.on("error", reject);
      req.end();
    }

    execute(url, 0);
  });
}

/**
 * Main coordinator: pull model from HuggingFace Hub or local path and slice 4KB DMA slabs.
 * Handles both single-file checkpoints and multi-shard index manifests.
 *
 * @param {string} target Model ID (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct") or local path.
 * @param {{
 *   outDir?: string,
 *   quant?: string,
 *   dryRun?: boolean,
 * }} options
 */
export async function runModelPull(target, options = {}) {
  const { outDir = "./models", dryRun = false } = options;

  console.log("\n" + ANSI.bold + "VITNA ANCHOR · MODEL INGESTION & DMA SLAB SLICER" + ANSI.reset);
  console.log(ANSI.dim + "Zero-Copy NVMe DMA Alignment · Direct I/O Optimization\n" + ANSI.reset);

  const isLocal = existsSync(target);

  if (isLocal) {
    const stat = statSync(target);

    // 1. Local index manifest (model.safetensors.index.json)
    if (stat.isFile() && target.endsWith(".index.json")) {
      const dir = dirname(target);
      const indexContent = readFileSync(target, "utf8");
      const indexInfo = parseSafeTensorsIndex(indexContent);
      const shardPaths = indexInfo.shards.map((s) => join(dir, s));

      console.log(`  Source Type       : ${ANSI.green}Local Sharded Checkpoint Index${ANSI.reset}`);
      console.log(`  Input Manifest    : ${ANSI.bold}${target}${ANSI.reset}`);
      console.log(`  Detected Shards   : ${ANSI.bold}${shardPaths.length} shard files${ANSI.reset}`);
      console.log(`  Output Directory  : ${ANSI.bold}${outDir}${ANSI.reset}\n`);

      const result = sliceShardedDmaSlabs(shardPaths, outDir, {
        modelName: basename(dir) || "sharded_model",
        dryRun,
        onProgress: ({ shardIdx, totalShards, current, total, tensorName }) => {
          const pct = ((current / total) * 100).toFixed(0);
          process.stdout.write(`\r  [Shard ${shardIdx + 1}/${totalShards}] [${pct}%] Slicing 4KB DMA slab: ${ANSI.dim}${tensorName.slice(0, 35)}${ANSI.reset}     `);
        },
      });

      console.log("\n");
      console.log(`  Status            : ${ANSI.green}COMPLETED (Multi-Shard 4KB DMA Aligned)${ANSI.reset}`);
      console.log(`  Total Shards      : ${ANSI.bold}${result.totalShards}${ANSI.reset}`);
      console.log(`  Tensors Processed : ${ANSI.bold}${result.tensorCount}${ANSI.reset}`);
      console.log(`  Index Manifest    : ${ANSI.bold}${result.manifestPath}${ANSI.reset}\n`);
      return result;
    }

    // 2. Local directory containing model.safetensors.index.json
    if (stat.isDirectory()) {
      const idxFile = join(target, "model.safetensors.index.json");
      if (existsSync(idxFile)) {
        return runModelPull(idxFile, options);
      }
    }

    // 3. Local single-file safetensors
    console.log(`  Source Type       : ${ANSI.green}Local File Checkpoint${ANSI.reset}`);
    console.log(`  Input Path        : ${ANSI.bold}${target}${ANSI.reset}`);
    console.log(`  Output Directory  : ${ANSI.bold}${outDir}${ANSI.reset}\n`);

    const result = sliceDmaAlignedSlabs(target, outDir, {
      dryRun,
      onProgress: ({ current, total, tensorName }) => {
        const pct = ((current / total) * 100).toFixed(0);
        process.stdout.write(`\r  [${pct}%] Slicing 4KB DMA slab: ${ANSI.dim}${tensorName.slice(0, 45)}${ANSI.reset}     `);
      },
    });

    console.log("\n");
    console.log(`  Status            : ${ANSI.green}COMPLETED (4KB DMA Aligned)${ANSI.reset}`);
    console.log(`  Tensors Processed : ${ANSI.bold}${result.tensorCount}${ANSI.reset}`);
    console.log(`  Aligned Output    : ${ANSI.bold}${result.alignedFilePath}${ANSI.reset}`);
    console.log(`  Index Manifest    : ${ANSI.bold}${result.manifestPath}${ANSI.reset}`);
    console.log(`  Air-Gap SHA-256   : ${ANSI.amber}${result.airgapHash.slice(0, 32)}...${ANSI.reset}\n`);
    return result;
  }

  // HuggingFace Hub remote resolution
  const repoId = target;
  console.log(`  Source Type       : ${ANSI.green}HuggingFace Hub${ANSI.reset}`);
  console.log(`  Repository ID     : ${ANSI.bold}${repoId}${ANSI.reset}`);
  console.log(`  Output Directory  : ${ANSI.bold}${outDir}${ANSI.reset}\n`);

  const indexManifestUrl = `https://huggingface.co/${repoId}/raw/main/model.safetensors.index.json`;
  const singleSafetensorUrl = `https://huggingface.co/${repoId}/resolve/main/model.safetensors`;

  // Probe if model is sharded or single-file
  const hasIndex = await probeRemoteUrl(indexManifestUrl);

  if (hasIndex) {
    console.log(`  Architecture Type : ${ANSI.bold}Sharded Frontier Checkpoint (model.safetensors.index.json)${ANSI.reset}`);

    if (dryRun) {
      console.log(`  Dry Run Plan      : ${ANSI.amber}Simulating Multi-Shard Hub Resolution & 4KB Layout${ANSI.reset}`);
      console.log(`  Remote Index URL  : ${indexManifestUrl}`);
      console.log(`  Target Directory  : ${outDir}`);
      console.log(`  DMA Sector Size   : 4096 bytes (O_DIRECT unbuffered ready)\n`);
      return {
        repoId,
        sharded: true,
        dryRun: true,
        status: "ready",
        sectorSize: SECTOR_SIZE,
      };
    }

    mkdirSync(outDir, { recursive: true });
    const localIndexFile = join(outDir, "model.safetensors.index.json");
    await downloadFile(indexManifestUrl, localIndexFile);

    const indexContent = readFileSync(localIndexFile, "utf8");
    const indexInfo = parseSafeTensorsIndex(indexContent);
    console.log(`  Detected Shards   : ${ANSI.bold}${indexInfo.shards.length} shards${ANSI.reset} (~${(indexInfo.totalSize / (1024 ** 3)).toFixed(1)} GB total)`);

    const localShardPaths = [];
    for (let i = 0; i < indexInfo.shards.length; i++) {
      const shardName = indexInfo.shards[i];
      const shardUrl = `https://huggingface.co/${repoId}/resolve/main/${shardName}`;
      const localShardPath = join(outDir, shardName);
      localShardPaths.push(localShardPath);

      console.log(`  Downloading shard [${i + 1}/${indexInfo.shards.length}]: ${shardName}...`);
      await downloadFile(shardUrl, localShardPath, {
        onProgress: (down, tot) => {
          const mb = (down / (1024 * 1024)).toFixed(1);
          const totMb = tot > 0 ? (tot / (1024 * 1024)).toFixed(1) + " MB" : "streaming";
          process.stdout.write(`\r    Streamed ${ANSI.green}${mb} MB${ANSI.reset} of ${totMb}...    `);
        },
      });
      console.log("\n");
    }

    console.log("  All shards downloaded. Repacking into multi-shard 4KB DMA slabs...\n");
    const result = sliceShardedDmaSlabs(localShardPaths, outDir, {
      modelName: basename(repoId),
      onProgress: ({ shardIdx, totalShards, current, total, tensorName }) => {
        const pct = ((current / total) * 100).toFixed(0);
        process.stdout.write(`\r  [Shard ${shardIdx + 1}/${totalShards}] [${pct}%] Slicing 4KB DMA slab: ${ANSI.dim}${tensorName.slice(0, 35)}${ANSI.reset}     `);
      },
    });

    console.log("\n");
    console.log(`  Status            : ${ANSI.green}COMPLETED (Multi-Shard 4KB DMA Aligned)${ANSI.reset}`);
    console.log(`  Total Shards      : ${ANSI.bold}${result.totalShards}${ANSI.reset}`);
    console.log(`  Tensors Processed : ${ANSI.bold}${result.tensorCount}${ANSI.reset}`);
    console.log(`  Index Manifest    : ${ANSI.bold}${result.manifestPath}${ANSI.reset}\n`);
    return result;
  }

  // Single-file fallback
  console.log(`  Architecture Type : ${ANSI.bold}Monolithic Single-File Checkpoint (model.safetensors)${ANSI.reset}`);

  if (dryRun) {
    console.log(`  Dry Run Plan      : ${ANSI.amber}Simulating Hub resolution & 4KB alignment layout${ANSI.reset}`);
    console.log(`  Target Hub URL    : ${singleSafetensorUrl}`);
    console.log(`  Target Slab File  : ${join(outDir, `${basename(repoId)}.dma.anchor`)}`);
    console.log(`  DMA Sector Size   : 4096 bytes (O_DIRECT unbuffered ready)\n`);
    return {
      repoId,
      sharded: false,
      dryRun: true,
      status: "ready",
      sectorSize: SECTOR_SIZE,
    };
  }

  mkdirSync(outDir, { recursive: true });
  const localSourcePath = join(outDir, `${repoId.replace(/[\/\\]/g, "_")}.safetensors`);

  console.log(`  Downloading checkpoint from HuggingFace...`);
  try {
    await downloadFile(singleSafetensorUrl, localSourcePath, {
      onProgress: (down, tot) => {
        const mb = (down / (1024 * 1024)).toFixed(1);
        const totMb = tot > 0 ? (tot / (1024 * 1024)).toFixed(1) + " MB" : "streaming";
        process.stdout.write(`\r  Streamed ${ANSI.green}${mb} MB${ANSI.reset} of ${totMb}...    `);
      },
    });
    console.log("\n  Download finished. Repacking into 4KB DMA slabs...\n");

    const result = sliceDmaAlignedSlabs(localSourcePath, outDir, {
      onProgress: ({ current, total, tensorName }) => {
        const pct = ((current / total) * 100).toFixed(0);
        process.stdout.write(`\r  [${pct}%] Slicing 4KB DMA slab: ${ANSI.dim}${tensorName.slice(0, 45)}${ANSI.reset}     `);
      },
    });

    console.log("\n");
    console.log(`  Status            : ${ANSI.green}COMPLETED (4KB DMA Aligned)${ANSI.reset}`);
    console.log(`  Tensors Processed : ${ANSI.bold}${result.tensorCount}${ANSI.reset}`);
    console.log(`  Aligned Output    : ${ANSI.bold}${result.alignedFilePath}${ANSI.reset}`);
    console.log(`  Index Manifest    : ${ANSI.bold}${result.manifestPath}${ANSI.reset}`);
    console.log(`  Air-Gap SHA-256   : ${ANSI.amber}${result.airgapHash.slice(0, 32)}...${ANSI.reset}\n`);
    return result;
  } catch (err) {
    console.error(`\n  ${ANSI.amber}Ingestion aborted: ${err.message}${ANSI.reset}\n`);
    throw err;
  }
}
