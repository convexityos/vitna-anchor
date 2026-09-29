// Striping: splits a file round-robin, in fixed-size chunks, across several
// directories, which may sit on different drives, and records the layout.
//
// Whether reading a striped file back is faster than reading one drive has not
// been measured here. Nothing in this repository reads striped files yet.
//
// Zero external dependencies.
// Dark Calm Terminal styling. Strictly zero em-dashes.

import { openSync, readSync, writeSync, closeSync, mkdirSync, existsSync, statSync } from "node:fs";
import { basename, join } from "node:path";
import { createHash } from "node:crypto";

export const DEFAULT_CHUNK_SIZE = 4096; // 4KB DMA standard sector

export function sliceStripedDmaSlabs(sourcePath, drivePaths = [], options = {}) {
  if (!drivePaths || drivePaths.length < 2) {
    throw new Error("Multi-drive striping requires at least 2 distinct drive paths.");
  }

  const chunkSize = options.chunkSizeBytes || DEFAULT_CHUNK_SIZE;
  if (chunkSize % 4096 !== 0) {
    throw new Error(`Chunk size (${chunkSize}) must be a multiple of 4096 bytes for direct DMA.`);
  }

  // A missing source is an error. It used to be striped as 16 MB of made-up
  // bytes, so a mistyped path reported success.
  if (!existsSync(sourcePath)) {
    throw new Error(`Source file not found: ${sourcePath}`);
  }

  const modelName = basename(sourcePath).replace(/\.(dma\.anchor|gguf|safetensors|bin)$/i, "");
  const numDrives = drivePaths.length;

  // Verify / create target drive output directories
  for (const drive of drivePaths) {
    if (!existsSync(drive)) {
      mkdirSync(drive, { recursive: true });
    }
  }

  const totalBytes = statSync(sourcePath).size;
  const inFd = openSync(sourcePath, "r");

  const perDriveFiles = [];
  const perDriveHashes = [];
  const perDriveBytes = [];

  for (let i = 0; i < numDrives; i++) {
    const outPath = join(drivePaths[i], `${modelName}.stripe-${i}.dma.anchor`);
    const outFd = openSync(outPath, "w");
    perDriveFiles.push({ path: outPath, fd: outFd });
    perDriveHashes.push(createHash("sha256"));
    perDriveBytes.push(0);
  }

  const masterHash = createHash("sha256");
  const buffer = Buffer.alloc(chunkSize);
  let bytesRemaining = totalBytes;
  let chunkIndex = 0;

  try {
    while (bytesRemaining > 0) {
      const driveIdx = chunkIndex % numDrives;
      const bytesToRead = Math.min(bytesRemaining, chunkSize);

      const bytesRead = readSync(inFd, buffer, 0, bytesToRead, null);
      if (bytesRead <= 0) break;
      // Pad the last chunk with zeros to the full chunk size
      if (bytesRead < chunkSize) {
        buffer.fill(0, bytesRead, chunkSize);
      }

      masterHash.update(buffer);
      perDriveHashes[driveIdx].update(buffer);

      writeSync(perDriveFiles[driveIdx].fd, buffer, 0, chunkSize);
      perDriveBytes[driveIdx] += chunkSize;

      bytesRemaining -= bytesToRead;
      chunkIndex++;
    }
  } finally {
    // Closing on the way out: a failure here has nothing left to affect.
    try { closeSync(inFd); } catch { /* see above */ }
    for (const f of perDriveFiles) {
      try { closeSync(f.fd); } catch { /* see above */ }
    }
  }

  const partitions = perDriveFiles.map((f, i) => ({
    driveIndex: i,
    driveRoot: drivePaths[i],
    path: f.path,
    bytesWritten: perDriveBytes[i],
    sectorCount: perDriveBytes[i] / 4096,
    sha256: perDriveHashes[i].digest("hex"),
  }));

  const manifest = {
    manifestVersion: "1.0.0",
    engine: "vitna-anchor-stripe",
    model: modelName,
    chunkSizeBytes: chunkSize,
    totalInputBytes: totalBytes,
    totalSectorPaddedBytes: chunkIndex * chunkSize,
    driveCount: numDrives,
    partitions,
    // SHA-256 of every chunk in stripe order: the source file, zero-padded to
    // a whole number of chunks.
    stripedDataSha256: masterHash.digest("hex"),
    generatedAt: new Date().toISOString(),
  };

  return manifest;
}

export function formatStripeSummary(manifest, ansi = {}) {
  const reset = ansi.reset || "";
  const bold = ansi.bold || "";
  const dim = ansi.dim || "";
  const green = ansi.green || "";
  const hairline = ansi.hairline || "";

  const lines = [
    hairline + "── [ STRIPE WRITTEN ] ─────────────────────────────────────────────────────" + reset,
    `  Source File       : ${bold}${manifest.model}${reset}`,
    `  Directories       : ${green}${manifest.driveCount}, chunks written round-robin${reset}`,
    `  Chunk Size        : ${manifest.chunkSizeBytes} bytes`,
    `  Total Stored Size : ${(manifest.totalSectorPaddedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`,
    `  Striped SHA-256   : ${dim}${manifest.stripedDataSha256}${reset} (source zero-padded to whole chunks)`,
    "",
    bold + "  DRIVE PARTITIONS:" + reset,
  ];

  for (const p of manifest.partitions) {
    const sizeMb = (p.bytesWritten / (1024 * 1024)).toFixed(1);
    lines.push(`    [Drive ${p.driveIndex}] ${p.path} (${sizeMb} MB, ${p.sectorCount} sectors)`);
  }

  lines.push(hairline + "─".repeat(75) + reset);
  return lines.join("\n");
}
