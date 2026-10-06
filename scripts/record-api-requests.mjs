// Records what a client sends: a proxy between it and the server, which
// forwards every request and writes each to a log with its body and only the
// headers that describe the API (its version, betas, the client), never a key,
// a token or a cookie, and the response after it. Gate A11's recorded shapes
// were taken this way (reference/api-shapes/, made from the log by
// scripts/api-shapes.mjs).
//
//   node scripts/record-api-requests.mjs <listen port> <server port> <log.jsonl>
//
// Point the client's base URL at the listen port, with a placeholder key.

import http from "node:http";
import fs from "node:fs";

const [, , listenPort, targetPort, logFile] = process.argv;
if (!logFile) {
  console.error("usage: node scripts/record-api-requests.mjs <listen port> <server port> <log.jsonl>");
  process.exit(2);
}
const KEEP = /^(anthropic-version|anthropic-beta|content-type|user-agent|x-app|x-stainless-[a-z-]+|openai-beta|originator|accept)$/;
const SECRET = /^(x-api-key|authorization|cookie|proxy-authorization)$/;
let n = 0;

function parse(buf) {
  try {
    return JSON.parse(buf.toString("utf8"));
  } catch {
    return { unparsed: buf.toString("utf8").slice(0, 2000) };
  }
}

http
  .createServer((req, res) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      const id = ++n;
      const headers = Object.fromEntries(Object.entries(req.headers).filter(([k]) => KEEP.test(k)));
      fs.appendFileSync(logFile, JSON.stringify({ id, at: new Date().toISOString(), method: req.method, url: req.url, headers, body: body.length ? parse(body) : null }) + "\n");
      const forward = Object.fromEntries(Object.entries(req.headers).filter(([k]) => !SECRET.test(k) && k !== "host" && k !== "content-length" && k !== "transfer-encoding"));
      forward["content-length"] = body.length;
      const up = http.request({ host: "127.0.0.1", port: Number(targetPort), method: req.method, path: req.url, headers: forward }, (ur) => {
        res.writeHead(ur.statusCode, ur.headers);
        const out = [];
        ur.on("data", (c) => {
          out.push(c);
          res.write(c);
        });
        ur.on("end", () => {
          res.end();
          fs.appendFileSync(logFile, JSON.stringify({ id, status: ur.statusCode, response: Buffer.concat(out).toString("utf8") }) + "\n");
        });
      });
      up.on("error", (e) => {
        fs.appendFileSync(logFile, JSON.stringify({ id, error: String(e) }) + "\n");
        if (res.headersSent) {
          res.destroy();
          return;
        }
        res.writeHead(502, { "content-type": "text/plain" });
        res.end(String(e));
      });
      up.end(body);
    });
  })
  .listen(Number(listenPort), "127.0.0.1", () => console.log(`recording on http://127.0.0.1:${listenPort}, forwarding to port ${targetPort}`));
