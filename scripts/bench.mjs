// Decode speed over the public endpoint, Anthropic Messages API, streamed.
//   DS41_API_KEY=... DS41_URL=https://<domain> node scripts/bench.mjs single|c8|c32
// single: two runs of one stream. cN: N streams at once.
// Each request asks for a 1,500-word essay with max_tokens 2048; output tokens include thinking.
// Decode tok/s is the output tokens divided by the time from the first delta to the end.
const KEY = process.env.DS41_API_KEY;
const BASE = process.env.DS41_URL;
const MODEL = process.env.SERVED_MODEL_NAME || "DeepSeek-V4.1-Flash-1017-abliterated";
if (!KEY || !BASE) { console.error("set DS41_API_KEY and DS41_URL"); process.exit(1); }
const URL = `${BASE}/v1/messages`;

async function streamCall(body) {
  const t0 = performance.now();
  const r = await fetch(URL, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ model: MODEL, stream: true, ...body }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  const dec = new TextDecoder();
  let buf = "", first = null, outTok = 0;
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    if (first === null && /_delta/.test(buf)) first = performance.now();
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.startsWith("data:")) continue;
      try {
        const ev = JSON.parse(line.slice(5));
        if (ev.type === "message_delta" && ev.usage) outTok = ev.usage.output_tokens ?? outTok;
      } catch {}
    }
  }
  const end = performance.now();
  return { ttft: (first - t0) / 1000, total: (end - t0) / 1000, outTok,
           decodeTps: outTok / ((end - first) / 1000) };
}

const prompt = "Write a detailed 1500-word essay on the history of the printing press, with headings.";
const mode = process.argv[2] || "single";
if (mode === "single") {
  for (let k = 0; k < 2; k++) {
    const s = await streamCall({ max_tokens: 2048, messages: [{ role: "user", content: prompt }] });
    console.log(`single run${k}: ttft=${s.ttft.toFixed(2)}s out=${s.outTok} decode=${s.decodeTps.toFixed(1)} tok/s total=${s.total.toFixed(1)}s`);
  }
} else if (/^c\d+$/.test(mode)) {
  const n = Number(mode.slice(1));
  const t0 = performance.now();
  const res = await Promise.all(Array.from({ length: n }, (_, i) =>
    streamCall({ max_tokens: 2048, messages: [{ role: "user", content: prompt + ` (variant ${i})` }] })));
  const wall = (performance.now() - t0) / 1000;
  const tot = res.reduce((a, r) => a + r.outTok, 0);
  const per = res.map(r => r.decodeTps);
  console.log(`${mode}: wall=${wall.toFixed(1)}s total_out=${tot} aggregate=${(tot / wall).toFixed(1)} tok/s per-stream mean=${(per.reduce((a, b) => a + b, 0) / n).toFixed(1)} min=${Math.min(...per).toFixed(1)}`);
} else {
  console.error("mode: single | c8 | c32 | cN");
  process.exit(1);
}
