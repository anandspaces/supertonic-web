// src/lib/ttsEngine.ts
import type { Tensor } from 'onnxruntime-web/all';

// TTS Helper Classes
export class UnicodeProcessor {
  indexer: number[];

  constructor(indexer: number[]) {
    this.indexer = indexer;
  }

  call(textList: string[]) {
    const processedTexts = textList.map((text) => this.preprocessText(text));
    const textIdsLengths = processedTexts.map((text) => text.length);
    const maxLen = Math.max(...textIdsLengths);

    const textIds = processedTexts.map((text) => {
      const row = new Array(maxLen).fill(0);
      for (let j = 0; j < text.length; j++) {
        const codePoint = text.codePointAt(j);
        row[j] = codePoint !== undefined && codePoint < this.indexer.length ? this.indexer[codePoint] : -1;
      }
      return row;
    });

    const textMask = this.getTextMask(textIdsLengths);
    return { textIds, textMask };
  }

  preprocessText(text: string) {
    text = text.normalize('NFKD');
    const emojiPattern = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]+/gu;
    text = text.replace(emojiPattern, '');

    const replacements: Record<string, string> = {
      '–': '-', '‑': '-', '—': '-', '¯': ' ', '_': ' ',
      '\u201C': '"', '\u201D': '"', '\u2018': "'", '\u2019': "'",
      '´': "'", '`': "'", '[': ' ', ']': ' ', '|': ' ', '/': ' ',
      '#': ' ', '→': ' ', '←': ' ',
    };
    
    for (const [k, v] of Object.entries(replacements)) {
      text = text.replaceAll(k, v);
    }

    text = text.replace(/[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]/g, '');
    text = text.replace(/[♥☆♡©\\]/g, '');

    const exprReplacements: Record<string, string> = {
      '@': ' at ', 'e.g.,': 'for example, ', 'i.e.,': 'that is, ',
    };
    
    for (const [k, v] of Object.entries(exprReplacements)) {
      text = text.replaceAll(k, v);
    }

    text = text.replace(/ ,/g, ',').replace(/ \./g, '.').replace(/ !/g, '!');
    text = text.replace(/ \?/g, '?').replace(/ ;/g, ';').replace(/ :/g, ':');
    text = text.replace(/ '/g, "'");

    while (text.includes('""')) text = text.replace('""', '"');
    while (text.includes("''")) text = text.replace("''", "'");
    while (text.includes('``')) text = text.replace('``', '`');

    text = text.replace(/\s+/g, ' ').trim();

    if (!/[.!?;:,'\"')\]}…。」』】〉》›»]$/.test(text)) {
      text += '.';
    }

    return text;
  }

  getTextMask(textIdsLengths: number[]) {
    const maxLen = Math.max(...textIdsLengths);
    return this.lengthToMask(textIdsLengths, maxLen);
  }

  lengthToMask(lengths: number[], maxLen: number | null = null) {
    const actualMaxLen = maxLen || Math.max(...lengths);
    return lengths.map((len) => {
      const row = new Array(actualMaxLen).fill(0.0);
      for (let j = 0; j < Math.min(len, actualMaxLen); j++) {
        row[j] = 1.0;
      }
      return [row];
    });
  }
}

export class Style {
  ttl: Tensor;
  dp: Tensor;

  constructor(ttlTensor: Tensor, dpTensor: Tensor) {
    this.ttl = ttlTensor;
    this.dp = dpTensor;
  }
}

export class TextToSpeech {
  cfgs: any;
  textProcessor: UnicodeProcessor;
  dpOrt: any;
  textEncOrt: any;
  vectorEstOrt: any;
  vocoderOrt: any;
  sampleRate: number;

  constructor(cfgs: any, textProcessor: UnicodeProcessor, dpOrt: any, textEncOrt: any, vectorEstOrt: any, vocoderOrt: any) {
    this.cfgs = cfgs;
    this.textProcessor = textProcessor;
    this.dpOrt = dpOrt;
    this.textEncOrt = textEncOrt;
    this.vectorEstOrt = vectorEstOrt;
    this.vocoderOrt = vocoderOrt;
    this.sampleRate = cfgs.ae.sample_rate;
  }

  async _infer(textList: string[], style: Style, totalStep: number, speed = 1.05, progressCallback: ((step: number, total: number) => void) | null = null) {
    const ort = await import('onnxruntime-web/all');
    const bsz = textList.length;
    const { textIds, textMask } = this.textProcessor.call(textList);

    const textIdsFlat = new BigInt64Array(textIds.flat().map((x: number) => BigInt(x)));
    const textIdsShape = [bsz, textIds[0].length];
    const textIdsTensor = new ort.Tensor('int64', textIdsFlat, textIdsShape);

    const textMaskFlat = new Float32Array(textMask.flat(2) as number[]);
    const textMaskShape = [bsz, 1, textMask[0][0].length];
    const textMaskTensor = new ort.Tensor('float32', textMaskFlat, textMaskShape);

    const dpOutputs = await this.dpOrt.run({
      text_ids: textIdsTensor,
      style_dp: style.dp,
      text_mask: textMaskTensor,
    });

    const duration = Array.from(dpOutputs.duration.data as Float32Array);
    for (let i = 0; i < duration.length; i++) {
      duration[i] /= speed;
    }

    const textEncOutputs = await this.textEncOrt.run({
      text_ids: textIdsTensor,
      style_ttl: style.ttl,
      text_mask: textMaskTensor,
    });
    const textEmb = textEncOutputs.text_emb;

    let { xt, latentMask } = this.sampleNoisyLatent(
      duration,
      this.sampleRate,
      this.cfgs.ae.base_chunk_size,
      this.cfgs.ttl.chunk_compress_factor,
      this.cfgs.ttl.latent_dim
    );

    const latentMaskFlat = new Float32Array(latentMask.flat(2) as number[]);
    const latentMaskShape = [bsz, 1, latentMask[0][0].length];
    const latentMaskTensor = new ort.Tensor('float32', latentMaskFlat, latentMaskShape);

    const totalStepArray = new Float32Array(bsz).fill(totalStep);
    const totalStepTensor = new ort.Tensor('float32', totalStepArray, [bsz]);

    for (let step = 0; step < totalStep; step++) {
      if (progressCallback) {
        progressCallback(step + 1, totalStep);
      }

      const currentStepArray = new Float32Array(bsz).fill(step);
      const currentStepTensor = new ort.Tensor('float32', currentStepArray, [bsz]);

      const xtFlat = new Float32Array(xt.flat(2) as number[]);
      const xtShape = [bsz, xt[0].length, xt[0][0].length];
      const xtTensor = new ort.Tensor('float32', xtFlat, xtShape);

      const vectorEstOutputs = await this.vectorEstOrt.run({
        noisy_latent: xtTensor,
        text_emb: textEmb,
        style_ttl: style.ttl,
        latent_mask: latentMaskTensor,
        text_mask: textMaskTensor,
        current_step: currentStepTensor,
        total_step: totalStepTensor,
      });

      const denoised = Array.from(vectorEstOutputs.denoised_latent.data as Float32Array);
      const latentDim = xt[0].length;
      const latentLen = xt[0][0].length;
      xt = [];
      let idx = 0;
      for (let b = 0; b < bsz; b++) {
        const batch: number[][] = [];
        for (let d = 0; d < latentDim; d++) {
          const row: number[] = [];
          for (let t = 0; t < latentLen; t++) {
            row.push(denoised[idx++]);
          }
          batch.push(row);
        }
        xt.push(batch);
      }
    }

    const finalXtFlat = new Float32Array(xt.flat(2) as number[]);
    const finalXtShape = [bsz, xt[0].length, xt[0][0].length];
    const finalXtTensor = new ort.Tensor('float32', finalXtFlat, finalXtShape);

    const vocoderOutputs = await this.vocoderOrt.run({
      latent: finalXtTensor,
    });

    const wav = Array.from(vocoderOutputs.wav_tts.data as Float32Array);
    return { wav, duration };
  }

  async call(text: string, style: Style, totalStep: number, speed = 1.05, silenceDuration = 0.3, progressCallback: ((step: number, total: number) => void) | null = null) {
    if (style.ttl.dims[0] !== 1) {
      throw new Error('Single speaker text to speech only supports single style');
    }
    
    const textList = this.chunkText(text);
    let wavCat: number[] = [];
    let durCat = 0;

    for (const chunk of textList) {
      const { wav, duration } = await this._infer([chunk], style, totalStep, speed, progressCallback);

      if (wavCat.length === 0) {
        wavCat = wav;
        durCat = duration[0];
      } else {
        const silenceLen = Math.floor(silenceDuration * this.sampleRate);
        const silence = new Array(silenceLen).fill(0);
        wavCat = [...wavCat, ...silence, ...wav];
        durCat += duration[0] + silenceDuration;
      }
    }

    return { wav: wavCat, duration: [durCat] };
  }

  chunkText(text: string, maxLen = 300) {
    if (typeof text !== 'string') {
      throw new Error(`chunkText expects a string, got ${typeof text}`);
    }

    const paragraphs = text.trim().split(/\n\s*\n+/).filter((p) => p.trim());
    const chunks: string[] = [];

    for (let paragraph of paragraphs) {
      paragraph = paragraph.trim();
      if (!paragraph) continue;

      const sentences = paragraph.split(/(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+/);
      let currentChunk = '';

      for (let sentence of sentences) {
        if (currentChunk.length + sentence.length + 1 <= maxLen) {
          currentChunk += (currentChunk ? ' ' : '') + sentence;
        } else {
          if (currentChunk) {
            chunks.push(currentChunk.trim());
          }
          currentChunk = sentence;
        }
      }

      if (currentChunk) {
        chunks.push(currentChunk.trim());
      }
    }

    return chunks;
  }

  sampleNoisyLatent(duration: number[], sampleRate: number, baseChunkSize: number, chunkCompress: number, latentDim: number) {
    const bsz = duration.length;
    const maxDur = Math.max(...duration);
    const wavLenMax = Math.floor(maxDur * sampleRate);
    const wavLengths = duration.map((d) => Math.floor(d * sampleRate));

    const chunkSize = baseChunkSize * chunkCompress;
    const latentLen = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
    const latentDimVal = latentDim * chunkCompress;

    const xt: number[][][] = [];
    for (let b = 0; b < bsz; b++) {
      const batch: number[][] = [];
      for (let d = 0; d < latentDimVal; d++) {
        const row: number[] = [];
        for (let t = 0; t < latentLen; t++) {
          const u1 = Math.max(0.0001, Math.random());
          const u2 = Math.random();
          const val = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          row.push(val);
        }
        batch.push(row);
      }
      xt.push(batch);
    }

    const latentLengths = wavLengths.map((len) => Math.floor((len + chunkSize - 1) / chunkSize));
    const latentMask = this.lengthToMask(latentLengths, latentLen);

    for (let b = 0; b < bsz; b++) {
      for (let d = 0; d < latentDimVal; d++) {
        for (let t = 0; t < latentLen; t++) {
          xt[b][d][t] *= latentMask[b][0][t];
        }
      }
    }

    return { xt, latentMask };
  }

  lengthToMask(lengths: number[], maxLen: number | null = null) {
    const actualMaxLen = maxLen || Math.max(...lengths);
    return lengths.map((len) => {
      const row = new Array(actualMaxLen).fill(0.0);
      for (let j = 0; j < Math.min(len, actualMaxLen); j++) {
        row[j] = 1.0;
      }
      return [row];
    });
  }
}

// Utility Functions
export function writeWavFile(audioData: number[], sampleRate: number) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = audioData.length * 2;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  const int16Data = new Int16Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    const clamped = Math.max(-1.0, Math.min(1.0, audioData[i]));
    int16Data[i] = Math.floor(clamped * 32767);
  }

  const dataView = new Uint8Array(buffer, 44);
  dataView.set(new Uint8Array(int16Data.buffer));

  return buffer;
}

export async function loadVoiceStyle(voiceStylePaths: string[]) {
  const ort = await import('onnxruntime-web/all');
  const bsz = voiceStylePaths.length;
  const firstPath = voiceStylePaths[0].startsWith('/') ? voiceStylePaths[0] : `/${voiceStylePaths[0]}`;
  const firstResponse = await fetch(firstPath);
  
  if (!firstResponse.ok) {
    throw new Error(`Failed to load voice style: ${firstResponse.status}`);
  }
  
  const firstStyle = await firstResponse.json();
  const ttlDims = firstStyle.style_ttl.dims;
  const dpDims = firstStyle.style_dp.dims;

  const ttlDim1 = ttlDims[1];
  const ttlDim2 = ttlDims[2];
  const dpDim1 = dpDims[1];
  const dpDim2 = dpDims[2];

  const ttlSize = bsz * ttlDim1 * ttlDim2;
  const dpSize = bsz * dpDim1 * dpDim2;
  const ttlFlat = new Float32Array(ttlSize);
  const dpFlat = new Float32Array(dpSize);

  for (let i = 0; i < bsz; i++) {
    const path = voiceStylePaths[i].startsWith('/') ? voiceStylePaths[i] : `/${voiceStylePaths[i]}`;
    const response = await fetch(path);
    
    if (!response.ok) {
      throw new Error(`Failed to load voice style ${i}: ${response.status}`);
    }
    
    const voiceStyle = await response.json();
    const ttlData = voiceStyle.style_ttl.data.flat(Infinity) as number[];
    const ttlOffset = i * ttlDim1 * ttlDim2;
    ttlFlat.set(ttlData, ttlOffset);

    const dpData = voiceStyle.style_dp.data.flat(Infinity) as number[];
    const dpOffset = i * dpDim1 * dpDim2;
    dpFlat.set(dpData, dpOffset);
  }

  const ttlShape = [bsz, ttlDim1, ttlDim2];
  const dpShape = [bsz, dpDim1, dpDim2];

  const ttlTensor = new ort.Tensor('float32', ttlFlat, ttlShape);
  const dpTensor = new ort.Tensor('float32', dpFlat, dpShape);

  return new Style(ttlTensor, dpTensor);
}

export async function loadTextToSpeech(onnxDir: string, sessionOptions: any = {}) {
  const ort = await import('onnxruntime-web/all');
  
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';

  const cfgsPath = onnxDir.startsWith('/') ? onnxDir : `/${onnxDir}`;
  const cfgsResponse = await fetch(`${cfgsPath}/tts.json`);
  
  if (!cfgsResponse.ok) {
    throw new Error(`Failed to load config: ${cfgsResponse.status}`);
  }
  
  const cfgs = await cfgsResponse.json();

  const indexerPath = onnxDir.startsWith('/') ? onnxDir : `/${onnxDir}`;
  const indexerResponse = await fetch(`${indexerPath}/unicode_indexer.json`);
  
  if (!indexerResponse.ok) {
    throw new Error(`Failed to load text processor: ${indexerResponse.status}`);
  }
  
  const indexer = await indexerResponse.json();
  const textProcessor = new UnicodeProcessor(indexer);

  const dpPath = `${onnxDir}/duration_predictor.onnx`;
  const textEncPath = `${onnxDir}/text_encoder.onnx`;
  const vectorEstPath = `${onnxDir}/vector_estimator.onnx`;
  const vocoderPath = `${onnxDir}/vocoder.onnx`;

  const loadOnnx = async (path: string) => {
    const fullPath = path.startsWith('/') ? path : `/${path}`;
    return await ort.InferenceSession.create(fullPath, sessionOptions);
  };

  const [dpOrt, textEncOrt, vectorEstOrt, vocoderOrt] = await Promise.all([
    loadOnnx(dpPath),
    loadOnnx(textEncPath),
    loadOnnx(vectorEstPath),
    loadOnnx(vocoderPath),
  ]);

  const textToSpeech = new TextToSpeech(cfgs, textProcessor, dpOrt, textEncOrt, vectorEstOrt, vocoderOrt);
  return { textToSpeech, cfgs };
}