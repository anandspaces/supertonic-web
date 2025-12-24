// src/lib/voiceChatEngine.ts
import { TextToSpeech, Style, loadTextToSpeech, loadVoiceStyle, writeWavFile } from './ttsEngine';
import { GEMINI_API_KEY, GEMINI_MODEL } from '../constants';

// Type definitions for Speech Recognition
interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

export interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

interface SpeechRecognitionConstructor {
  new (): SpeechRecognition;
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

// Voice Chat Engine Class
export class VoiceChatEngine {
  private ttsEngine: TextToSpeech | null = null;
  private voiceStyle: Style | null = null;
  private apiKey: string;

  constructor() {
    this.apiKey = GEMINI_API_KEY;
  }

  hasApiKey(): boolean {
    return !!this.apiKey;
  }

  async initializeTTS(onStatusUpdate?: (status: string) => void): Promise<string> {
    try {
      if (onStatusUpdate) {
        onStatusUpdate('Loading TTS models...');
      }
      
      const basePath = 'assets/onnx';
      let executionProvider = 'wasm';
      
      try {
        const result = await loadTextToSpeech(basePath, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all'
        });
        
        this.ttsEngine = result.textToSpeech;
        executionProvider = 'webgpu';
      } catch (webgpuError) {
        console.log('WebGPU not available, using WebAssembly');
        const result = await loadTextToSpeech(basePath, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        this.ttsEngine = result.textToSpeech;
      }

      this.voiceStyle = await loadVoiceStyle(['assets/voice_styles/M1.json']);
      
      const status = `TTS ready (${executionProvider.toUpperCase()})`;
      if (onStatusUpdate) {
        onStatusUpdate(status);
      }
      return status;
    } catch (error) {
      const errorMsg = `TTS Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      if (onStatusUpdate) {
        onStatusUpdate(errorMsg);
      }
      throw new Error(errorMsg);
    }
  }

  initializeSpeechRecognition(
    onResult: (transcript: string) => void,
    onError: (error: string) => void,
    onEnd: () => void
  ): SpeechRecognition | null {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      onError('Speech recognition not supported in this browser');
      return null;
    }

    const SpeechRecognitionClass = (window.SpeechRecognition || window.webkitSpeechRecognition) as SpeechRecognitionConstructor;
    const recognition = new SpeechRecognitionClass();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      onResult(finalTranscript || interimTranscript);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      onError(event.error);
    };

    recognition.onend = onEnd;

    return recognition;
  }

  async callGeminiAPI(text: string): Promise<string> {
    if (!this.apiKey) {
      throw new Error('Gemini API key not found. Please set VITE_GEMINI_API_KEY in your environment variables');
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${this.apiKey}`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: text
          }]
        }]
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || `Gemini API request failed: ${response.status}`);
    }

    const data = await response.json();
    return data.candidates[0].content.parts[0].text;
  }

  async generateSpeech(text: string): Promise<ArrayBuffer> {
    if (!this.ttsEngine || !this.voiceStyle) {
      throw new Error('TTS engine not initialized');
    }

    const { wav } = await this.ttsEngine.call(
      text,
      this.voiceStyle,
      5,
      1.05,
      0.3
    );

    const wavLen = Math.floor(this.ttsEngine.sampleRate * wav.length / this.ttsEngine.sampleRate);
    const wavOut = wav.slice(0, wavLen);
    return writeWavFile(wavOut, this.ttsEngine.sampleRate);
  }

  playAudio(wavBuffer: ArrayBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      const blob = new Blob([wavBuffer], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

      audio.onended = () => {
        URL.revokeObjectURL(url);
        resolve();
      };

      audio.onerror = (error) => {
        URL.revokeObjectURL(url);
        reject(error);
      };

      audio.play().catch(reject);
    });
  }
}