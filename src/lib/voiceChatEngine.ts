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

const INTERVIEWER_PROMPT = `You are a professional, friendly interviewer conducting a conversation. Your role is to:
- Ask thoughtful, engaging questions
- Listen actively and respond naturally
- Keep responses concise (1-3 sentences typically)
- Show genuine interest in the person's answers
- Follow up on interesting points
- Maintain a warm, conversational tone
- Avoid being overly formal or robotic

Start by greeting the person warmly and asking an opening question to get to know them better.`;

// Voice Chat Engine Class
export class VoiceChatEngine {
  private ttsEngine: TextToSpeech | null = null;
  private voiceStyle: Style | null = null;
  private apiKey: string;
  private conversationHistory: Array<{ role: string; text: string }> = [];

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
    
    // Build conversation context with interviewer prompt
    const contents = [
      {
        role: 'user',
        parts: [{ text: INTERVIEWER_PROMPT }]
      },
      {
        role: 'model',
        parts: [{ text: 'Hello! I\'m delighted to chat with you today. What brings you here, and what would you like to talk about?' }]
      }
    ];

    // Add conversation history
    this.conversationHistory.forEach(msg => {
      contents.push({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.text }]
      });
    });

    // Add current user message
    contents.push({
      role: 'user',
      parts: [{ text }]
    });

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.9,
          maxOutputTokens: 200,
        }
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || `Gemini API request failed: ${response.status}`);
    }

    const data = await response.json();
    const responseText = data.candidates[0].content.parts[0].text;

    // Update conversation history
    this.conversationHistory.push({ role: 'user', text });
    this.conversationHistory.push({ role: 'assistant', text: responseText });

    // Keep only last 10 messages
    if (this.conversationHistory.length > 10) {
      this.conversationHistory = this.conversationHistory.slice(-10);
    }

    return responseText;
  }

  async streamGeminiAPI(
    text: string,
    onChunk: (chunk: string) => void,
    onComplete: (fullText: string) => void,
    signal?: AbortSignal
  ): Promise<void> {
    if (!this.apiKey) {
      throw new Error('Gemini API key not found. Please set VITE_GEMINI_API_KEY in your environment variables');
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:streamGenerateContent?key=${this.apiKey}`;
    
    // Build conversation context with interviewer prompt
    const contents = [
      {
        role: 'user',
        parts: [{ text: INTERVIEWER_PROMPT }]
      },
      {
        role: 'model',
        parts: [{ text: 'Hello! I\'m delighted to chat with you today. What brings you here, and what would you like to talk about?' }]
      }
    ];

    // Add conversation history
    this.conversationHistory.forEach(msg => {
      contents.push({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.text }]
      });
    });

    // Add current user message
    contents.push({
      role: 'user',
      parts: [{ text }]
    });

    console.log('[STREAMING] Starting request to Gemini...');

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 8192,
        }
      }),
      signal
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[STREAMING] Error response:', errorText);
      throw new Error(`Gemini API request failed: ${response.status} - ${errorText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No reader available');
    }

    let accumulatedText = '';
    let chunkCount = 0;
    let jsonBuffer = '';
    let braceCount = 0;
    let inJson = false;
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('[STREAMING] Stream ended');
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      
      // Process character by character to handle multi-line JSON
      for (let i = 0; i < chunk.length; i++) {
        const char = chunk[i];
        
        if (char === '{') {
          braceCount++;
          inJson = true;
        }
        
        if (inJson) {
          jsonBuffer += char;
        }
        
        if (char === '}') {
          braceCount--;
          
          // When braces are balanced, we have a complete JSON object
          if (braceCount === 0 && inJson) {
            try {
              const data = JSON.parse(jsonBuffer);
              
              // Log the complete parsed object for debugging
              console.log('[STREAMING] Parsed object:', JSON.stringify(data, null, 2));
              
              // Try to extract text from various possible locations
              const candidate = data.candidates?.[0];
              let chunkText = null;
              
              // Check content.parts[].text (normal response)
              if (candidate?.content?.parts) {
                for (const part of candidate.content.parts) {
                  if (part.text) {
                    chunkText = part.text;
                    break;
                  }
                }
              }
              
              if (chunkText) {
                chunkCount++;
                console.log(`[STREAMING] Chunk ${chunkCount}:`, chunkText);
                accumulatedText += chunkText;
                onChunk(chunkText);
              } else {
                // Check for error or finish reason
                if (candidate?.finishReason) {
                  console.log('[STREAMING] Finish reason:', candidate.finishReason);
                }
                if (data.promptFeedback) {
                  console.log('[STREAMING] Prompt feedback:', data.promptFeedback);
                }
                if (candidate?.content) {
                  console.log('[STREAMING] Content structure:', JSON.stringify(candidate.content, null, 2));
                }
              }
            } catch (e) {
              console.warn('[STREAMING] Failed to parse JSON:', jsonBuffer.substring(0, 100), e);
            }
            
            // Reset for next JSON object
            jsonBuffer = '';
            inJson = false;
          }
        }
      }
    }

    console.log(`[STREAMING] Complete. Total chunks: ${chunkCount}, Total text length: ${accumulatedText.length}`);

    if (!accumulatedText) {
      throw new Error('No response received from Gemini API. The model may have blocked the content or encountered an error. Try using a different model or adjusting the prompt.');
    }

    // Update conversation history
    this.conversationHistory.push({ role: 'user', text });
    this.conversationHistory.push({ role: 'assistant', text: accumulatedText });

    // Keep only last 10 messages
    if (this.conversationHistory.length > 10) {
      this.conversationHistory = this.conversationHistory.slice(-10);
    }

    onComplete(accumulatedText);
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

  clearHistory(): void {
    this.conversationHistory = [];
  }

  getConversationHistory() {
    return [...this.conversationHistory];
  }
}