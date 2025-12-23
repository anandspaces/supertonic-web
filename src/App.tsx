// src/App.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  loadTextToSpeech,
  loadVoiceStyle,
  writeWavFile,
  TextToSpeech,
  Style
} from './lib/helper';

interface VoiceStyle {
  value: string;
  label: string;
}

const VOICE_STYLES: VoiceStyle[] = [
  { value: 'assets/voice_styles/M1.json', label: 'Male 1 (M1)' },
  { value: 'assets/voice_styles/M2.json', label: 'Male 2 (M2)' },
  { value: 'assets/voice_styles/F1.json', label: 'Female 1 (F1)' },
  { value: 'assets/voice_styles/F2.json', label: 'Female 2 (F2)' },
];

const DEFAULT_VOICE_STYLE_PATH = 'assets/voice_styles/M1.json';
const DEFAULT_TEXT = 'This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen.';

type StatusType = 'info' | 'success' | 'error';

interface GeneratedAudio {
  url: string;
  text: string;
  audioDuration: string;
  generationTime: string;
}

function App() {
  // State
  const [text, setText] = useState<string>(DEFAULT_TEXT);
  const [voiceStylePath, setVoiceStylePath] = useState<string>(DEFAULT_VOICE_STYLE_PATH);
  const [totalStep, setTotalStep] = useState<number>(5);
  const [speed, setSpeed] = useState<number>(1.05);
  const [statusMessage, setStatusMessage] = useState<string>('ℹ️ <strong>Loading models...</strong> Please wait...');
  const [statusType, setStatusType] = useState<StatusType>('info');
  const [backendType, setBackendType] = useState<string>('WebAssembly');
  const [showBackend, setShowBackend] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [generatedAudio, setGeneratedAudio] = useState<GeneratedAudio | null>(null);
  const [voiceStyleInfo, setVoiceStyleInfo] = useState<string>('Loading...');

  // Refs
  const textToSpeechRef = useRef<TextToSpeech | null>(null);
  const cfgsRef = useRef<any>(null);
  const currentStyleRef = useRef<Style | null>(null);

  // Helper functions
  const getFilenameFromPath = (path: string): string => {
    return path.split('/').pop() || '';
  };

  const showStatus = (message: string, type: StatusType = 'info') => {
    setStatusMessage(message);
    setStatusType(type);
  };

  const showError = (message: string) => {
    setErrorMessage(message);
  };

  const hideError = () => {
    setErrorMessage('');
  };

  // Load style from JSON
  const loadStyleFromJSON = async (stylePath: string): Promise<Style> => {
    try {
      const style = await loadVoiceStyle([stylePath], true);
      return style;
    } catch (error) {
      console.error('Error loading voice style:', error);
      throw error;
    }
  };

  // Initialize models
  useEffect(() => {
    const initializeModels = async () => {
      try {
        showStatus('ℹ️ <strong>Loading configuration...</strong>');

        const basePath = 'assets/onnx';

        let executionProvider = 'wasm';
        try {
          const result = await loadTextToSpeech(
            basePath,
            {
              executionProviders: ['webgpu'],
              graphOptimizationLevel: 'all'
            },
            (modelName: string, current: number, total: number) => {
              showStatus(`ℹ️ <strong>Loading ONNX models (${current}/${total}):</strong> ${modelName}...`);
            }
          );

          textToSpeechRef.current = result.textToSpeech;
          cfgsRef.current = result.cfgs;

          executionProvider = 'webgpu';
          setBackendType('WebGPU');
        } catch (webgpuError) {
          console.log('WebGPU not available, falling back to WebAssembly');

          const result = await loadTextToSpeech(
            basePath,
            {
              executionProviders: ['wasm'],
              graphOptimizationLevel: 'all'
            },
            (modelName: string, current: number, total: number) => {
              showStatus(`ℹ️ <strong>Loading ONNX models (${current}/${total}):</strong> ${modelName}...`);
            }
          );

          textToSpeechRef.current = result.textToSpeech;
          cfgsRef.current = result.cfgs;
        }

        showStatus('ℹ️ <strong>Loading default voice style...</strong>');

        currentStyleRef.current = await loadStyleFromJSON(DEFAULT_VOICE_STYLE_PATH);
        setVoiceStyleInfo(`${getFilenameFromPath(DEFAULT_VOICE_STYLE_PATH)} (default)`);

        showStatus(`✅ <strong>Models loaded!</strong> Using ${executionProvider.toUpperCase()}. You can now generate speech.`, 'success');
        setShowBackend(true);
        setIsLoading(false);
      } catch (error: any) {
        console.error('Error loading models:', error);
        showStatus(`❌ <strong>Error loading models:</strong> ${error.message}`, 'error');
        setIsLoading(false);
      }
    };

    initializeModels();
  }, []);

  // Handle voice style change
  const handleVoiceStyleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedValue = e.target.value;

    if (!selectedValue) return;

    try {
      setIsLoading(true);
      showStatus('ℹ️ <strong>Loading voice style...</strong>', 'info');

      currentStyleRef.current = await loadStyleFromJSON(selectedValue);
      setVoiceStylePath(selectedValue);
      setVoiceStyleInfo(getFilenameFromPath(selectedValue));

      showStatus(`✅ <strong>Voice style loaded:</strong> ${getFilenameFromPath(selectedValue)}`, 'success');
      setIsLoading(false);
    } catch (error: any) {
      showError(`Error loading voice style: ${error.message}`);

      // Restore default style
      setVoiceStylePath(DEFAULT_VOICE_STYLE_PATH);
      try {
        currentStyleRef.current = await loadStyleFromJSON(DEFAULT_VOICE_STYLE_PATH);
        setVoiceStyleInfo(`${getFilenameFromPath(DEFAULT_VOICE_STYLE_PATH)} (default)`);
      } catch (styleError) {
        console.error('Error restoring default style:', styleError);
      }

      setIsLoading(false);
    }
  };

  // Generate speech
  const generateSpeech = async () => {
    const inputText = text.trim();
    if (!inputText) {
      showError('Please enter some text to synthesize.');
      return;
    }

    if (!textToSpeechRef.current || !cfgsRef.current) {
      showError('Models are still loading. Please wait.');
      return;
    }

    if (!currentStyleRef.current) {
      showError('Voice style is not ready. Please wait.');
      return;
    }

    const startTime = Date.now();

    try {
      setIsGenerating(true);
      hideError();
      setGeneratedAudio(null);

      showStatus('ℹ️ <strong>Generating speech from text...</strong>');
      const tic = Date.now();

      const { wav, duration } = await textToSpeechRef.current.call(
        inputText,
        currentStyleRef.current,
        totalStep,
        speed,
        0.3,
        (step: number, total: number) => {
          showStatus(`ℹ️ <strong>Denoising (${step}/${total})...</strong>`);
        }
      );

      const toc = Date.now();
      console.log(`Text-to-speech synthesis: ${((toc - tic) / 1000).toFixed(2)}s`);

      showStatus('ℹ️ <strong>Creating audio file...</strong>');
      const wavLen = Math.floor(textToSpeechRef.current.sampleRate * duration[0]);
      const wavOut = wav.slice(0, wavLen);

      const wavBuffer = writeWavFile(wavOut, textToSpeechRef.current.sampleRate);
      const blob = new Blob([wavBuffer], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);

      const endTime = Date.now();
      const totalTimeSec = ((endTime - startTime) / 1000).toFixed(2);
      const audioDurationSec = duration[0].toFixed(2);

      setGeneratedAudio({
        url,
        text: inputText,
        audioDuration: audioDurationSec,
        generationTime: totalTimeSec,
      });

      showStatus('✅ <strong>Speech synthesis completed successfully!</strong>', 'success');
    } catch (error: any) {
      console.error('Error during synthesis:', error);
      showStatus(`❌ <strong>Error during synthesis:</strong> ${error.message}`, 'error');
      showError(`Error during synthesis: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadAudio = (url: string, filename: string) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  };

  return (
    <div className="container">
      <h1>🎤 Supertonic</h1>
      <p className="subtitle">Text-to-Speech with ONNX Runtime Web</p>

      <div className={`status-box ${statusType === 'success' ? 'success' : statusType === 'error' ? 'error' : ''}`}>
        <div className="status-text-wrapper">
          <div dangerouslySetInnerHTML={{ __html: statusMessage }} />
        </div>
        <div
          className={`backend-badge ${showBackend ? 'visible' : ''}`}
          style={{ background: backendType === 'WebGPU' ? '#4caf50' : '#ff9800' }}
        >
          {backendType}
        </div>
      </div>

      <div className="main-content">
        <div className="left-panel">
          <div className="section">
            <div className="ref-audio-label">
              <label htmlFor="voiceStyleSelect">Voice Style: </label>
              <span className="ref-audio-info">{voiceStyleInfo}</span>
            </div>
            <select
              id="voiceStyleSelect"
              value={voiceStylePath}
              onChange={handleVoiceStyleChange}
              disabled={isLoading}
            >
              {VOICE_STYLES.map((style) => (
                <option key={style.value} value={style.value}>
                  {style.label}
                </option>
              ))}
            </select>
          </div>

          <div className="section">
            <label htmlFor="text">Text to Synthesize:</label>
            <textarea
              id="text"
              placeholder="Enter the text you want to convert to speech..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className="params-grid">
            <div className="section">
              <label htmlFor="totalStep">Total Steps (higher = better quality):</label>
              <input
                type="number"
                id="totalStep"
                value={totalStep}
                onChange={(e) => setTotalStep(Number(e.target.value))}
                min="1"
                max="50"
              />
            </div>

            <div className="section">
              <label htmlFor="speed">Speed (0.9-1.5 recommended):</label>
              <input
                type="number"
                id="speed"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                min="0.5"
                max="2.0"
                step="0.05"
              />
            </div>
          </div>

          <button onClick={generateSpeech} disabled={isLoading || isGenerating}>
            Generate Speech
          </button>

          {errorMessage && (
            <div className="error active">{errorMessage}</div>
          )}
        </div>

        <div className="right-panel">
          <div className="results">
            {!generatedAudio && !isGenerating && (
              <div className="results-placeholder">
                <div className="results-placeholder-icon">🎤</div>
                <p>Generated speech will appear here</p>
              </div>
            )}

            {isGenerating && (
              <div className="results-placeholder generating">
                <div className="results-placeholder-icon">⏳</div>
                <p>Generating speech...</p>
              </div>
            )}

            {generatedAudio && (
              <div className="result-item">
                <div className="result-text-container">
                  <div className="result-text-label">Input Text</div>
                  <div className="result-text">{generatedAudio.text}</div>
                </div>
                <div className="result-info">
                  <div className="info-item">
                    <span>📊 Audio Length</span>
                    <strong>{generatedAudio.audioDuration}s</strong>
                  </div>
                  <div className="info-item">
                    <span>⏱️ Generation Time</span>
                    <strong>{generatedAudio.generationTime}s</strong>
                  </div>
                </div>
                <div className="result-player">
                  <audio controls>
                    <source src={generatedAudio.url} type="audio/wav" />
                  </audio>
                </div>
                <div className="result-actions">
                  <button onClick={() => downloadAudio(generatedAudio.url, 'synthesized_speech.wav')}>
                    <span>⬇️</span>
                    <span>Download WAV</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;