// src/TTS.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  loadTextToSpeech,
  loadVoiceStyle,
  writeWavFile,
  TextToSpeech,
  Style
} from '../lib/helper';

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

export default function TTS() {
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

  const textToSpeechRef = useRef<TextToSpeech | null>(null);
  const cfgsRef = useRef<any>(null);
  const currentStyleRef = useRef<Style | null>(null);

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

  const loadStyleFromJSON = async (stylePath: string): Promise<Style> => {
    try {
      const style = await loadVoiceStyle([stylePath], true);
      return style;
    } catch (error) {
      console.error('Error loading voice style:', error);
      throw error;
    }
  };

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
    <div className="min-h-screen bg-linear-to-br from-indigo-500 to-purple-600 flex items-center justify-center p-5">
      <div className="bg-white rounded-3xl p-10 max-w-7xl w-full shadow-2xl">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">🎤 Supertonic</h1>
        <p className="text-lg text-gray-600 mb-8">Text-to-Speech with ONNX Runtime Web</p>

        <div className={`${
          statusType === 'success' ? 'bg-green-50 border-green-500 text-green-800' :
          statusType === 'error' ? 'bg-red-50 border-red-500 text-red-800' :
          'bg-blue-50 border-blue-500 text-blue-800'
        } border-l-4 rounded p-4 mb-2 transition-all flex justify-between items-center flex-wrap gap-4 min-h-12.5`}>
          <div className="flex-1 min-w-50">
            <div dangerouslySetInnerHTML={{ __html: statusMessage }} />
          </div>
          <div className={`${showBackend ? 'visible' : 'invisible'} inline-block px-3 py-1.5 ${
            backendType === 'WebGPU' ? 'bg-green-500' : 'bg-orange-500'
          } text-white rounded-xl text-sm font-semibold whitespace-nowrap ml-2`}>
            {backendType}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mt-8 items-start">
          <div className="flex flex-col">
            <div className="mb-6">
              <div className="mb-2">
                <label htmlFor="voiceStyleSelect" className="inline font-semibold text-gray-800 text-sm">
                  Voice Style:{' '}
                </label>
                <span className="text-green-600 font-bold text-sm">{voiceStyleInfo}</span>
              </div>
              <select
                id="voiceStyleSelect"
                value={voiceStylePath}
                onChange={handleVoiceStyleChange}
                disabled={isLoading}
                className="w-full p-3 border-2 border-gray-200 rounded-lg text-base transition-colors focus:outline-none focus:border-indigo-500"
              >
                {VOICE_STYLES.map((style) => (
                  <option key={style.value} value={style.value}>
                    {style.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-6">
              <label htmlFor="text" className="block font-semibold text-gray-800 mb-2 text-sm">
                Text to Synthesize:
              </label>
              <textarea
                id="text"
                placeholder="Enter the text you want to convert to speech..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full p-3 border-2 border-gray-200 rounded-lg text-base transition-colors resize-y min-h-25 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label htmlFor="totalStep" className="block font-semibold text-gray-800 mb-2 text-sm">
                  Total Steps (higher = better quality):
                </label>
                <input
                  type="number"
                  id="totalStep"
                  value={totalStep}
                  onChange={(e) => setTotalStep(Number(e.target.value))}
                  min="1"
                  max="50"
                  className="w-full p-3 border-2 border-gray-200 rounded-lg text-base transition-colors focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label htmlFor="speed" className="block font-semibold text-gray-800 mb-2 text-sm">
                  Speed (0.9-1.5 recommended):
                </label>
                <input
                  type="number"
                  id="speed"
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                  min="0.5"
                  max="2.0"
                  step="0.05"
                  className="w-full p-3 border-2 border-gray-200 rounded-lg text-base transition-colors focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <button
              onClick={generateSpeech}
              disabled={isLoading || isGenerating}
              className="w-full bg-linear-to-r from-indigo-500 to-purple-600 text-white font-semibold text-lg py-4 rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
            >
              Generate Speech
            </button>

            {errorMessage && (
              <div className="bg-red-50 text-red-700 p-4 rounded-lg mt-5">
                {errorMessage}
              </div>
            )}
          </div>

          <div className="flex flex-col flex-1">
            {!generatedAudio && !isGenerating && (
              <div className="bg-white rounded-2xl shadow-md p-16 text-center text-gray-400 transition-all hover:shadow-lg flex flex-col justify-center items-center flex-1 min-h-100">
                <div className="text-6xl mb-5 opacity-60 animate-bounce">🎤</div>
                <p className="text-lg text-gray-500 font-medium m-0">Generated speech will appear here</p>
              </div>
            )}

            {isGenerating && (
              <div className="bg-white rounded-2xl shadow-md p-16 text-center text-gray-400 transition-all hover:shadow-lg flex flex-col justify-center items-center flex-1 min-h-100">
                <div className="text-6xl mb-5 opacity-60 animate-spin">⏳</div>
                <p className="text-lg text-gray-500 font-medium m-0">Generating speech...</p>
              </div>
            )}

            {generatedAudio && (
              <div className="bg-white rounded-2xl shadow-md overflow-hidden transition-all hover:shadow-lg flex flex-col flex-1">
                <div className="p-5 bg-linear-to-br from-indigo-50 to-white border-b border-gray-100 flex-1 flex flex-col overflow-hidden">
                  <div className="text-xs uppercase tracking-wide text-indigo-600 font-semibold mb-2">
                    Input Text
                  </div>
                  <div className="text-gray-800 leading-relaxed text-sm whitespace-pre-wrap overflow-y-auto pr-2 flex-1">
                    {generatedAudio.text}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-0 bg-indigo-50">
                  <div className="p-4 flex items-center gap-2 text-sm text-gray-600 border-b border-r border-gray-200">
                    <span>📊 Audio Length</span>
                    <strong className="ml-auto text-gray-800 text-lg font-semibold">
                      {generatedAudio.audioDuration}s
                    </strong>
                  </div>
                  <div className="p-4 flex items-center gap-2 text-sm text-gray-600 border-b border-gray-200">
                    <span>⏱️ Generation Time</span>
                    <strong className="ml-auto text-gray-800 text-lg font-semibold">
                      {generatedAudio.generationTime}s
                    </strong>
                  </div>
                </div>
                <div className="p-5 bg-white">
                  <audio controls className="w-full h-12 outline-none focus:outline-2 focus:outline-indigo-500 focus:outline-offset-2 rounded">
                    <source src={generatedAudio.url} type="audio/wav" />
                  </audio>
                </div>
                <div className="p-5 pt-4 bg-white">
                  <button
                    onClick={() => downloadAudio(generatedAudio.url, 'synthesized_speech.wav')}
                    className="w-full bg-linear-to-r from-indigo-500 to-purple-600 text-white font-semibold text-base py-3 rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5 flex items-center justify-center gap-2"
                  >
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