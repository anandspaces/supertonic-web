// src/App.tsx
import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, Loader2, AlertCircle } from 'lucide-react';
import { VoiceChatEngine } from './lib/voiceChatEngine';
import type { SpeechRecognition } from './lib/voiceChatEngine';

interface Message {
  role: 'user' | 'assistant';
  text: string;
}

export default function VoiceChat() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [status, setStatus] = useState('Initializing...');
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [hasApiKey, setHasApiKey] = useState(false);
  const [initError, setInitError] = useState('');
  
  const engineRef = useRef<VoiceChatEngine | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Initialize Engine
  useEffect(() => {
    const initEngine = async () => {
      const engine = new VoiceChatEngine();
      engineRef.current = engine;
      
      // Check if API key exists in environment
      if (engine.hasApiKey()) {
        setHasApiKey(true);
        try {
          await engine.initializeTTS(setStatus);
          setStatus('Ready');
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          setInitError(`TTS initialization failed: ${errorMsg}`);
          setStatus('TTS initialization failed');
        }
      } else {
        setHasApiKey(false);
        setInitError('VITE_GEMINI_API_KEY environment variable is not set. Please add it to your .env file.');
        setStatus('Missing API key');
      }
    };

    initEngine();
  }, []);

  // Start Listening
  const startListening = () => {
    if (!engineRef.current) {
      alert('Engine not initialized');
      return;
    }

    if (!hasApiKey) {
      alert('Gemini API key not found in environment variables. Please set VITE_GEMINI_API_KEY.');
      return;
    }

    const recognition = engineRef.current.initializeSpeechRecognition(
      (text) => setTranscript(text),
      (error) => {
        console.error('Speech recognition error:', error);
        setStatus(`Recognition error: ${error}`);
        setIsListening(false);
      },
      () => {
        if (isListening) {
          handleSpeechEnd();
        }
      }
    );

    if (!recognition) return;

    recognitionRef.current = recognition;
    setIsListening(true);
    setTranscript('');
    setAiResponse('');
    setStatus('Listening...');
    
    recognition.start();
  };

  // Stop Listening
  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  // Handle Speech End
  const handleSpeechEnd = async () => {
    if (!transcript.trim()) {
      setStatus('No speech detected');
      setIsListening(false);
      return;
    }

    if (!engineRef.current) {
      setStatus('Engine not available');
      setIsListening(false);
      return;
    }

    setIsListening(false);
    setIsProcessing(true);
    setStatus('Processing with Gemini...');

    try {
      const response = await engineRef.current.callGeminiAPI(transcript);
      setAiResponse(response);
      
      // Keep only last 10 messages to prevent memory issues
      setConversationHistory(prev => {
        const updated: Message[] = [...prev, 
          { role: 'user' as const, text: transcript },
          { role: 'assistant' as const, text: response }
        ];
        return updated.slice(-10);
      });

      setStatus('Generating speech...');
      const wavBuffer = await engineRef.current.generateSpeech(response);
      
      setStatus('Playing response...');
      await engineRef.current.playAudio(wavBuffer);
      
      setStatus('Ready');
    } catch (error) {
      console.error('Processing error:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
      setTranscript('');
    }
  };

  // Clear conversation history
  const clearHistory = () => {
    setConversationHistory([]);
    setTranscript('');
    setAiResponse('');
  };

  return (
    <div className="min-h-screen w-full bg-linear-to-br from-purple-600 via-blue-600 to-indigo-700 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-4xl overflow-hidden">
        {/* Header */}
        <div className="bg-linear-to-r from-purple-600 to-indigo-600 p-6 text-white">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Volume2 size={32} />
            Voice Chat AI
          </h1>
          <p className="text-purple-100 mt-2">Real-time voice conversation powered by Gemini & TTS</p>
        </div>

        {/* Error Banner */}
        {!hasApiKey && (
          <div className="p-6 bg-red-50 border-b border-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-red-600 shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-semibold text-red-900 mb-1">Configuration Error</p>
                <p className="text-sm text-red-800 mb-2">{initError}</p>
                <div className="text-xs text-red-700 bg-red-100 p-3 rounded-lg font-mono">
                  <p className="mb-1">Create a <span className="font-bold">.env</span> file in your project root:</p>
                  <p className="text-red-900">VITE_GEMINI_API_KEY=your_key_here</p>
                </div>
                <p className="text-xs text-red-700 mt-2">
                  Get your API key from{' '}
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline font-semibold"
                  >
                    Google AI Studio
                  </a>
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Status Bar */}
        <div className="p-4 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isProcessing && <Loader2 size={16} className="animate-spin text-purple-600" />}
              <span className="text-sm font-medium text-gray-700">{status}</span>
            </div>
            {conversationHistory.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-gray-500 hover:text-gray-700 px-3 py-1 rounded hover:bg-gray-100 transition-colors"
              >
                Clear History
              </button>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="p-6">
          {/* Microphone Button */}
          <div className="flex justify-center mb-6">
            <button
              onClick={isListening ? stopListening : startListening}
              disabled={isProcessing || !hasApiKey}
              className={`
                relative w-32 h-32 rounded-full flex items-center justify-center
                transition-all duration-300 transform hover:scale-105
                ${isListening 
                  ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
                  : 'bg-linear-to-br from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700'
                }
                ${isProcessing || !hasApiKey ? 'opacity-50 cursor-not-allowed' : 'shadow-xl'}
                disabled:transform-none
              `}
            >
              {isListening ? (
                <MicOff size={48} className="text-white" />
              ) : (
                <Mic size={48} className="text-white" />
              )}
              {isListening && (
                <span className="absolute -bottom-8 text-sm font-semibold text-red-600">
                  Listening...
                </span>
              )}
            </button>
          </div>

          {/* Current Transcript */}
          {transcript && (
            <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-xs font-semibold text-blue-600 mb-1">You said:</p>
              <p className="text-gray-800">{transcript}</p>
            </div>
          )}

          {/* Current AI Response */}
          {aiResponse && (
            <div className="mb-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
              <p className="text-xs font-semibold text-purple-600 mb-1">AI Response:</p>
              <p className="text-gray-800">{aiResponse}</p>
            </div>
          )}

          {/* Conversation History */}
          {conversationHistory.length > 0 && (
            <div className="mt-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Conversation History</h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {conversationHistory.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg ${
                      msg.role === 'user'
                        ? 'bg-blue-50 border border-blue-200'
                        : 'bg-purple-50 border border-purple-200'
                    }`}
                  >
                    <p className="text-xs font-semibold mb-1 text-gray-600">
                      {msg.role === 'user' ? '👤 You' : '🤖 AI'}
                    </p>
                    <p className="text-sm text-gray-800">{msg.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Instructions */}
          {hasApiKey && conversationHistory.length === 0 && !transcript && !aiResponse && (
            <div className="text-center text-gray-500 mt-8">
              <p className="mb-2">Click the microphone to start talking</p>
              <p className="text-sm">The AI will respond with voice automatically</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}