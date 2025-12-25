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
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const engineRef = useRef<VoiceChatEngine | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const finalTranscriptRef = useRef<string>('');
  const shouldProcessRef = useRef<boolean>(false);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Logger function
  const log = (message: string, type: 'info' | 'error' | 'warn' = 'info') => {
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
    const logMessage = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
    console.log(logMessage);
  };

  // Initialize Engine
  useEffect(() => {
    log('=== APP COMPONENT MOUNTED ===');
    const initEngine = async () => {
      log('Creating VoiceChatEngine instance...');
      const engine = new VoiceChatEngine();
      engineRef.current = engine;
      
      log('Checking for API key...');
      if (engine.hasApiKey()) {
        log('API key found, initializing TTS...');
        setHasApiKey(true);
        try {
          log('Starting TTS initialization...');
          const ttsStatus = await engine.initializeTTS((status) => {
            log(`TTS Status Update: ${status}`);
            setStatus(status);
          });
          log(`TTS initialized successfully: ${ttsStatus}`);
          setStatus('Ready');
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          log(`TTS initialization failed: ${errorMsg}`, 'error');
          setInitError(`TTS initialization failed: ${errorMsg}`);
          setStatus('TTS initialization failed');
        }
      } else {
        log('API key not found', 'error');
        setHasApiKey(false);
        setInitError('Environment variable is not set.');
        setStatus('Missing API key');
      }
    };

    initEngine();

    return () => {
      log('Component unmounting, cleaning up...');
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Process audio queue for chunked TTS
  const processAudioQueue = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);

    while (audioQueueRef.current.length > 0) {
      const textChunk = audioQueueRef.current.shift();
      if (textChunk && engineRef.current) {
        try {
          log(`Generating speech for chunk: "${textChunk.substring(0, 30)}..."`);
          const wavBuffer = await engineRef.current.generateSpeech(textChunk);
          await engineRef.current.playAudio(wavBuffer);
        } catch (error) {
          log(`TTS error: ${error}`, 'error');
        }
      }
    }

    isPlayingRef.current = false;
    setIsSpeaking(false);
  };

  // Start Listening
  const startListening = () => {
    log('=== START LISTENING CLICKED ===');
    
    if (!engineRef.current) {
      log('Engine not initialized', 'error');
      alert('Engine not initialized');
      return;
    }

    if (!hasApiKey) {
      log('API key missing, cannot start', 'error');
      alert('API key is missing!');
      return;
    }

    log('Initializing speech recognition...');
    finalTranscriptRef.current = '';
    shouldProcessRef.current = true;
    
    const recognition = engineRef.current.initializeSpeechRecognition(
      (text) => {
        log(`Recognition result: "${text.substring(0, 50)}..."`);
        setTranscript(text);
        
        if (text.trim()) {
          finalTranscriptRef.current = text.trim();
          log(`Updated finalTranscriptRef: "${finalTranscriptRef.current.substring(0, 50)}..."`);
        }
      },
      (error) => {
        log(`Speech recognition error: ${error}`, 'error');
        setStatus(`Recognition error: ${error}`);
        setIsListening(false);
        
        if (error === 'not-allowed') {
          alert('Microphone access denied. Please allow microphone access.');
        }
      },
      () => {
        log('Recognition ended event triggered');
        log(`shouldProcessRef: ${shouldProcessRef.current}`);
        log(`Final transcript at end: "${finalTranscriptRef.current.substring(0, 50)}..."`);
        
        if (shouldProcessRef.current && finalTranscriptRef.current) {
          log('Triggering handleSpeechEnd...');
          handleSpeechEnd();
        } else {
          log(`Skipping processing - shouldProcess: ${shouldProcessRef.current}, hasTranscript: ${!!finalTranscriptRef.current}`, 'warn');
          setIsListening(false);
        }
      }
    );

    if (!recognition) {
      log('Failed to create recognition instance', 'error');
      return;
    }

    recognitionRef.current = recognition;
    setIsListening(true);
    setTranscript('');
    setAiResponse('');
    setStatus('Listening...');
    
    log('Starting recognition...');
    try {
      recognition.start();
      log('Recognition started successfully');
    } catch (error) {
      log(`Failed to start recognition: ${error}`, 'error');
      setIsListening(false);
      setStatus('Failed to start listening');
    }
  };

  // Stop Listening
  const stopListening = () => {
    log('=== STOP LISTENING CLICKED ===');
    if (recognitionRef.current) {
      log('Stopping recognition and will process transcript...');
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      log('No recognition to stop', 'warn');
    }
  };

  // Handle Speech End with streaming
  const handleSpeechEnd = async () => {
    log('=== HANDLE SPEECH END CALLED ===');
    
    const textToProcess = finalTranscriptRef.current;
    log(`Text to process: "${textToProcess.substring(0, 100)}..."`);
    
    if (!textToProcess) {
      log('No transcript available, aborting', 'warn');
      setStatus('No speech detected');
      setIsListening(false);
      return;
    }

    if (!engineRef.current) {
      log('Engine not available', 'error');
      setStatus('Engine not available');
      setIsListening(false);
      return;
    }

    setIsListening(false);
    setIsProcessing(true);
    setStatus('Processing...');
    setAiResponse('');
    log('Starting AI processing with streaming...');

    abortControllerRef.current = new AbortController();
    let wordBuffer = '';
    let fullResponse = '';

    try {
      await engineRef.current.streamGeminiAPI(
        textToProcess,
        (chunk) => {
          // On each chunk received
          fullResponse += chunk;
          wordBuffer += chunk;
          setAiResponse(fullResponse);

          // Split into ~8-word chunks for TTS
          const words = wordBuffer.split(' ');
          if (words.length >= 8) {
            const speechChunk = words.slice(0, 8).join(' ');
            log(`Adding chunk to queue: "${speechChunk}"`);
            audioQueueRef.current.push(speechChunk);
            wordBuffer = words.slice(8).join(' ');
            processAudioQueue();
          }
        },
        (fullText) => {
          // On completion
          log(`Streaming completed. Full text: "${fullText.substring(0, 100)}..."`);
          
          // Push remaining words as final chunk
          if (wordBuffer.trim()) {
            log(`Adding final chunk to queue: "${wordBuffer.trim()}"`);
            audioQueueRef.current.push(wordBuffer.trim());
            processAudioQueue();
          }

          // Update conversation history
          const history = engineRef.current?.getConversationHistory() || [];
          setConversationHistory(history.map(msg => ({
            role: msg.role as 'user' | 'assistant',
            text: msg.text
          })));
        },
        abortControllerRef.current.signal
      );

      log('Waiting for audio queue to complete...');
      setStatus('Completing speech...');
      
      // Wait for audio queue to finish
      while (audioQueueRef.current.length > 0 || isPlayingRef.current) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setStatus('Ready');
      log('=== PROCESSING COMPLETE ===');
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      log(`Processing error: ${errorMsg}`, 'error');
      console.error('Full error:', error);
      setStatus(`Error: ${errorMsg}`);
      alert(`Error: ${errorMsg}`);
    } finally {
      log('Cleaning up processing state...');
      setIsProcessing(false);
      setTranscript('');
      finalTranscriptRef.current = '';
    }
  };

  // Clear conversation history
  const clearHistory = () => {
    log('Clearing conversation history');
    engineRef.current?.clearHistory();
    setConversationHistory([]);
    setTranscript('');
    setAiResponse('');
    audioQueueRef.current = [];
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
          <p className="text-purple-100 mt-2">Real-time voice conversation powered by AI</p>
        </div>

        {/* Error Banner */}
        {!hasApiKey && (
          <div className="p-6 bg-red-50 border-b border-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-red-600 shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-semibold text-red-900 mb-1">Configuration Error</p>
                <p className="text-sm text-red-800 mb-2">{initError}</p>
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

        <div className="p-6">
          {/* Status Bar */}
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg mb-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {(isProcessing || isSpeaking) && <Loader2 size={16} className="animate-spin text-purple-600" />}
                <span className="text-sm font-medium text-gray-700">
                  {status}
                  {isSpeaking && ' 🔊'}
                </span>
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