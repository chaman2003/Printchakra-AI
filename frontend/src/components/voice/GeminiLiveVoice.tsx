import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Alert, AlertIcon, Badge, Box, Button, Flex, Text, VStack, useColorModeValue } from '@chakra-ui/react';
import type { LiveServerMessage, Session } from '@google/genai';
import { emitVoiceCommand } from '../../aiassist/commandBus';
import { authHeader } from '../../auth/session';

// Voice assistant on Gemini Live. The browser connects straight to Gemini with a
// short-lived token from /api/gemini-token, so the API key stays on the server.
// The assistant can operate the app through two tools (navigate, app_command), see api/_voice.ts.

const MIC_RATE = 16000; // Gemini expects 16 kHz mono PCM input.
const PLAY_RATE = 24000; // Gemini returns 24 kHz mono PCM output.

const PAGE_PATHS: Record<string, string> = { dashboard: '/dashboard', playground: '/playground', phone: '/phone', home: '/' };
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

type Line = { role: 'user' | 'assistant'; text: string; open: boolean };
type Status = 'idle' | 'connecting' | 'listening' | 'speaking' | 'error';

const STATUS_LABEL: Record<Status, string> = {
  idle: 'Off',
  connecting: 'Connecting',
  listening: 'Listening',
  speaking: 'Speaking',
  error: 'Error',
};

const STATUS_COLOR: Record<Status, string> = {
  idle: 'gray',
  connecting: 'yellow',
  listening: 'green',
  speaking: 'purple',
  error: 'red',
};

// Captures raw mic frames on the audio thread and posts them to the main thread.
const MIC_WORKLET = `
class MicCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor('mic-capture', MicCapture);
`;

// Averages each block of input samples into one output sample, then converts to Int16.
const toPcm16 = (input: Float32Array, inRate: number): Int16Array => {
  const ratio = inRate / MIC_RATE;
  const outLength = Math.floor(input.length / ratio);
  const out = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end && j < input.length; j++) sum += input[j];
    const sample = Math.max(-1, Math.min(1, sum / (end - start)));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return out;
};

const bytesToBase64 = (bytes: Uint8Array): string => {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
};

const base64ToPcm16 = (data: string): Float32Array => {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 0x8000;
  return out;
};

const GeminiLiveVoice: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const navigateRef = useRef(navigate);
  const pathRef = useRef(location.pathname);
  navigateRef.current = navigate;
  pathRef.current = location.pathname;

  const [status, setStatus] = useState<Status>('idle');
  const [lines, setLines] = useState<Line[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sessionRef = useRef<Session | null>(null);
  const micCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micNodeRef = useRef<AudioWorkletNode | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const playHeadRef = useRef(0);
  const playingRef = useRef(new Set<AudioBufferSourceNode>());
  const stoppingRef = useRef(false);

  const panelBg = useColorModeValue('whiteAlpha.900', 'blackAlpha.400');
  const lineBg = useColorModeValue('gray.50', 'whiteAlpha.100');

  const appendText = useCallback((role: Line['role'], text: string) => {
    setLines((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.open && last.role === role) {
        return [...prev.slice(0, -1), { ...last, text: last.text + text }];
      }
      return [...prev, { role, text, open: true }];
    });
  }, []);

  const closeOpenLines = useCallback(() => {
    setLines((prev) => prev.map((line) => (line.open ? { ...line, open: false } : line)));
  }, []);

  const stopPlayback = useCallback(() => {
    playingRef.current.forEach((node) => {
      try {
        node.stop();
      } catch {
        // Already stopped.
      }
    });
    playingRef.current.clear();
    if (playCtxRef.current) playHeadRef.current = playCtxRef.current.currentTime;
  }, []);

  const playChunk = useCallback((data: string) => {
    const ctx = playCtxRef.current;
    if (!ctx) return;
    const samples = base64ToPcm16(data);
    const buffer = ctx.createBuffer(1, samples.length, PLAY_RATE);
    buffer.copyToChannel(samples, 0);
    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(ctx.destination);
    // Queue chunks back to back so audio does not overlap or gap.
    const startAt = Math.max(ctx.currentTime, playHeadRef.current);
    node.start(startAt);
    playHeadRef.current = startAt + buffer.duration;
    playingRef.current.add(node);
    node.onended = () => {
      playingRef.current.delete(node);
      if (playingRef.current.size === 0 && sessionRef.current) setStatus('listening');
    };
  }, []);

  const stopSession = useCallback(() => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;

    micNodeRef.current?.port.close();
    micNodeRef.current?.disconnect();
    micNodeRef.current = null;
    micStreamRef.current?.getTracks().forEach((track) => track.stop());
    micStreamRef.current = null;
    micCtxRef.current?.close().catch(() => undefined);
    micCtxRef.current = null;

    stopPlayback();
    playCtxRef.current?.close().catch(() => undefined);
    playCtxRef.current = null;

    try {
      sessionRef.current?.close();
    } catch {
      // Socket already closed.
    }
    sessionRef.current = null;
    closeOpenLines();
    setStatus('idle');
    stoppingRef.current = false;
  }, [closeOpenLines, stopPlayback]);

  const startSession = useCallback(async () => {
    setError(null);
    setLines([]);
    setStatus('connecting');

    // Create audio contexts inside the click handler so browsers allow playback.
    const playCtx = new AudioContext({ sampleRate: PLAY_RATE });
    playCtxRef.current = playCtx;
    playHeadRef.current = 0;

    // Runs a tool the assistant asked for and reports the outcome back to it.
    const runToolCalls = async (calls: Array<{ id?: string; name?: string; args?: Record<string, unknown> }>) => {
      const responses = [];
      for (const call of calls) {
        let output: Record<string, unknown>;
        try {
          if (call.name === 'navigate') {
            const path = PAGE_PATHS[String(call.args?.page)];
            if (!path) throw new Error('Unknown page');
            if (pathRef.current !== path) navigateRef.current(path);
            output = { ok: true, page: call.args?.page };
          } else if (call.name === 'app_command') {
            // Dashboard commands are handled by the Dashboard page, so make sure it is open first.
            if (pathRef.current !== '/dashboard') {
              navigateRef.current('/dashboard');
              await sleep(1200);
            }
            emitVoiceCommand({
              type: 'dashboard_command',
              payload: { command: call.args?.command, params: call.args?.params ?? {}, raw: call.args },
            });
            output = { ok: true, note: 'Command sent to the dashboard' };
          } else {
            throw new Error(`Unknown tool ${call.name}`);
          }
        } catch (err: any) {
          output = { ok: false, error: err?.message || 'Tool failed' };
        }
        responses.push({ id: call.id, name: call.name, response: output });
      }
      try {
        sessionRef.current?.sendToolResponse({ functionResponses: responses });
      } catch {
        // Session closed while the tool ran.
      }
    };

    try {
      const tokenRes = await fetch('/api/gemini-token', { method: 'POST', headers: authHeader() });
      const tokenBody = await tokenRes.json().catch(() => ({}));
      if (!tokenRes.ok || !tokenBody.token) {
        throw new Error(tokenBody.error || 'Could not start a voice session');
      }
      const { token, model, voice, language, systemInstruction, tools } = tokenBody as {
        token: string;
        model: string;
        voice: string;
        language: string;
        systemInstruction: string;
        tools: any[];
      };

      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      micStreamRef.current = micStream;

      const micCtx = new AudioContext();
      micCtxRef.current = micCtx;
      const workletUrl = URL.createObjectURL(new Blob([MIC_WORKLET], { type: 'application/javascript' }));
      await micCtx.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      // Load the SDK on first use so the initial page bundle stays small.
      const { GoogleGenAI, Modality } = await import('@google/genai');

      // The token locks model, voice and language. The client must send the same speech settings.
      const ai = new GoogleGenAI({ apiKey: token, httpOptions: { apiVersion: 'v1alpha' } });
      const session = await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } },
            languageCode: language,
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction,
          tools,
        },
        callbacks: {
          onmessage: (message: LiveServerMessage) => {
            if (message.toolCall?.functionCalls?.length) {
              void runToolCalls(message.toolCall.functionCalls);
            }
            const content = message.serverContent;
            if (!content) return;
            if (content.interrupted) {
              stopPlayback();
            }
            for (const part of content.modelTurn?.parts ?? []) {
              if (part.inlineData?.data) {
                setStatus('speaking');
                playChunk(part.inlineData.data);
              }
            }
            if (content.inputTranscription?.text) appendText('user', content.inputTranscription.text);
            if (content.outputTranscription?.text) appendText('assistant', content.outputTranscription.text);
            if (content.turnComplete) {
              closeOpenLines();
              if (playingRef.current.size === 0) setStatus('listening');
            }
          },
          onerror: (e: ErrorEvent) => {
            setError(e?.message || 'Voice connection error');
            stopSession();
            setStatus('error');
          },
          onclose: (e: CloseEvent) => {
            if (sessionRef.current && !stoppingRef.current) {
              setError(e?.reason || 'Voice session ended');
              stopSession();
              setStatus('error');
            }
          },
        },
      });
      sessionRef.current = session;

      // Mic -> worklet -> 16 kHz PCM -> Gemini. The muted gain keeps the node pulled without echoing the mic.
      const source = micCtx.createMediaStreamSource(micStream);
      const node = new AudioWorkletNode(micCtx, 'mic-capture');
      micNodeRef.current = node;
      const mute = micCtx.createGain();
      mute.gain.value = 0;
      source.connect(node);
      node.connect(mute);
      mute.connect(micCtx.destination);

      // Batch ~100 ms of audio per message. Worklet blocks are 128 samples, which is too chatty for a socket.
      let pending: Float32Array[] = [];
      let pendingLength = 0;
      const batchSamples = Math.round(micCtx.sampleRate * 0.1);
      node.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const current = sessionRef.current;
        if (!current) return;
        pending.push(event.data);
        pendingLength += event.data.length;
        if (pendingLength < batchSamples) return;

        const merged = new Float32Array(pendingLength);
        let offset = 0;
        for (const chunk of pending) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }
        pending = [];
        pendingLength = 0;

        const pcm = toPcm16(merged, micCtx.sampleRate);
        current.sendRealtimeInput({
          audio: {
            data: bytesToBase64(new Uint8Array(pcm.buffer)),
            mimeType: `audio/pcm;rate=${MIC_RATE}`,
          },
        });
      };

      setStatus('listening');
    } catch (err: any) {
      const message =
        err?.name === 'NotAllowedError'
          ? 'Microphone permission was denied.'
          : err?.message || 'Could not start voice';
      setError(message);
      stopSession();
      setStatus('error');
    }
  }, [appendText, closeOpenLines, playChunk, stopPlayback, stopSession]);

  // Release mic, audio and socket when the panel closes.
  useEffect(() => () => stopSession(), [stopSession]);

  const active = status !== 'idle' && status !== 'error';

  return (
    <Box p={4} bg={panelBg} borderRadius="xl" h="100%" display="flex" flexDirection="column" gap={3}>
      <Flex align="center" justify="space-between">
        <Text fontWeight="700" fontSize="sm">Voice Assistant</Text>
        <Badge colorScheme={STATUS_COLOR[status]} variant="subtle">
          {STATUS_LABEL[status]}
        </Badge>
      </Flex>

      {error && (
        <Alert status="error" borderRadius="md" fontSize="sm">
          <AlertIcon />
          {error}
        </Alert>
      )}

      <Button
        colorScheme={active ? 'red' : 'blue'}
        onClick={active ? stopSession : startSession}
        isLoading={status === 'connecting'}
        loadingText="Connecting"
      >
        {active ? 'Stop conversation' : 'Start conversation'}
      </Button>

      <VStack align="stretch" spacing={2} flex="1" overflowY="auto">
        {lines.length === 0 && (
          <Text fontSize="xs" opacity={0.7}>
            Press start and speak. Replies are spoken and transcribed here.
          </Text>
        )}
        {lines.map((line, index) => (
          <Box key={index} bg={lineBg} p={2} borderRadius="md" fontSize="sm">
            <Text fontSize="xs" fontWeight="600" opacity={0.7}>
              {line.role === 'user' ? 'You' : 'Assistant'}
            </Text>
            <Text whiteSpace="pre-wrap">{line.text}</Text>
          </Box>
        ))}
      </VStack>
    </Box>
  );
};

export default GeminiLiveVoice;
