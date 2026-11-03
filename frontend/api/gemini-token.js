// Mints a single-use Live API token so the browser never receives GEMINI_API_KEY.
// The token locks model, voice, language, instructions and tools to the settings below, and the same
// settings are returned so the browser connects with an identical config.
// shortcut: no per-visitor rate limit, a token is single-use and expires in 30 minutes, add IP limiting if traffic grows.

const { GoogleGenAI, Modality } = require('@google/genai');
const { LIVE_SYSTEM_INSTRUCTION, LIVE_TOOLS } = require('./_voice');

const send = (res, status, body) => {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
};

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return send(res, 405, { error: 'Method not allowed' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  const model = process.env.MODEL || 'gemini-3.8-live';
  const voice = process.env.VOICE || 'Kore';
  const language = process.env.LANGUAGE || 'en-IN';

  if (!apiKey) {
    console.error('GEMINI_API_KEY is not set');
    return send(res, 500, { error: 'Voice is not configured on the server' });
  }

  const liveConfig = {
    responseModalities: [Modality.AUDIO],
    speechConfig: {
      voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } },
      languageCode: language,
    },
    systemInstruction: LIVE_SYSTEM_INSTRUCTION,
    tools: LIVE_TOOLS,
  };

  try {
    const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } });
    const token = await ai.authTokens.create({
      config: {
        uses: 1,
        expireTime: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
        newSessionExpireTime: new Date(Date.now() + 2 * 60 * 1000).toISOString(),
        liveConnectConstraints: { model, config: liveConfig },
        httpOptions: { apiVersion: 'v1alpha' },
      },
    });

    if (!token.name) throw new Error('Token response had no name');
    return send(res, 200, {
      token: token.name,
      model,
      voice,
      language,
      systemInstruction: LIVE_SYSTEM_INSTRUCTION,
      tools: LIVE_TOOLS,
    });
  } catch (err) {
    console.error('Gemini token creation failed:', err);
    return send(res, 502, { error: 'Could not start a voice session' });
  }
};
