// Mints a single-use Live API token so the browser never receives GEMINI_API_KEY.
// The token locks model, voice, language, instructions and tools to the settings below, and the same
// settings are returned so the browser connects with an identical config.
// Only logged-in users get a token: the caller's login token is checked against the API.
// shortcut: no per-user rate limit, a token is single-use and expires in 30 minutes, add IP limiting if traffic grows.

const { GoogleGenAI, Modality } = require('@google/genai');
const { LIVE_SYSTEM_INSTRUCTION, LIVE_TOOLS } = require('./_voice');

const send = (res, status, body) => {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
};

const API_URL = process.env.REACT_APP_API_URL || 'https://printchakra-api.vercel.app';

const isLoggedIn = async (authorization) => {
  if (!authorization) return false;
  try {
    const response = await fetch(`/auth/me`, { headers: { Authorization: authorization } });
    return response.ok;
  } catch (err) {
    console.error('Could not verify the login:', err);
    return false;
  }
};

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return send(res, 405, { error: 'Method not allowed' });
  }
  if (!(await isLoggedIn(req.headers.authorization))) {
    return send(res, 401, { error: 'Please log in' });
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
