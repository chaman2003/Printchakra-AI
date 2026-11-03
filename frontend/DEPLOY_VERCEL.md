# Deploying PrintChakra on Vercel

Everything runs on Vercel as two projects plus one database:

| Part | Vercel project | Folder | URL |
|---|---|---|---|
| React app + voice token function | `printchakra` | `frontend/` | https://printchakra.vercel.app |
| Flask API (upload, pipeline, OCR, files) | `printchakra-api` | `backend/` | https://printchakra-api.vercel.app |
| Shared storage + realtime events | Neon Postgres (`printchakra-db`) | | connected to `printchakra-api` |

## How the cloud version differs from the local one

| Local (Windows) | Vercel |
|---|---|
| Files on disk | Files in Postgres. Each function instance keeps a copy in `/tmp` and syncs before/after each request (`backend/app/core/cloud.py`). |
| Socket.IO | HTTP polling of `GET /events/poll` (`backend/app/core/events.py`, `frontend/src/context/pollingSocket.ts`). |
| PaddleOCR / Tesseract | Gemini vision (`backend/app/modules/ocr/gemini_ocr.py`). |
| Whisper + Ollama + TTS | Gemini Live in the browser (`frontend/src/components/voice/GeminiLiveVoice.tsx`). The assistant can also operate the app. |
| Windows printer | Browser print dialog. `/print/document` returns a file URL and the dashboard prints it. |
| Background thread after upload | The 12-step pipeline runs inside the upload request. |

Local development is unchanged: `python app.py` with `requirements-local.txt`, and `npm start`.

## Environment variables

`printchakra-api` (set with `vercel env add`, inside `backend/`):

| Name | Purpose |
|---|---|
| `DATABASE_URL` | Added by the Neon integration. |
| `GEMINI_API_KEY` | OCR. Secret. |
| `OCR_MODEL` | Default `gemini-3.5-flash-lite`. |
| `SECRET_KEY` | Flask secret. |

`printchakra` (inside `frontend/`, or from the repo root, see below):

| Name | Purpose |
|---|---|
| `REACT_APP_API_URL` | `https://printchakra-api.vercel.app` |
| `REACT_APP_REALTIME` | `poll` |
| `GEMINI_API_KEY`, `MODEL`, `VOICE`, `LANGUAGE` | Voice token function. |

## Deploy with the Vercel CLI

```
# API
cd backend
npx vercel deploy --prod

# Frontend: deploy from the REPO ROOT. The project's Root Directory is "frontend".
cd ..
npx vercel deploy --prod
```

The root `.vercelignore` keeps the upload small. The `backend` project is not connected to Git
(a push would try to build the repo root). The `printchakra` project keeps its GitHub link.

## Vercel MCP

`npx -y add-mcp https://mcp.vercel.com -a kiro-cli -g -y -n vercel` adds the Vercel MCP server to
`~/.kiro/settings/mcp.json`. Log in to it from a new Kiro session when prompted.

## Limits to know about (Hobby plan)

- Request bodies are limited to 4.5 MB. The Phone page shrinks large photos before upload.
- Function time limit is 120 s here. A scan takes about 25 s (pipeline plus OCR).
- Events use one function call per poll (2.5 s while active, 6 s idle, 15 s when the tab is hidden).
  The Hobby plan includes 1 million calls a month, so a tab left open all day can use a large share.
- Neon free tier holds 0.5 GB. Files are stored in the database as bytes.
- Anyone who opens the site can start voice sessions and run OCR on your Gemini key. Add rate limiting
  or sign-in before sharing the URL widely.
