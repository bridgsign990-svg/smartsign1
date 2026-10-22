# BridgeSign Deployment

## Secret Key Configuration

Generate a strong key with Python and configure it as the `SECRET_KEY`
environment variable. Do not commit the generated value or a `.env` file.
The repository's `.env.example` contains placeholders only, and BridgeSign
does not automatically load `.env` files.

For local development in PowerShell, set a per-session key before starting the
app:

```powershell
$env:SECRET_KEY = (& .\.venv-2\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))").Trim()
.\.venv-2\Scripts\python.exe app.py
```

Without `SECRET_KEY`, local startup uses a random ephemeral key. Browser
sessions and call invitations signed with it will not survive an app restart.
In production, set a persistent `SECRET_KEY` in the hosting provider's secret
environment settings. BridgeSign fails during startup if a Render or Railway
deployment is missing this variable. Keeping the existing production value
unchanged preserves current sessions and invitations.

## Phone And Call Testing Requirements

Camera and microphone access require HTTPS on phones. A plain LAN URL such as `http://192.168.x.x:5000` will usually block media access.

The call module also uses WebSockets for signaling, so deploy as a dynamic web service, not a static site.

## Railway Deployment

1. Push this project to a GitHub repository.
2. Go to [railway.com](https://railway.app) and create a new project from that repository.
3. Railway will auto-detect the `Dockerfile` and `railway.json`.
4. In your Railway service settings, add these environment variables:
   - `SECRET_KEY` — set to a random string
   - `BRIDGESIGN_HOST` = `0.0.0.0`
   - `BRIDGESIGN_WARM_INFERENCE` = `0`
   - `DATABASE_URL` = the connection URL for a PostgreSQL database (use Railway's private/reference URL)
   - `RAILWAY_ENVIRONMENT` = `production`
5. Railway auto-assigns `PORT` — do not set it manually.
6. Deploy. Railway will build the Docker image and give you a public HTTPS URL.
7. Open the deployed URL on your phone or browser.

## Render Web Service

1. Push this project to a GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Use the Docker runtime. The included `Dockerfile` installs the Linux shared libraries required by MediaPipe, including `libGLESv2.so.2`.
4. If Render asks for a start command, leave it blank so it uses the Dockerfile `CMD`, or use:
   `gunicorn app:app --worker-class gthread --workers 1 --threads 8 --bind 0.0.0.0:$PORT --timeout 120`
5. Set the health check path to `/health` if Render does not read it from `render.yaml`.
6. Add a managed PostgreSQL database and set `DATABASE_URL` to its internal connection URL. Confirm these environment variables if Render does not read them from `render.yaml`:
   `PORT=10000`, `HOSTNAME=0.0.0.0`, `BRIDGESIGN_HOST=0.0.0.0`, `BRIDGESIGN_WARM_INFERENCE=0`, `RENDER=true`, `GUNICORN_TIMEOUT=180`.
7. If you see **502 Bad Gateway**, check Render Logs for `WORKER TIMEOUT`, `SIGKILL`, or `Out of memory`. The sign model must stay under ~80MB on disk; use at least the **Starter** plan (512MB RAM) or disable warm inference (`BRIDGESIGN_WARM_INFERENCE=0`).
7. Set `SECRET_KEY` to a random value if Render does not generate it from `render.yaml`.
8. Open the deployed `https://...onrender.com` URL on your phone.

If an existing Render service was created with the Python native runtime and Render will not let you change its runtime, create a new Web Service with Docker selected. MediaPipe needs OS-level packages that are not present in Render's native Python image.

The Render service sets `BRIDGESIGN_WARM_INFERENCE=0` so the web server can pass health checks immediately. MediaPipe and the sign models load lazily the first time a camera inference endpoint receives a frame.

Accounts, custom emergency phrases, and learning data use the SQL database. Set `DATABASE_URL` to a managed PostgreSQL instance for production; without it, the app creates `data/bridgesign.sqlite3` for local development. The SQLite fallback is persistent only when its directory is on persistent storage. Existing `users.json` and `phrases_*.json` data are imported on startup and left untouched as a backup.

Temporary uploads and other file-based runtime data still use `BRIDGESIGN_DATA_DIR`; use a persistent disk or volume if those files need to survive redeploys.

## Learning Module

The learning course is seeded idempotently at app startup. Authenticated APIs are available under `/api/learning`, including courses, course detail, module detail, lesson content/progress, exercises, quiz start/answer/resume/complete, camera practice, mastery, and recommendations. Quiz answers and attempts are stored server-side and scoped to the logged-in account.

The starter course uses the local CWASA SiGML assets for word demonstrations. Those assets are drafts and should be reviewed by fluent ASL signers before being treated as authoritative instruction. The current recognition model supports static letters A-Z only; camera evaluation is enabled only for the seeded A-C lesson. Word-sign lessons use avatar demonstrations, written explanations, and context quizzes without claiming camera recognition.

## Why One Worker

The call room signaling state is currently stored in memory in `call_room.py`. Running multiple workers can split two callers across different processes, so they may not find each other. Use one threaded worker until signaling is moved to Redis or another shared store.

## Call Module Note

The current WebRTC config uses public STUN only. It should work for many phone and laptop tests, but some mobile networks or strict NATs may need a TURN server later.

