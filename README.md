# SkillVoice: AI Avatar Technical Interview Platform

A live spoken assessment platform. The participant selects a technical skill, a photorealistic avatar interviewer asks three adaptive questions, answers are recorded on the microphone, transcribed by OpenAI Whisper, evaluated by Claude, and a final scored report is generated.

## How it works

1. Participant enters their name and selects a skill (Python, SQL, JavaScript, Prompt Engineering, GenAI and LLM Engineering, or Machine Learning)
2. Claude generates the first question, D-ID renders it as a talking avatar video
3. Participant records a spoken answer (up to three minutes)
4. Whisper transcribes the answer, Claude scores it on accuracy, depth and clarity, and generates the next question adapted to the answer
5. After three questions, Claude produces a full report with an overall score out of 100, a band, strengths, development areas and a recommendation, downloadable as PDF

If the D-ID key is missing or credits run out, the platform automatically falls back to an illustrated interviewer with a natural browser voice, so a demo never breaks.

## Environment variables

Set these in Railway under the service Variables tab:

* `ANTHROPIC_API_KEY` (required) your existing Anthropic key
* `OPENAI_API_KEY` (required) your existing OpenAI key, used for Whisper transcription
* `DID_API_KEY` (recommended) from D-ID, see below
* `AVATAR_IMAGE_URL` (optional) URL of a portrait photo to use as the interviewer face. Defaults to a D-ID stock presenter
* `DID_VOICE` (optional) Microsoft voice ID. Defaults to `en-IN-NeerjaNeural`, an Indian English female voice
* `CLAUDE_MODEL` (optional) defaults to `claude-sonnet-4-6`
* `TOTAL_QUESTIONS` (optional) defaults to 3

## Getting the D-ID key

1. Go to https://studio.d-id.com and sign up (14 day trial, 20 credits, videos carry a watermark on trial)
2. Open your Account page and generate an API key
3. Copy it exactly as shown and paste it into the `DID_API_KEY` variable in Railway. The server handles both plain and base64 key formats automatically

Each question rendered as avatar video consumes credits. A three question interview uses roughly one to two minutes of video. For regular use, the Pro plan (around 48 USD per month) includes API access.

## Deploying on GitHub and Railway

Same flow as your earlier two projects:

1. Create a new private repository on GitHub, for example `skillvoice-interview`
2. Upload the files. For the nested file, click Add file, then Create new file, and type `public/index.html` in the filename field to create the folder, then paste the content. Upload `server.js`, `package.json`, `railway.json` and `.gitignore` at the root. Do not upload `node_modules`
3. In Railway, create a New Project, choose Deploy from GitHub repo, and select this repository. Railway detects Node automatically from `package.json` and uses `npm start`
4. Open the service, go to the Variables tab, and add the environment variables listed above
5. Under Settings, Networking, click Generate Domain to get your public URL
6. Open the URL, allow microphone access when prompted, and run a test interview

## Notes

* Sessions are held in memory, which is fine for demos. A redeploy or restart clears active sessions
* The report page prints cleanly, so Download report as PDF uses the browser print dialog
* Question generation takes 10 to 30 seconds when the avatar video is rendering; the loading state covers this
* Microphone capture requires HTTPS, which Railway domains provide by default
