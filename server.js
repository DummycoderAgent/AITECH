// AI Technical Skills Interview Platform
// Stack: Express + OpenAI Whisper (speech to text) + Anthropic Claude (evaluation) + D-ID (avatar video)

const express = require("express");
const multer = require("multer");
const crypto = require("crypto");
const path = require("path");

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "public")));

// ── Config ────────────────────────────────────────────────────────────────
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || "";
const DID_API_KEY = process.env.DID_API_KEY || "";
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || "claude-sonnet-4-6";
const DID_VOICE = process.env.DID_VOICE || "en-IN-NeerjaNeural";
const AVATAR_IMAGE_URL =
  process.env.AVATAR_IMAGE_URL ||
  "https://d-id-public-bucket.s3.amazonaws.com/alice.jpg";
const TOTAL_QUESTIONS = parseInt(process.env.TOTAL_QUESTIONS || "3", 10);

// In memory session store (fine for a POC / demo deployment)
const sessions = new Map();

// ── Helpers ───────────────────────────────────────────────────────────────
function didAuthHeader() {
  // D-ID keys from the account page look like "username:password".
  // If the key contains a colon we base64 encode it, otherwise we assume
  // it is already base64 and pass it through.
  if (!DID_API_KEY) return null;
  const encoded = DID_API_KEY.includes(":")
    ? Buffer.from(DID_API_KEY).toString("base64")
    : DID_API_KEY;
  return `Basic ${encoded}`;
}

async function callClaude(systemPrompt, userPrompt, maxTokens = 1200) {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: CLAUDE_MODEL,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    }),
  });
  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`Claude API error ${resp.status}: ${t.slice(0, 300)}`);
  }
  const data = await resp.json();
  const text = (data.content || [])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  return text;
}

function parseJsonLoose(text) {
  let t = text.trim();
  t = t.replace(/^```(json)?/i, "").replace(/```$/i, "").trim();
  const first = t.indexOf("{");
  const last = t.lastIndexOf("}");
  if (first >= 0 && last > first) t = t.slice(first, last + 1);
  return JSON.parse(t);
}

async function transcribeAudio(buffer, mimetype, filename) {
  const fd = new FormData();
  fd.append("file", new Blob([buffer], { type: mimetype || "audio/webm" }), filename || "answer.webm");
  fd.append("model", "whisper-1");
  fd.append("language", "en");
  const resp = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: fd,
  });
  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`Whisper API error ${resp.status}: ${t.slice(0, 300)}`);
  }
  const data = await resp.json();
  return (data.text || "").trim();
}

// Create a talking avatar video for the given text. Returns { videoUrl } or
// { videoUrl: null, fallback: true } if D-ID is not configured or fails.
async function createAvatarVideo(text) {
  const auth = didAuthHeader();
  if (!auth) return { videoUrl: null, fallback: true, reason: "no_key" };
  try {
    const createResp = await fetch("https://api.d-id.com/talks", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: auth },
      body: JSON.stringify({
        source_url: AVATAR_IMAGE_URL,
        script: {
          type: "text",
          input: text,
          provider: { type: "microsoft", voice_id: DID_VOICE },
        },
        config: { stitch: true },
      }),
    });
    if (!createResp.ok) {
      const t = await createResp.text();
      console.error("D-ID create failed:", createResp.status, t.slice(0, 300));
      return { videoUrl: null, fallback: true, reason: `create_${createResp.status}` };
    }
    const created = await createResp.json();
    const talkId = created.id;

    // Poll until the video is ready (up to 100 seconds)
    for (let i = 0; i < 50; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const pollResp = await fetch(`https://api.d-id.com/talks/${talkId}`, {
        headers: { Authorization: auth },
      });
      if (!pollResp.ok) continue;
      const status = await pollResp.json();
      if (status.status === "done" && status.result_url) {
        return { videoUrl: status.result_url, fallback: false };
      }
      if (status.status === "error" || status.status === "rejected") {
        console.error("D-ID render error:", JSON.stringify(status).slice(0, 300));
        return { videoUrl: null, fallback: true, reason: "render_error" };
      }
    }
    return { videoUrl: null, fallback: true, reason: "timeout" };
  } catch (e) {
    console.error("D-ID exception:", e.message);
    return { videoUrl: null, fallback: true, reason: "exception" };
  }
}

// ── Prompts ───────────────────────────────────────────────────────────────
const INTERVIEWER_SYSTEM = `You are Maya, a senior technical interviewer at an enterprise talent assessment company. You conduct short spoken interviews. Your questions must be:
1. Answerable verbally in 60 to 120 seconds with no code writing required
2. Conceptual and scenario based, testing genuine understanding rather than trivia
3. Spoken in a natural, warm, professional style since your words are converted to speech by an avatar
4. One single question at a time, no multi part questions
Always respond with strict JSON only. No markdown, no preamble.`;

function firstQuestionPrompt(skill, name) {
  return `The candidate ${name || "there"} has chosen to be assessed on: ${skill}.
Generate the FIRST interview question. It should be a moderate difficulty calibration question that lets you judge their level.
Begin the question text with a one sentence friendly greeting that mentions the skill, then ask the question.
Respond with JSON: {"question": "<full spoken text including greeting>"}`;
}

function evaluateAndNextPrompt(session, transcript, isLast) {
  const history = session.qa
    .map(
      (q, i) =>
        `Question ${i + 1}: ${q.question}\nAnswer ${i + 1}: ${q.transcript}\nScores: accuracy ${q.eval?.accuracy}, depth ${q.eval?.depth}, clarity ${q.eval?.clarity}`
    )
    .join("\n\n");
  return `Skill being assessed: ${session.skill}
Candidate name: ${session.name || "unknown"}

Interview so far:
${history || "(this was the first question)"}

Current question: ${session.currentQuestion}
Candidate's spoken answer (transcribed, so ignore filler words and minor transcription noise): """${transcript}"""

Tasks:
1. Evaluate this answer on three dimensions, each 0 to 10: accuracy (technical correctness), depth (insight beyond surface level), clarity (structure of the spoken explanation).
2. Write one short sentence of neutral acknowledgement to speak back to the candidate (do not reveal the score, do not say correct or incorrect, just a natural transition like an interviewer would).
${
  isLast
    ? "3. This was the final question. Set next_question to null."
    : `3. Generate the NEXT question. Adapt it: if the answer was strong, probe deeper or move to an advanced scenario in ${session.skill}; if weak, shift to a different fundamental area to give them a fair chance. It must be answerable verbally with no code writing. Start the next question text with the short acknowledgement sentence, then ask the question.`
}

Respond with JSON only:
{"accuracy": n, "depth": n, "clarity": n, "feedback": "<one sentence internal note on the answer quality for the report>", "acknowledgement": "<short spoken transition>", "next_question": ${isLast ? "null" : '"<full spoken text>"'}}`;
}

function reportPrompt(session) {
  const history = session.qa
    .map(
      (q, i) =>
        `Question ${i + 1}: ${q.question}\nTranscript ${i + 1}: ${q.transcript}\nScores: accuracy ${q.eval.accuracy}/10, depth ${q.eval.depth}/10, clarity ${q.eval.clarity}/10\nEvaluator note: ${q.eval.feedback}`
    )
    .join("\n\n");
  return `Skill assessed: ${session.skill}
Candidate: ${session.name || "Candidate"}

Full interview record:
${history}

Produce a final assessment report. Weight accuracy 50 percent, depth 30 percent, clarity 20 percent when computing the overall score out of 100. Be honest and evidence based, citing what the candidate actually said.

Respond with JSON only:
{
  "overall_score": n,
  "band": "<one of: Expert, Proficient, Developing, Foundational>",
  "summary": "<3 to 4 sentence executive summary of the candidate's demonstrated capability>",
  "strengths": ["<specific strength grounded in their answers>", "..."],
  "improvements": ["<specific development area>", "..."],
  "recommendation": "<one sentence hiring style recommendation, e.g. suitability and suggested next step>"
}`;
}

// ── Routes ────────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    anthropic: !!ANTHROPIC_API_KEY,
    openai: !!OPENAI_API_KEY,
    did: !!DID_API_KEY,
    model: CLAUDE_MODEL,
  });
});

app.post("/api/start", async (req, res) => {
  try {
    const { name, skill } = req.body || {};
    if (!skill) return res.status(400).json({ error: "skill is required" });
    if (!ANTHROPIC_API_KEY || !OPENAI_API_KEY)
      return res.status(500).json({ error: "Server is missing ANTHROPIC_API_KEY or OPENAI_API_KEY" });

    const raw = await callClaude(INTERVIEWER_SYSTEM, firstQuestionPrompt(skill, name), 600);
    const q = parseJsonLoose(raw);

    const id = crypto.randomUUID();
    const session = {
      id,
      name: (name || "").slice(0, 60),
      skill,
      startedAt: Date.now(),
      qa: [],
      currentQuestion: q.question,
      qIndex: 0,
      done: false,
    };
    sessions.set(id, session);

    const avatar = await createAvatarVideo(q.question);
    res.json({
      sessionId: id,
      questionNumber: 1,
      totalQuestions: TOTAL_QUESTIONS,
      questionText: q.question,
      videoUrl: avatar.videoUrl,
      fallback: avatar.fallback || false,
    });
  } catch (e) {
    console.error("start error:", e.message);
    res.status(500).json({ error: "Could not start the interview. " + e.message });
  }
});

app.post("/api/answer/:sessionId", upload.single("audio"), async (req, res) => {
  try {
    const session = sessions.get(req.params.sessionId);
    if (!session) return res.status(404).json({ error: "Session not found. Please restart the assessment." });
    if (session.done) return res.status(400).json({ error: "This interview is already complete." });
    if (!req.file) return res.status(400).json({ error: "No audio received." });

    const transcript = await transcribeAudio(req.file.buffer, req.file.mimetype, req.file.originalname);
    if (!transcript || transcript.replace(/[^a-zA-Z]/g, "").length < 8) {
      return res.json({
        retry: true,
        message: "We could not hear a clear answer. Please check your microphone and record again.",
      });
    }

    const isLast = session.qIndex >= TOTAL_QUESTIONS - 1;
    const raw = await callClaude(INTERVIEWER_SYSTEM, evaluateAndNextPrompt(session, transcript, isLast), 900);
    const evalData = parseJsonLoose(raw);

    session.qa.push({
      question: session.currentQuestion,
      transcript,
      eval: {
        accuracy: Number(evalData.accuracy) || 0,
        depth: Number(evalData.depth) || 0,
        clarity: Number(evalData.clarity) || 0,
        feedback: evalData.feedback || "",
      },
    });

    if (isLast) {
      session.done = true;
      return res.json({ transcript, done: true });
    }

    session.qIndex += 1;
    session.currentQuestion = evalData.next_question;
    const avatar = await createAvatarVideo(evalData.next_question);
    res.json({
      transcript,
      done: false,
      next: {
        questionNumber: session.qIndex + 1,
        totalQuestions: TOTAL_QUESTIONS,
        questionText: evalData.next_question,
        videoUrl: avatar.videoUrl,
        fallback: avatar.fallback || false,
      },
    });
  } catch (e) {
    console.error("answer error:", e.message);
    res.status(500).json({ error: "Could not process the answer. " + e.message });
  }
});

app.get("/api/report/:sessionId", async (req, res) => {
  try {
    const session = sessions.get(req.params.sessionId);
    if (!session) return res.status(404).json({ error: "Session not found." });
    if (!session.qa.length) return res.status(400).json({ error: "No answers recorded yet." });

    const raw = await callClaude(
      "You are a rigorous, fair technical assessment analyst. Respond with strict JSON only.",
      reportPrompt(session),
      1200
    );
    const report = parseJsonLoose(raw);

    res.json({
      candidate: session.name || "Candidate",
      skill: session.skill,
      date: new Date(session.startedAt).toDateString(),
      report,
      questions: session.qa.map((q, i) => ({
        number: i + 1,
        question: q.question,
        transcript: q.transcript,
        accuracy: q.eval.accuracy,
        depth: q.eval.depth,
        clarity: q.eval.clarity,
        note: q.eval.feedback,
      })),
    });
  } catch (e) {
    console.error("report error:", e.message);
    res.status(500).json({ error: "Could not generate the report. " + e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Skill Interview AI running on port ${PORT}`));
