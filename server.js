// SkillVoice: AI Agentic Interviewer across job families
// Tracks: Operations, Sales, Customer Service, Technical
// Stack: Express + OpenAI Whisper (speech to text) + Anthropic Claude (interviewer brain) + D-ID (avatar video)

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
const AVATAR_IMAGE_URL =
  process.env.AVATAR_IMAGE_URL ||
  "https://d-id-public-bucket.s3.amazonaws.com/alice.jpg";
const TOTAL_QUESTIONS = parseInt(process.env.TOTAL_QUESTIONS || "3", 10);

const sessions = new Map();
let lastDidError = null;

// ── Role library ──────────────────────────────────────────────────────────
// Each track defines: the roles inside it, the three scoring dimensions
// (always weighted 50/30/20 in the final report), and interviewing guidance
// that shapes how Claude behaves for that job family.
const TRACKS = {
  Technical: {
    dims: ["Accuracy", "Depth", "Clarity"],
    roles: [
      "Python",
      "SQL and Data Modelling",
      "JavaScript and Modern Web",
      "Prompt Engineering",
      "Generative AI and LLM Engineering",
      "Machine Learning Fundamentals",
    ],
    guidance: `This is a technical proficiency interview on the selected skill. Questions must be conceptual and scenario based, testing genuine understanding rather than trivia. No code writing: everything must be answerable verbally. Probe how and why, not definitions.`,
  },
  Sales: {
    dims: ["Persuasion", "Commercial Acumen", "Communication"],
    roles: [
      "Frontline Sales Executive",
      "Key Account Manager",
      "Branch Manager",
      "National Sales Head",
    ],
    guidance: `This is a sales capability interview. Use realistic selling scenarios and calibrate to the seniority of the role: frontline roles get field scenarios (prospecting, discovery, objection handling, closing under pressure); managerial roles get scenarios on coaching a team, pipeline discipline and forecasting; national leadership roles get scenarios on channel strategy, pricing decisions, P and L thinking and leading through a down quarter. Probe for what the candidate would actually say and do, not textbook theory.`,
  },
  "Customer Service": {
    dims: ["Empathy", "Problem Resolution", "Communication"],
    roles: [
      "Voice Support Executive",
      "Chat Support Executive",
      "Email and Back Office Support",
      "In Person Customer Service",
    ],
    guidance: `This is a customer service capability interview. Use realistic customer scenarios tailored to the channel of the role: voice roles test live de escalation and tone under pressure; chat roles test handling multiple conversations, clarity in writing and judgement on canned versus personal responses (asked verbally); email and back office roles test accuracy, prioritisation and process adherence; in person roles test presence, patience and service recovery. Include at least one scenario with an upset or difficult customer.`,
  },
  Operations: {
    dims: ["Process Rigour", "Analytical Thinking", "Communication"],
    roles: [
      "Operations Analyst",
      "Financial Operations Associate",
      "Process Excellence and Quality",
      "MIS and Reporting Analyst",
    ],
    guidance: `This is an operations capability interview. Use realistic operational scenarios tailored to the role: analyst roles test structured problem solving, root cause thinking and working with incomplete data; financial operations roles test reconciliation logic, controls, accuracy under deadline pressure and escalation judgement; process and quality roles test process design, SLA management and continuous improvement thinking; MIS roles test data sense, reporting judgement and stakeholder communication. Probe for how the candidate structures a problem, not just the answer.`,
  },
};

// Languages available for spoken interviews (used by the Voice Support role,
// and safe to extend to other roles later). Maps to Whisper language codes,
// D-ID Microsoft voices, and browser speech synthesis locales.
const LANGUAGES = {
  English: { whisper: "en", voice: "en-IN-NeerjaNeural", bcp: "en-IN" },
  Hindi: { whisper: "hi", voice: "hi-IN-SwaraNeural", bcp: "hi-IN" },
  Tamil: { whisper: "ta", voice: "ta-IN-PallaviNeural", bcp: "ta-IN" },
  Telugu: { whisper: "te", voice: "te-IN-ShrutiNeural", bcp: "te-IN" },
  Kannada: { whisper: "kn", voice: "kn-IN-SapnaNeural", bcp: "kn-IN" },
  Bengali: { whisper: "bn", voice: "bn-IN-TanishaaNeural", bcp: "bn-IN" },
  Marathi: { whisper: "mr", voice: "mr-IN-AarohiNeural", bcp: "mr-IN" },
};

// ── Helpers ───────────────────────────────────────────────────────────────
function didAuthHeader() {
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
  return (data.content || [])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}

function parseJsonLoose(text) {
  let t = text.trim();
  t = t.replace(/^```(json)?/i, "").replace(/```$/i, "").trim();
  const first = t.indexOf("{");
  const last = t.lastIndexOf("}");
  if (first >= 0 && last > first) t = t.slice(first, last + 1);
  return JSON.parse(t);
}

async function transcribeAudio(buffer, mimetype, filename, langCode) {
  const fd = new FormData();
  fd.append("file", new Blob([buffer], { type: mimetype || "audio/webm" }), filename || "answer.webm");
  fd.append("model", "whisper-1");
  if (langCode) fd.append("language", langCode);
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

async function createAvatarVideo(text, voiceId) {
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
          provider: { type: "microsoft", voice_id: voiceId || "en-IN-NeerjaNeural" },
        },
        config: { stitch: true },
      }),
    });
    if (!createResp.ok) {
      const t = await createResp.text();
      lastDidError = `create ${createResp.status}: ${t.slice(0, 300)}`;
      console.error("D-ID create failed:", lastDidError);
      return { videoUrl: null, fallback: true, reason: `create_${createResp.status}` };
    }
    const created = await createResp.json();
    const talkId = created.id;

    // Poll until the video is ready (up to 150 seconds)
    for (let i = 0; i < 75; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const pollResp = await fetch(`https://api.d-id.com/talks/${talkId}`, {
        headers: { Authorization: auth },
      });
      if (!pollResp.ok) continue;
      const status = await pollResp.json();
      if (status.status === "done" && status.result_url) {
        lastDidError = null;
        return { videoUrl: status.result_url, fallback: false };
      }
      if (status.status === "error" || status.status === "rejected") {
        lastDidError = `render: ${JSON.stringify(status).slice(0, 300)}`;
        console.error("D-ID render error:", lastDidError);
        return { videoUrl: null, fallback: true, reason: "render_error" };
      }
    }
    lastDidError = "timeout: video was accepted but did not finish rendering in 150 seconds";
    console.error("D-ID timeout for talk", talkId);
    return { videoUrl: null, fallback: true, reason: "timeout" };
  } catch (e) {
    lastDidError = `exception: ${e.message}`;
    console.error("D-ID exception:", e.message);
    return { videoUrl: null, fallback: true, reason: "exception" };
  }
}

// ── Prompts ───────────────────────────────────────────────────────────────
function interviewerSystem(session) {
  const track = TRACKS[session.profile];
  const langLine =
    session.language && session.language !== "English"
      ? `Conduct the ENTIRE interview in ${session.language}. Every question, greeting and transition must be written in natural, professional ${session.language}.`
      : `Conduct the interview in professional English suitable for an Indian audience.`;
  return `You are Maya, a senior interviewer at an enterprise talent assessment company, conducting a spoken first round interview for the role of ${session.role} (${session.profile} track).

${track.guidance}

${langLine}

Rules for every question you produce:
1. Answerable verbally in 60 to 120 seconds
2. One single question at a time, no multi part questions
3. Spoken in a natural, warm, professional style since your words are converted to speech by an avatar
4. CONCISE: the entire spoken text including any greeting or transition must stay under 45 words, since every word adds avatar rendering time and cost

Always respond with strict JSON only. No markdown, no preamble.`;
}

function firstQuestionPrompt(session) {
  return `The candidate ${session.name || "there"} is interviewing for: ${session.role}.
Generate the FIRST interview question. It should be a moderate difficulty calibration question that lets you judge their level for this role.
Begin the question text with a one sentence friendly greeting that mentions the role, then ask the question.
Respond with JSON: {"question": "<full spoken text including greeting>"}`;
}

function evaluateAndNextPrompt(session, transcript, isLast) {
  const [d1, d2, d3] = session.dims;
  const history = session.qa
    .map(
      (q, i) =>
        `Question ${i + 1}: ${q.question}\nAnswer ${i + 1}: ${q.transcript}\nScores: ${d1} ${q.eval?.scores?.[0]}, ${d2} ${q.eval?.scores?.[1]}, ${d3} ${q.eval?.scores?.[2]}`
    )
    .join("\n\n");
  return `Role: ${session.role} (${session.profile} track)
Candidate name: ${session.name || "unknown"}

Interview so far:
${history || "(this was the first question)"}

Current question: ${session.currentQuestion}
Candidate's spoken answer (transcribed, so ignore filler words and minor transcription noise): """${transcript}"""

Tasks:
1. Evaluate this answer on three dimensions, each 0 to 10:
   d1 = ${d1} (how well the substance of the answer meets the bar for this role)
   d2 = ${d2} (the depth and quality of thinking shown beyond the surface)
   d3 = ${d3} (structure and effectiveness of the spoken delivery)
2. Write one short internal note on the answer quality for the final report.
${
  isLast
    ? "3. This was the final question. Set next_question to null."
    : `3. Generate the NEXT question. Adapt it: if the answer was strong, probe deeper or raise the difficulty for a ${session.role}; if weak, shift to a different fundamental area of the role to give them a fair chance. Start the next question text with a short natural spoken transition (do not reveal any judgement of the previous answer), then ask the question.`
}

Respond with JSON only:
{"d1": n, "d2": n, "d3": n, "feedback": "<one sentence internal note>", "next_question": ${isLast ? "null" : '"<full spoken text>"'}}`;
}

function reportPrompt(session) {
  const [d1, d2, d3] = session.dims;
  const history = session.qa
    .map(
      (q, i) =>
        `Question ${i + 1}: ${q.question}\nTranscript ${i + 1}: ${q.transcript}\nScores: ${d1} ${q.eval.scores[0]}/10, ${d2} ${q.eval.scores[1]}/10, ${d3} ${q.eval.scores[2]}/10\nEvaluator note: ${q.eval.feedback}`
    )
    .join("\n\n");
  return `Role assessed: ${session.role} (${session.profile} track)
Candidate: ${session.name || "Candidate"}
Interview language: ${session.language || "English"}

Full interview record:
${history}

Produce a final assessment report IN ENGLISH (regardless of interview language). Weight ${d1} 50 percent, ${d2} 30 percent, ${d3} 20 percent when computing the overall score out of 100. Be honest and evidence based, citing what the candidate actually said.

Respond with JSON only:
{
  "overall_score": n,
  "band": "<one of: Expert, Proficient, Developing, Foundational>",
  "summary": "<3 to 4 sentence executive summary of the candidate's demonstrated capability for this role>",
  "strengths": ["<specific strength grounded in their answers>", "..."],
  "improvements": ["<specific development area>", "..."],
  "recommendation": "<one sentence hiring style recommendation with a suggested next step>"
}`;
}

// ── Routes ────────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    anthropic: !!ANTHROPIC_API_KEY,
    openai: !!OPENAI_API_KEY,
    did: !!DID_API_KEY,
    didLastError: lastDidError,
    model: CLAUDE_MODEL,
  });
});

// Diagnostic: open in the browser to test the D-ID pipeline in one shot
app.get("/api/did-test", async (req, res) => {
  const auth = didAuthHeader();
  if (!auth) return res.json({ step: "key", ok: false, detail: "DID_API_KEY is not set" });
  const result = { keyPresent: true };
  try {
    const credResp = await fetch("https://api.d-id.com/credits", { headers: { Authorization: auth } });
    const credBody = await credResp.text();
    result.creditsCheck = { status: credResp.status, body: credBody.slice(0, 400) };
    if (!credResp.ok) {
      result.diagnosis =
        credResp.status === 401
          ? "The key is not authenticating. Copy it again from the D-ID Account page exactly as shown."
          : "Unexpected response from D-ID on the credits check.";
      return res.json(result);
    }
    const t0 = Date.now();
    const avatar = await createAvatarVideo("Hello, this is a quick system test.", "en-IN-NeerjaNeural");
    result.renderCheck = {
      seconds: Math.round((Date.now() - t0) / 1000),
      success: !!avatar.videoUrl,
      reason: avatar.reason || null,
      lastError: lastDidError,
      videoUrl: avatar.videoUrl || null,
    };
    result.diagnosis = avatar.videoUrl
      ? "D-ID is working. If the app still shows the illustrated interviewer, redeploy and retry."
      : "Key authenticates but rendering fails. See lastError above for the exact cause.";
    res.json(result);
  } catch (e) {
    result.error = e.message;
    res.json(result);
  }
});

// Expose the role library so the frontend always matches the backend
app.get("/api/roles", (req, res) => {
  res.json({
    tracks: Object.fromEntries(
      Object.entries(TRACKS).map(([k, v]) => [k, { roles: v.roles, dims: v.dims }])
    ),
    languages: Object.keys(LANGUAGES),
  });
});

app.post("/api/start", async (req, res) => {
  try {
    const { name, profile, role, language } = req.body || {};
    const track = TRACKS[profile];
    if (!track) return res.status(400).json({ error: "Invalid profile selected." });
    if (!track.roles.includes(role)) return res.status(400).json({ error: "Invalid role for this profile." });
    const lang = LANGUAGES[language] ? language : "English";
    if (!ANTHROPIC_API_KEY || !OPENAI_API_KEY)
      return res.status(500).json({ error: "Server is missing ANTHROPIC_API_KEY or OPENAI_API_KEY" });

    const id = crypto.randomUUID();
    const session = {
      id,
      name: (name || "").slice(0, 60),
      profile,
      role,
      language: lang,
      dims: track.dims,
      startedAt: Date.now(),
      qa: [],
      currentQuestion: null,
      qIndex: 0,
      done: false,
    };

    const raw = await callClaude(interviewerSystem(session), firstQuestionPrompt(session), 600);
    const q = parseJsonLoose(raw);
    session.currentQuestion = q.question;
    sessions.set(id, session);

    const avatar = await createAvatarVideo(q.question, LANGUAGES[lang].voice);
    res.json({
      sessionId: id,
      questionNumber: 1,
      totalQuestions: TOTAL_QUESTIONS,
      questionText: q.question,
      videoUrl: avatar.videoUrl,
      fallback: avatar.fallback || false,
      dims: track.dims,
      speechLocale: LANGUAGES[lang].bcp,
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

    const langCode = LANGUAGES[session.language]?.whisper || "en";
    const transcript = await transcribeAudio(req.file.buffer, req.file.mimetype, req.file.originalname, langCode);
    if (!transcript || transcript.replace(/\s/g, "").length < 8) {
      return res.json({
        retry: true,
        message: "We could not hear a clear answer. Please check your microphone and record again.",
      });
    }

    const isLast = session.qIndex >= TOTAL_QUESTIONS - 1;
    const raw = await callClaude(interviewerSystem(session), evaluateAndNextPrompt(session, transcript, isLast), 900);
    const evalData = parseJsonLoose(raw);

    session.qa.push({
      question: session.currentQuestion,
      transcript,
      eval: {
        scores: [Number(evalData.d1) || 0, Number(evalData.d2) || 0, Number(evalData.d3) || 0],
        feedback: evalData.feedback || "",
      },
    });

    if (isLast) {
      session.done = true;
      return res.json({ transcript, done: true });
    }

    session.qIndex += 1;
    session.currentQuestion = evalData.next_question;
    const avatar = await createAvatarVideo(evalData.next_question, LANGUAGES[session.language].voice);
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
      "You are a rigorous, fair talent assessment analyst. Respond with strict JSON only.",
      reportPrompt(session),
      1200
    );
    const report = parseJsonLoose(raw);

    res.json({
      candidate: session.name || "Candidate",
      profile: session.profile,
      role: session.role,
      language: session.language,
      dimensions: session.dims,
      date: new Date(session.startedAt).toDateString(),
      report,
      questions: session.qa.map((q, i) => ({
        number: i + 1,
        question: q.question,
        transcript: q.transcript,
        scores: q.eval.scores,
        note: q.eval.feedback,
      })),
    });
  } catch (e) {
    console.error("report error:", e.message);
    res.status(500).json({ error: "Could not generate the report. " + e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`SkillVoice Agentic Interviewer running on port ${PORT}`));
