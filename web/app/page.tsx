"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Source = { title: string; url: string };
type Message = {
  id: number;
  role: "assistant" | "user";
  content: string;
  sources?: Source[];
  risk?: string;
};
type Profile = { goal: string; experience: string; equipment: string; schedule: string; limitations: string };

const emptyProfile: Profile = { goal: "", experience: "", equipment: "", schedule: "", limitations: "" };
const suggestions = [
  "Build a 3-day beginner strength plan",
  "How can I improve my squat form?",
  "Help me recover between workouts",
];

function localReply(question: string) {
  const normalized = question.toLowerCase();
  if (/chest pain|faint|cannot breathe|can't breathe/.test(normalized)) {
    return "Stop exercising now. These symptoms can require urgent medical attention. Contact your local emergency service or seek urgent in-person care; do not rely on this chat to assess the cause.";
  }
  if (normalized.includes("pain") || normalized.includes("injur")) {
    return "Pause the movement and avoid training through sharp, sudden, or worsening pain. Persistent pain, swelling, weakness, numbness, or an inability to bear weight should be assessed by a qualified clinician.";
  }
  if (normalized.includes("squat")) {
    return "Start with a stable stance, keep pressure across the whole foot, brace before descending, and let your knees track with your toes. Use a pain-free depth you can control. A side and front video can help narrow down what to adjust.";
  }
  return "Tell me your experience level, available equipment, training days, and any limitations so I can make this practical for you.";
}

export default function Home() {
  const [draft, setDraft] = useState("");
  const [profile, setProfile] = useState<Profile>(emptyProfile);
  const [editingProfile, setEditingProfile] = useState(false);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: "assistant",
      content:
        "Hi, I’m Form. I can help with training plans, exercise technique, recovery, and general nutrition—while keeping your context and safety in view. What are you working toward?",
    },
  ]);
  const endRef = useRef<HTMLDivElement>(null);

  const messageCount = useMemo(() => messages.filter((message) => message.role === "user").length, [messages]);
  const profileCount = Object.values(profile).filter(Boolean).length;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, loading]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = draft.trim();
    if (!question || loading) return;
    const now = Date.now();
    setMessages((current) => [...current, { id: now, role: "user", content: question }]);
    setDraft("");
    setLoading(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, profile }),
      });
      if (!response.ok) throw new Error("Chat request failed");
      const result = await response.json();
      setMessages((current) => [
        ...current,
        { id: now + 1, role: "assistant", content: result.answer, sources: result.sources, risk: result.risk_level },
      ]);
    } catch {
      setMessages((current) => [
        ...current,
        { id: now + 1, role: "assistant", content: localReply(question), risk: "offline" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="context-panel">
        <div>
          <div className="brand-mark" aria-hidden="true">F</div>
          <p className="eyebrow">FORM / FITNESS COACH</p>
          <h1>Train with context, not guesswork.</h1>
          <p className="intro">Evidence-aware guidance shaped around your goals, experience, and equipment.</p>
        </div>

        <div className="profile-card">
          <div className="profile-heading">
            <span>Your training context</span>
            <span className="status-dot">{profileCount ? `${profileCount}/5 set` : "Optional"}</span>
          </div>
          {editingProfile ? (
            <div className="profile-form">
              {(["goal", "experience", "equipment", "schedule", "limitations"] as const).map((field) => (
                <label key={field}>
                  <span>{field}</span>
                  <input
                    value={profile[field]}
                    onChange={(event) => setProfile((current) => ({ ...current, [field]: event.target.value }))}
                    placeholder={field === "schedule" ? "e.g. 3 days/week" : `Your ${field}`}
                  />
                </label>
              ))}
              <button type="button" onClick={() => setEditingProfile(false)}>Save context</button>
            </div>
          ) : (
            <>
              <dl>
                <div><dt>Goal</dt><dd>{profile.goal || "Not set"}</dd></div>
                <div><dt>Experience</dt><dd>{profile.experience || "Not set"}</dd></div>
                <div><dt>Equipment</dt><dd>{profile.equipment || "Not set"}</dd></div>
              </dl>
              <button type="button" onClick={() => setEditingProfile(true)}>{profileCount ? "Edit context" : "Add context"}</button>
            </>
          )}
        </div>

        <p className="scope-note">General fitness education, not diagnosis or medical treatment.</p>
      </aside>

      <section className="chat-panel" aria-label="Fitness coaching chat">
        <header className="chat-header">
          <div><p className="eyebrow">TODAY&apos;S SESSION</p><h2>Ask Form</h2></div>
          <span className="session-count">{messageCount} questions</span>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <span className="message-label">{message.role === "assistant" ? "FORM" : "YOU"}</span>
              <p>{message.content}</p>
              {message.sources && message.sources.length > 0 && (
                <div className="source-list">
                  {message.sources.map((source) => (
                    <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.title} ↗</a>
                  ))}
                </div>
              )}
              {message.risk === "offline" && <span className="offline-note">Offline safety fallback</span>}
            </article>
          ))}
          {loading && <article className="message assistant loading"><span className="message-label">FORM</span><p>Checking the guidance<span>…</span></p></article>}
          <div ref={endRef} />
        </div>

        {messageCount === 0 && (
          <div className="suggestions" aria-label="Suggested questions">
            {suggestions.map((suggestion) => (
              <button key={suggestion} type="button" onClick={() => setDraft(suggestion)}>
                {suggestion}<span aria-hidden="true">↗</span>
              </button>
            ))}
          </div>
        )}

        <form className="composer" onSubmit={submit}>
          <label htmlFor="question" className="sr-only">Ask a fitness question</label>
          <textarea
            id="question"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask about your training..."
            rows={2}
          />
          <button type="submit" aria-label="Send question" disabled={!draft.trim() || loading}>
            <span>{loading ? "Wait" : "Send"}</span><span aria-hidden="true">↑</span>
          </button>
        </form>
      </section>
    </main>
  );
}
