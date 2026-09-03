type Profile = Record<string, string>;

const sources = {
  activity: { title: "WHO physical activity guidance", url: "https://www.who.int/publications/i/item/9789240014886" },
  squat: { title: "NSCA squat principles", url: "https://www.nsca.com/education/articles/kinetic-select/the-back-squat/" },
  nutrition: { title: "WHO healthy diet", url: "https://www.who.int/news-room/fact-sheets/detail/healthy-diet" },
};

function fallback(message: string, profile: Profile) {
  const text = message.toLowerCase();
  const context = [profile.goal, profile.experience, profile.equipment].filter(Boolean).join(", ");
  const tailored = context ? ` Based on the context you shared (${context}), start conservatively and adjust from your results.` : "";
  if (/chest pain|faint|cannot breathe|can't breathe/.test(text)) {
    return { answer: "Stop exercising now. These symptoms can require urgent medical attention. Contact your local emergency service or seek urgent in-person care; do not rely on this chat to assess the cause.", risk_level: "urgent", sources: [], model_backend: "safety-policy" };
  }
  if (/sharp pain|severe pain|swelling|numbness|injur/.test(text)) {
    return { answer: "Pause the activity. I can provide general education, but persistent, severe, or worsening symptoms should be assessed by a qualified clinician.", risk_level: "high", sources: [], model_backend: "site-fallback" };
  }
  if (text.includes("squat")) {
    return { answer: `Use a stable stance, maintain pressure across the whole foot, brace before descending, and let your knees track with your toes. Choose a depth you can control without pain.${tailored}`, risk_level: "standard", sources: [sources.squat], model_backend: "site-fallback" };
  }
  if (/diet|nutrition|protein|meal/.test(text)) {
    return { answer: `Build meals around adequate energy, varied foods, protein sources, fruits or vegetables, and habits you can sustain.${tailored}`, risk_level: "standard", sources: [sources.nutrition], model_backend: "site-fallback" };
  }
  return { answer: `Build from a manageable starting dose, keep technique consistent, and progress one variable at a time.${tailored || " Share your goal, experience, equipment, and weekly schedule for a more specific plan."}`, risk_level: "standard", sources: [sources.activity], model_backend: "site-fallback" };
}

export async function POST(request: Request) {
  const body = await request.json() as { message?: string; profile?: Profile };
  const message = body.message?.trim();
  if (!message) return Response.json({ error: "message is required" }, { status: 422 });

  const upstream = process.env.FITNESS_API_URL?.replace(/\/$/, "");
  if (upstream) {
    try {
      const response = await fetch(`${upstream}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, profile: body.profile ?? {} }),
      });
      if (response.ok) return new Response(response.body, { status: response.status, headers: { "Content-Type": "application/json" } });
    } catch {
      // The grounded site fallback keeps the product useful if the model service is unavailable.
    }
  }
  return Response.json(fallback(message, body.profile ?? {}));
}
