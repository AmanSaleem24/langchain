
const status = document.getElementById("status");
const result = document.getElementById("result");

async function getVideoId() {
  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  if (!tab?.url) {
    throw new Error("Could not identify the active tab.");
  }

  const url = new URL(tab.url);

  if (url.hostname === "youtu.be") {
    return url.pathname.slice(1);
  }

  if (
    url.hostname === "www.youtube.com" ||
    url.hostname === "youtube.com" ||
    url.hostname === "m.youtube.com"
  ) {
    if (url.pathname === "/watch") {
      return url.searchParams.get("v");
    }

    const match = url.pathname.match(
      /^\/(?:shorts|embed)\/([^/]+)/
    );

    if (match) return match[1];
  }

  throw new Error("Open a YouTube video first.");
}

async function callBackend(endpoint, payload) {
  const response = await fetch(
    `http://127.0.0.1:8000/${endpoint}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Backend request failed.");
  }

  return data;
}

document.getElementById("summarize").addEventListener(
  "click",
  async () => {
    status.textContent = "Summarizing video...";
    result.textContent = "";

    try {
      const video_id = await getVideoId();
      const data = await callBackend("summarize", { video_id });

      result.textContent = data.summary;
      status.textContent = "Summary ready.";
    } catch (error) {
      status.textContent = error.message;
    }
  }
);

document.getElementById("ask").addEventListener(
  "click",
  async () => {
    status.textContent = "Searching transcript...";
    result.textContent = "";

    try {
      const video_id = await getVideoId();
      const question = document.getElementById("question")
        .value.trim();

      if (!question) {
        throw new Error("Enter a question first.");
      }

      const data = await callBackend("ask", {
        video_id,
        question
      });

      result.textContent = data.answer;
      status.textContent = "Answer ready.";
    } catch (error) {
      status.textContent = error.message;
    }
  }
);
