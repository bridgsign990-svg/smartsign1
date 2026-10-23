(() => {
  const form = document.getElementById("videoRecognitionForm");
  if (!form) return;

  const fileInput = document.getElementById("videoRecognitionFile");
  const submitButton = document.getElementById("videoRecognitionSubmit");
  const status = document.getElementById("videoRecognitionStatus");
  const results = document.getElementById("videoRecognitionResults");
  const modelLabel = document.getElementById("videoRecognitionModel");
  const predictions = document.getElementById("videoRecognitionPredictions");

  form.addEventListener("submit", async event => {
    event.preventDefault();
    results.hidden = true;
    predictions.replaceChildren();
    status.dataset.state = "";
    status.textContent = "Uploading and analyzing the isolated sign clip…";
    submitButton.disabled = true;

    try {
      const response = await fetch("/api/video-recognition/predict", {
        method: "POST",
        body: new FormData(form),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Video analysis failed.");

      modelLabel.textContent = `${payload.model} · ${payload.language} · ${payload.frames_processed} frames`;
      for (const prediction of payload.predictions) {
        const row = document.createElement("div");
        row.className = "video-prediction";
        const label = document.createElement("strong");
        label.textContent = prediction.label;
        const confidence = document.createElement("span");
        confidence.textContent = `${(prediction.confidence * 100).toFixed(1)}% model score`;
        row.append(label, confidence);
        predictions.append(row);
      }
      results.hidden = false;
      status.dataset.state = "success";
      status.textContent = "Analysis complete. Review these as model guesses, not guaranteed translations.";
    } catch (error) {
      status.dataset.state = "error";
      status.textContent = error.message || "Video analysis failed.";
    } finally {
      submitButton.disabled = false;
    }
  });
})();
