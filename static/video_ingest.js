(() => {
  const form = document.getElementById("staticVideoForm");
  const signSelect = document.getElementById("videoSignLabel");
  const signerInput = document.getElementById("videoSignerId");
  const signerOptions = document.getElementById("videoSignerOptions");
  const fileInput = document.getElementById("staticVideoFile");
  const preview = document.getElementById("staticVideoPreview");
  const submitButton = document.getElementById("processStaticVideo");
  const status = document.getElementById("staticVideoStatus");
  const tabButton = document.querySelector('.tab-btn[data-tab="video-data"]');
  if (!form || !signSelect || !signerInput || !fileInput || !submitButton || !status || !tabButton) return;

  let loadedOptions = false;
  let loadingOptions = null;
  let previewUrl = "";

  function setStatus(message, kind = "info") {
    status.textContent = message;
    status.dataset.state = kind;
    status.hidden = !message;
  }

  async function loadOptions(force = false) {
    if (loadedOptions && !force) return;
    if (loadingOptions && !force) return loadingOptions;
    loadingOptions = (async () => {
      setStatus("Loading the sign catalog…");
      try {
        const response = await fetch("/api/static-video/options", { credentials: "same-origin" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || "Could not load the sign catalog.");
        const selected = signSelect.value;
        signSelect.replaceChildren(new Option("Select a sign", ""));
        for (const sign of payload.signs || []) signSelect.add(new Option(sign.replaceAll("_", " "), sign));
        if ((payload.signs || []).includes(selected)) signSelect.value = selected;
        signerOptions?.replaceChildren(...(payload.signers || []).map(value => new Option(value, value)));
        loadedOptions = true;
        setStatus("");
      } catch (error) {
        setStatus(error.message || "Could not load the sign catalog.", "error");
      } finally {
        loadingOptions = null;
      }
    })();
    return loadingOptions;
  }

  tabButton.addEventListener("click", () => loadOptions());

  fileInput.addEventListener("change", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = "";
    const file = fileInput.files?.[0];
    if (!file) {
      preview.hidden = true;
      preview.removeAttribute("src");
      return;
    }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const file = fileInput.files?.[0];
    if (!file) return;
    if (file.size > 250 * 1024 * 1024) {
      setStatus("Choose a video smaller than 250 MB.", "error");
      return;
    }

    const payload = new FormData();
    payload.append("video", file, file.name);
    payload.append("sign_label", signSelect.value);
    payload.append("signer_id", signerInput.value.trim());
    submitButton.disabled = true;
    setStatus("Processing video and extracting hand landmarks…");
    try {
      const response = await fetch("/api/static-video/import", {
        method: "POST",
        body: payload,
        credentials: "same-origin",
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.error || "Video processing failed.");
      const summary = `${result.sign_label} · ${result.signer_id}: ${result.samples_saved} samples from ${result.frames_sampled} sampled frames. Saved to ${result.staging_file}; production data unchanged.`;
      setStatus(summary, "success");
      await loadOptions(true);
    } catch (error) {
      setStatus(error.message || "Video processing failed. No samples were added.", "error");
    } finally {
      submitButton.disabled = false;
      if (typeof lucide !== "undefined") lucide.createIcons();
    }
  });
})();
