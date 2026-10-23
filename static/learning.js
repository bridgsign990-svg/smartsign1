const learningState = {
  dashboard: null,
  lesson: null,
  stage: "introduction",
  signIndex: 0,
  quiz: null,
  questionIndex: 0,
  answeredIds: new Set(),
  cameraStream: null,
  cameraBusy: false,
  avatarReady: false,
  loadingDashboard: false,
  avatarHost: document.getElementById("learningAvatarHost"),
};

const learningStages = [
  ["introduction", "Introduction"],
  ["demonstration", "Watch"],
  ["explanation", "Breakdown"],
  ["practice", "Practice"],
  ["exercise", "Mini challenge"],
  ["test", "Test"],
];

function learningEscape(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function learningValidationLabel(sign) {
  return sign.validated
    ? "Content reviewed by a qualified sign-language reviewer."
    : "Learning content pending sign-language review.";
}

function learningSignType(sign) {
  if (sign.content_type === "LETTER") return "Fingerspelled letter";
  if (sign.content_type === "WORD") return "Word sign";
  return "Sign type not specified";
}

function learningSignCategory(sign) {
  if (!sign.category || sign.category === "unclassified") return "Category not set";
  return sign.category.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function learningDifficulty(sign) {
  if (!sign.difficulty || sign.difficulty === "unspecified") return "Not set";
  return sign.difficulty.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function learningLessonStatus(status) {
  return ({
    unlocked: "Ready to learn",
    in_progress: "In progress",
    review_required: "Ready to continue",
    completed: "Completed",
  })[status] || "Lesson";
}

async function learningRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Learning request failed (${response.status}).`);
  return data;
}

function setLearningError(message = "") {
  const error = document.getElementById("learningError");
  if (!error) return;
  error.textContent = message;
  error.hidden = !message;
}

async function refreshLearningDashboard() {
  if (learningState.loadingDashboard) return;
  learningState.loadingDashboard = true;
  setLearningError();
  const content = document.getElementById("learningDashboardContent");
  if (!content) return;
  content.innerHTML = '<p class="learning-state">Loading your course…</p>';
  try {
    const dashboard = await learningRequest("/api/learning/progress");
    learningState.dashboard = dashboard;
    renderLearningDashboard(dashboard);
  } catch (error) {
    content.innerHTML = `<p class="learning-state">${learningEscape(error.message)}</p>`;
  } finally {
    learningState.loadingDashboard = false;
  }
}

function renderLearningDashboard(dashboard) {
  const content = document.getElementById("learningDashboardContent");
  if (!content) return;
  const course = dashboard.courses?.[0];
  if (!course) {
    content.innerHTML = '<p class="learning-state">No courses are available yet.</p>';
    return;
  }
  const currentLesson = course.current_lesson;
  const recommendations = dashboard.needs_practice || [];
  const recent = dashboard.recent_tests || [];
  const lessonRows = course.modules.map(module => `
    <section class="course-module" aria-labelledby="module-${module.id}">
      <div class="course-module-title"><strong id="module-${module.id}">Module ${module.order}: ${learningEscape(module.title)}</strong><span>${learningEscape(module.description)}</span></div>
      ${module.lessons.map(lesson => {
        const icon = lesson.status === "completed" ? "check" : lesson.status === "locked" ? "lock-keyhole" : lesson.status === "review_required" ? "rotate-ccw" : "play";
        const label = lesson.status === "completed" ? "Review" : lesson.status === "locked" ? "Locked" : lesson.status === "review_required" ? "Review" : "Start";
        return `<div class="course-lesson-row">
          <span class="lesson-status-mark"><i data-lucide="${icon}" width="16" height="16" aria-hidden="true"></i></span>
          <span><span class="lesson-row-title">${learningEscape(lesson.title)}</span><span class="lesson-row-sub">${learningEscape(lesson.description)}</span></span>
          <button type="button" data-open-lesson="${lesson.id}" ${lesson.status === "locked" ? "disabled" : ""} aria-label="${label} ${learningEscape(lesson.title)}">${label}</button>
        </div>`;
      }).join("")}
    </section>`).join("");
  const practiceRows = recommendations.length
    ? recommendations.map(item => {
      const lesson = course.modules.flatMap(module => module.lessons).find(candidate => candidate.sign_ids.includes(item.sign_id));
      return `<div class="recommendation-row"><span><strong>${learningEscape(item.name)}</strong><small>${Math.round(item.score)}% learning indicator</small></span>${lesson ? `<button type="button" class="action-btn secondary-btn" data-practice-lesson="${lesson.id}" data-practice-sign="${item.sign_id}" title="Practice ${learningEscape(item.name)}"><i data-lucide="rotate-cw" width="14" height="14"></i></button>` : ""}</div>`;
    }).join("")
    : '<p>No signs are flagged for extra practice yet.</p>';
  const recentRows = recent.length
    ? recent.map(item => `<div class="recent-test-row"><span>${learningEscape(item.lesson)}</span><strong>${item.score}/${item.total} · ${item.percentage}%</strong></div>`).join("")
    : '<p>No tests completed yet.</p>';
  content.innerHTML = `
    <section class="course-overview">
      <h3>${learningEscape(course.title)}</h3>
      <p class="course-meta">${learningEscape(course.description)} · ${learningEscape(course.level)} level</p>
      <div class="course-progress-track" role="progressbar" aria-label="Course progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${course.progress}"><div class="course-progress-fill" style="width:${course.progress}%"></div></div>
      <div class="course-progress-label"><span>${course.progress}% complete</span><span>${course.completed_lessons} of ${course.total_lessons} lessons</span></div>
    </section>
    <div class="learning-stats">
      <div class="learning-stat"><strong>${course.completed_lessons}/${course.total_lessons}</strong><span>Lessons completed</span></div>
      <div class="learning-stat"><strong>${dashboard.mastered_signs?.length || 0}</strong><span>Signs mastered</span></div>
      <div class="learning-stat"><strong>${recommendations.length}</strong><span>Need practice</span></div>
    </div>
    <section class="course-outline" aria-label="Course lessons">
      <h3>Course outline</h3>
      ${lessonRows}
    </section>
    <aside class="learning-aside">
      <h3>Continue learning</h3>
      ${currentLesson ? `<p>${learningEscape(currentLesson.module)} · ${learningEscape(currentLesson.title)}</p><button class="action-btn start-btn" type="button" data-open-lesson="${currentLesson.id}" style="margin:12px 0"><i data-lucide="arrow-right" width="15" height="15"></i> Continue lesson</button>` : "<p>You've completed every lesson in this course.</p>"}
      <h3 style="margin-top:24px">Recommended practice</h3>${practiceRows}
      <h3 style="margin-top:24px">Recent tests</h3>${recentRows}
      <h3 style="margin-top:24px">Mastery</h3><div id="masterySummary"></div>
    </aside>`;
  const outline = content.querySelector(".course-outline");
  outline.addEventListener("click", event => {
    const button = event.target.closest("[data-open-lesson]");
    if (button && !button.disabled) openLearningLesson(Number(button.dataset.openLesson));
  });
  content.querySelector(".learning-aside").addEventListener("click", event => {
    const button = event.target.closest("[data-open-lesson]");
    if (button) openLearningLesson(Number(button.dataset.openLesson));
    const practice = event.target.closest("[data-practice-lesson]");
    if (practice) openLearningLesson(Number(practice.dataset.practiceLesson), { stage: "practice", signId: Number(practice.dataset.practiceSign) });
  });
  learningRequest("/api/learning/mastery").then(data => {
    const summary = document.getElementById("masterySummary");
    if (!summary) return;
    summary.innerHTML = data.signs?.length
      ? data.signs.slice(0, 4).map(sign => `<div class="recent-test-row"><span>${learningEscape(sign.name)}</span><strong>${sign.score}%</strong></div>`).join("")
      : "<p>Practice attempts will build your personal learning indicators.</p>";
  }).catch(() => {});
  if (typeof lucide !== "undefined") lucide.createIcons();
}

async function openLearningLesson(lessonId, options = {}) {
  setLearningError();
  stopLearningCamera();
  try {
    const lesson = await learningRequest(`/api/learning/lessons/${lessonId}`);
    learningState.lesson = lesson;
    learningState.signIndex = 0;
    learningState.quiz = null;
    learningState.answeredIds = new Set();
    learningState.stage = options.stage || (learningStages.some(([id]) => id === lesson.current_section) ? lesson.current_section : "introduction");
    if (options.signId) {
      const signIndex = lesson.signs.findIndex(sign => sign.id === options.signId);
      if (signIndex >= 0) learningState.signIndex = signIndex;
    }
    document.getElementById("learningDashboard").hidden = true;
    document.getElementById("learningLessonView").hidden = false;
    renderLearningLesson();
    if (lesson.current_section === "test" && lesson.active_quiz_session) {
      learningState.stage = "test";
      await startOrResumeQuiz(lesson.active_quiz_session);
    }
  } catch (error) {
    setLearningError(error.message);
  }
}

function closeLearningLesson() {
  stopLearningCamera();
  const avatarHost = learningState.avatarHost;
  if (avatarHost) {
    avatarHost.hidden = true;
    document.getElementById("learningLessonView").appendChild(avatarHost);
  }
  document.getElementById("learningLessonView").hidden = true;
  document.getElementById("learningDashboard").hidden = false;
  refreshLearningDashboard();
}

function renderLearningLesson() {
  const lesson = learningState.lesson;
  if (!lesson) return;
  const lessonView = document.getElementById("learningLessonView");
  if (learningState.avatarHost?.parentElement !== lessonView) lessonView.appendChild(learningState.avatarHost);
  const index = learningState.dashboard?.courses?.[0]?.modules
    ?.flatMap(module => module.lessons).findIndex(item => item.id === lesson.id) ?? -1;
  document.getElementById("lessonPosition").textContent = `${lesson.module} · Lesson ${index + 1}`;
  document.getElementById("lessonContent").innerHTML = `
    <header class="lesson-heading">
      <p class="card-eyebrow">${learningEscape(lesson.module)} · ${learningEscape(learningLessonStatus(lesson.status))}</p>
      <h1>${learningEscape(lesson.title)}</h1><p>${learningEscape(lesson.description)}</p>
    </header>
    <nav class="lesson-stage-nav" aria-label="Lesson sections">${learningStages.map(([id, label], stageIndex) => `<button type="button" data-stage="${id}" aria-current="${learningState.stage === id ? "step" : "false"}" ${learningCanOpenStage(id, stageIndex) ? "" : "disabled"}>${learningEscape(label)}</button>`).join("")}</nav>
    <section id="learningStage" class="lesson-stage" tabindex="-1"></section>
    <footer class="lesson-footer"><button type="button" class="action-btn secondary-btn" id="previousLessonStage"><i data-lucide="arrow-left" width="15" height="15"></i> Back</button><button type="button" class="action-btn start-btn" id="nextLessonStage">Continue <i data-lucide="arrow-right" width="15" height="15"></i></button></footer>`;
  const content = document.getElementById("lessonContent");
  content.querySelector(".lesson-stage-nav").addEventListener("click", event => {
    const button = event.target.closest("[data-stage]");
    if (button) changeLearningStage(button.dataset.stage);
  });
  document.getElementById("previousLessonStage").addEventListener("click", () => changeLearningStage(learningStages[Math.max(0, learningStages.findIndex(([id]) => id === learningState.stage) - 1)][0]));
  document.getElementById("nextLessonStage").addEventListener("click", () => {
    if (learningState.stage === "test") return;
    const nextIndex = learningStages.findIndex(([id]) => id === learningState.stage) + 1;
    if (nextIndex < learningStages.length && learningCanOpenStage(learningStages[nextIndex][0], nextIndex)) changeLearningStage(learningStages[nextIndex][0]);
  });
  const nextIndex = learningStages.findIndex(([id]) => id === learningState.stage) + 1;
  document.getElementById("nextLessonStage").disabled = nextIndex >= learningStages.length || !learningCanOpenStage(learningStages[nextIndex]?.[0], nextIndex);
  renderLearningStage();
}

function learningCanOpenStage(stage, stageIndex) {
  const lesson = learningState.lesson;
  if (!lesson || !stage) return false;
  if (lesson.status === "completed") return stage !== "test";
  const maxReached = Math.round((lesson.progress_percentage || 0) * 5 / 100);
  if (stageIndex > maxReached + 1) return false;
  if (stage === "test" && !lesson.exercise_completed && !lesson.active_quiz_session) return false;
  return true;
}

async function changeLearningStage(stage) {
  if (!learningStages.some(([id]) => id === stage)) return;
  stopLearningCamera();
  try {
    await learningRequest(`/api/learning/lessons/${learningState.lesson.id}/progress`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ section: stage }),
    });
  } catch (error) {
    setLearningError(error.message);
    return;
  }
  learningState.stage = stage;
  learningState.lesson.current_section = stage;
  learningState.lesson.progress_percentage = Math.max(learningState.lesson.progress_percentage || 0, Math.round(learningStages.findIndex(([id]) => id === stage) * 100 / 5));
  renderLearningLesson();
  if (stage === "test" && learningState.lesson.active_quiz_session && !learningState.quiz) {
    await startOrResumeQuiz(learningState.lesson.active_quiz_session);
  }
}

function currentLearningSign() {
  return learningState.lesson?.signs?.[learningState.signIndex] || null;
}

function renderLearningStage() {
  const stage = document.getElementById("learningStage");
  const lesson = learningState.lesson;
  if (!stage || !lesson) return;
  document.querySelectorAll(".lesson-stage-nav [data-stage]").forEach(button => {
    button.setAttribute("aria-current", button.dataset.stage === learningState.stage ? "step" : "false");
  });
  const stageIndex = learningStages.findIndex(([id]) => id === learningState.stage);
  document.getElementById("previousLessonStage").disabled = stageIndex <= 0;
  document.getElementById("nextLessonStage").hidden = learningState.stage === "test";
  document.querySelector(".lesson-footer").hidden = learningState.stage === "result";
  document.querySelector(".lesson-stage-nav").hidden = learningState.stage === "result";
  const avatarHost = learningState.avatarHost;
  if (avatarHost.parentElement !== document.getElementById("learningLessonView")) document.getElementById("learningLessonView").appendChild(avatarHost);
  avatarHost.hidden = true;
  if (learningState.stage === "introduction") {
    stage.innerHTML = `<h2>Start with the meaning</h2><p>${learningEscape(lesson.introduction)}</p><div class="lesson-objective"><strong>Learning objective:</strong> ${learningEscape(lesson.objective)}</div><p>This lesson introduces ${lesson.signs.map(sign => learningEscape(sign.display_name)).join(", ")}.</p>`;
  } else if (learningState.stage === "demonstration") {
    const sign = currentLearningSign();
    if (!sign) return;
    const avatarDescription = sign.avatar_available
      ? "Avatar demonstration is available; playback does not establish sign accuracy."
      : "Avatar demonstration is not available for this sign.";
    const contentStatus = learningValidationLabel(sign);
    stage.innerHTML = `<h2>Watch the sign</h2><p>Study the available reference, then compare it with the written information. An asset loading successfully does not verify linguistic accuracy.</p>
      <div class="sign-tabs" role="group" aria-label="Signs in this lesson">${lesson.signs.map((item, index) => `<button type="button" data-sign-index="${index}" aria-pressed="${index === learningState.signIndex}">${learningEscape(item.display_name)}</button>`).join("")}</div>
      <div class="sign-learning-layout"><div class="sign-detail"><h3>${learningEscape(sign.display_name)}</h3><p class="sign-meta-line">${learningEscape(learningSignType(sign))} · ${learningEscape(learningSignCategory(sign))} · Difficulty: ${learningEscape(learningDifficulty(sign))}</p><p><strong>Meaning:</strong> ${learningEscape(sign.meaning || "Requires sign-language content authoring.")}</p><p><strong>Available description:</strong> ${learningEscape(sign.description || "No source description is available.")}</p><p><strong>Movement notes:</strong> ${learningEscape(sign.movement_notes || (sign.description ? "No separate movement notes are recorded; see the available description above." : "Requires sign-language content authoring."))}</p><p><strong>When to use it:</strong> ${learningEscape(sign.usage_context || "Requires sign-language content authoring.")}</p><p><strong>Example:</strong> ${learningEscape(sign.example_sentence || "Requires sign-language content authoring.")}</p><p class="content-status-line">${learningEscape(contentStatus)} ${learningEscape(avatarDescription)} ${sign.recognition_available ? (sign.practice_mode === "motion" ? "Camera practice (motion clip) is available." : "Static letter camera practice is available.") : "Camera practice is not available for this sign."}</p>
      ${sign.avatar_available ? `<div class="avatar-actions"><button type="button" class="action-btn start-btn" data-play-sign="${learningState.signIndex}"><i data-lucide="play" width="15" height="15"></i> Play</button><button type="button" class="action-btn secondary-btn" data-play-sign="${learningState.signIndex}"><i data-lucide="rotate-cw" width="15" height="15"></i> Replay</button><button type="button" class="action-btn secondary-btn" data-reset-avatar><i data-lucide="rotate-ccw" width="15" height="15"></i> Reset</button><button type="button" class="action-btn secondary-btn" data-previous-sign ${learningState.signIndex === 0 ? "disabled" : ""}><i data-lucide="arrow-left" width="15" height="15"></i> Previous sign</button><button type="button" class="action-btn secondary-btn" data-next-sign ${learningState.signIndex >= lesson.signs.length - 1 ? "disabled" : ""}>Next sign <i data-lucide="arrow-right" width="15" height="15"></i></button></div>` : `<p class="practice-note">${learningEscape(avatarDescription)}</p>`}</div>${sign.avatar_available ? '<div class="avatar-mount"></div>' : '<div class="avatar-unavailable" role="status">Avatar asset unavailable. Written lesson content remains available.</div>'}</div>
      ${sign.image_available ? `<figure class="sign-reference-image"><img src="${learningEscape(sign.image_url)}" alt="Repository illustration for ${learningEscape(sign.display_name)}"><figcaption>Illustration reference · ${learningEscape(contentStatus)}</figcaption></figure>` : ""}
      ${sign.video_available ? `<video controls src="${learningEscape(sign.video_url)}" aria-label="${learningEscape(sign.display_name)} sign reference"></video>` : ""}`;
    stage.querySelectorAll("[data-sign-index]").forEach(button => button.addEventListener("click", () => {
      learningState.signIndex = Number(button.dataset.signIndex);
      renderLearningStage();
    }));
    stage.querySelectorAll("[data-play-sign]").forEach(button => button.addEventListener("click", () => playLearningSign(currentLearningSign())));
    stage.querySelector("[data-reset-avatar]")?.addEventListener("click", resetLearningAvatar);
    stage.querySelector("[data-previous-sign]")?.addEventListener("click", () => changeLearningSign(-1));
    stage.querySelector("[data-next-sign]")?.addEventListener("click", () => changeLearningSign(1));
    if (sign.avatar_available) mountLearningAvatar(stage);
  } else if (learningState.stage === "explanation") {
    const sign = currentLearningSign();
    const notes = sign?.movement_notes || "No movement-specific notes are available; the avatar asset is a draft and has not been signer-reviewed.";
    stage.innerHTML = `<h2>Break down the movement</h2><p><strong>${learningEscape(sign?.display_name || "Sign")}:</strong> ${learningEscape(notes)}</p><ol class="lesson-step-list">${lesson.steps.map(step => `<li>${learningEscape(step)}</li>`).join("")}</ol><p>These lesson steps are general study prompts, not verified movement analysis.</p>${sign?.avatar_available ? `<button type="button" class="action-btn secondary-btn" id="reviewSignButton"><i data-lucide="rotate-cw" width="15" height="15"></i> Replay ${learningEscape(sign.display_name)}</button><div class="avatar-mount"></div>` : '<p class="avatar-unavailable">Avatar demonstration unavailable for this sign.</p>'}`;
    document.getElementById("reviewSignButton")?.addEventListener("click", () => playLearningSign(sign));
    if (sign?.avatar_available) mountLearningAvatar(stage);
  } else if (learningState.stage === "practice") {
    renderLearningPractice(stage);
  } else if (learningState.stage === "exercise") {
    const exercise = lesson.exercise;
    stage.innerHTML = `<h2>Mini challenge</h2><p>${learningEscape(exercise.prompt)}</p><div class="exercise-options">${exercise.options.map((option, index) => `<button type="button" data-exercise-answer="${index}">${learningEscape(option.replaceAll("_", " "))}</button>`).join("")}</div><p id="exerciseFeedback" class="exercise-feedback" role="status"></p>`;
    stage.querySelectorAll("[data-exercise-answer]").forEach(button => button.addEventListener("click", () => submitLearningExercise(Number(button.dataset.exerciseAnswer))));
  } else if (learningState.stage === "test") {
    renderLearningQuiz(stage);
  }
  if (typeof lucide !== "undefined") lucide.createIcons();
}

function mountLearningAvatar(stage) {
  const mount = stage.querySelector(".avatar-mount, .quiz-avatar-mount");
  const host = learningState.avatarHost;
  if (!mount || !host) return;
  host.hidden = false;
  mount.appendChild(host);
  const avatar = window.BridgeSignAvatar?.init("learningAvatarHost", {
    globalName: "learningAvatar",
    loaderId: "learningAvatarLoading",
    readyText: "Avatar ready",
    activeNote: "Demonstrating the selected sign.",
  });
  learningState.avatarReady = Boolean(avatar);
}

async function playLearningSign(animation) {
  const sign = animation;
  if (!sign?.avatar_asset?.path) {
    setLearningError("No playable avatar asset is mapped to this sign.");
    return;
  }
  const avatar = window.learningAvatar || window.BridgeSignAvatar?.init("learningAvatarHost", { globalName: "learningAvatar", loaderId: "learningAvatarLoading" });
  if (!avatar) {
    setLearningError("The sign avatar could not be started. The written explanation is still available.");
    return;
  }
  if (typeof avatar.queueAsset !== "function") {
    setLearningError("The current avatar player cannot load the mapped asset type.");
    return;
  }
  try {
    if (avatar._readyPromise) await avatar._readyPromise;
    if (!avatar.queueAsset(sign.avatar_asset.path, sign.name)) {
      setLearningError("The mapped avatar path is not supported by CWASA.");
      return;
    }
    setLearningError(sign.validated ? "" : "Avatar can play, but sign content still requires human review.");
  } catch (error) {
    setLearningError(`Avatar unavailable: ${error.message}`);
  }
}

function resetLearningAvatar() {
  const avatar = window.learningAvatar;
  if (typeof avatar?.stop === "function") {
    avatar.stop();
    setLearningError();
  } else {
    setLearningError("The current avatar player does not expose a reset control.");
  }
}

function changeLearningSign(delta) {
  const nextIndex = learningState.signIndex + delta;
  if (nextIndex < 0 || nextIndex >= (learningState.lesson?.signs.length || 0)) return;
  learningState.signIndex = nextIndex;
  renderLearningStage();
}

function renderLearningPractice(stage) {
  const sign = currentLearningSign();
  if (!sign?.practice_available) {
    stage.innerHTML = `<h2>Practice the sign</h2><p>Use the available resources to review ${learningEscape(sign?.display_name || "this sign")}.</p><p class="practice-note">${sign?.recognition_available ? "Camera evaluation is available for this model label, but not enabled for this lesson." : "Camera evaluation is not supported for this sign by the current model."} No exact hand-position diagnosis is claimed.</p>${sign?.avatar_available ? '<div class="avatar-mount"></div>' : '<p class="avatar-unavailable">Avatar demonstration unavailable.</p>'}`;
    if (sign?.avatar_available) mountLearningAvatar(stage);
    return;
  }
  const motionPractice = sign.practice_mode === "motion";
  const practiceHint = motionPractice
    ? "Keep your hand in frame in good light. Press Check sign, then perform the whole movement during the short recording."
    : "Frame one clear handshape in good light, hold it steady, then request one model prediction.";
  const practiceNote = motionPractice
    ? "This motion sign is checked by BridgeSign's gesture model from a short clip. Confidence feedback is a learning indicator, not a diagnosis of exact hand-position errors."
    : "Letters are checked from a single snapshot. Confidence feedback is a learning indicator, not a diagnosis of exact hand-position errors.";
  stage.innerHTML = `<h2>Practice ${learningEscape(sign.display_name)}</h2><p>${learningEscape(practiceHint)}</p><div class="practice-camera"><video id="learningPracticeVideo" autoplay playsinline muted aria-label="Practice camera preview"></video><div class="practice-actions"><button type="button" class="action-btn start-btn" id="startLearningCamera"><i data-lucide="camera" width="15" height="15"></i> Start camera</button><button type="button" class="action-btn secondary-btn" id="checkLearningSign" disabled><i data-lucide="scan" width="15" height="15"></i> Check sign</button><button type="button" class="action-btn clear-btn" id="stopLearningCamera" disabled><i data-lucide="video-off" width="15" height="15"></i> Stop</button></div><p id="practiceResult" class="practice-result" role="status"></p></div><p class="practice-note">${learningEscape(practiceNote)}</p><div class="avatar-mount"></div>`;
  document.getElementById("startLearningCamera").addEventListener("click", startLearningCamera);
  document.getElementById("checkLearningSign").addEventListener("click", checkLearningSign);
  document.getElementById("stopLearningCamera").addEventListener("click", stopLearningCamera);
  if (sign.avatar_available) mountLearningAvatar(stage);
}

async function startLearningCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setLearningError("Camera access is not available in this browser or page context. Use HTTPS or localhost.");
    return;
  }
  try {
    learningState.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    const video = document.getElementById("learningPracticeVideo");
    video.srcObject = learningState.cameraStream;
    await video.play();
    document.getElementById("checkLearningSign").disabled = false;
    document.getElementById("stopLearningCamera").disabled = false;
    setLearningError();
  } catch (error) {
    setLearningError(error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access to practice." : `Camera unavailable: ${error.message}`);
  }
}

function stopLearningCamera() {
  if (learningState.cameraStream) learningState.cameraStream.getTracks().forEach(track => track.stop());
  learningState.cameraStream = null;
  const video = document.getElementById("learningPracticeVideo");
  if (video) video.srcObject = null;
  const check = document.getElementById("checkLearningSign");
  const stop = document.getElementById("stopLearningCamera");
  if (check) check.disabled = true;
  if (stop) stop.disabled = true;
}

function captureLearningFrame(video, maxDim, quality) {
  const scale = maxDim ? Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight)) : 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
}

// Record ~1.8 s of downscaled frames for motion signs, with a short countdown.
async function captureLearningClip(video, form, frameCount = 18, intervalMs = 100) {
  const resultNode = document.getElementById("practiceResult");
  for (const count of [3, 2, 1]) {
    if (resultNode) resultNode.textContent = `Get ready... ${count}`;
    await new Promise(resolve => setTimeout(resolve, 600));
  }
  if (resultNode) resultNode.textContent = "Recording - perform the sign now";
  for (let index = 0; index < frameCount; index += 1) {
    const blob = await captureLearningFrame(video, 320, 0.7);
    if (blob) form.append("frames", blob, `clip_${index}.jpg`);
    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }
  if (resultNode) resultNode.textContent = "Checking...";
}

async function checkLearningSign() {
  if (learningState.cameraBusy) return;
  const video = document.getElementById("learningPracticeVideo");
  const sign = currentLearningSign();
  if (!video?.videoWidth || !sign) return;
  learningState.cameraBusy = true;
  const button = document.getElementById("checkLearningSign");
  button.disabled = true;
  const form = new FormData();
  if (sign.practice_mode === "motion") {
    await captureLearningClip(video, form);
  } else {
    const blob = await captureLearningFrame(video, 0, 0.82);
    form.append("frame", blob, "practice.jpg");
  }
  form.append("lesson_id", learningState.lesson.id);
  form.append("sign_id", sign.id);
  try {
    const result = await learningRequest("/api/learning/practice", { method: "POST", body: form });
    const confidence = Math.round(result.confidence * 100);
    const resultNode = document.getElementById("practiceResult");
    resultNode.textContent = `${result.feedback} Prediction: ${result.prediction || "No clear sign"}. Confidence: ${confidence}%.`;
    resultNode.dataset.result = result.correct ? "success" : "retry";
  } catch (error) {
    setLearningError(error.message);
  } finally {
    learningState.cameraBusy = false;
    const checkButton = document.getElementById("checkLearningSign");
    if (checkButton && learningState.cameraStream) checkButton.disabled = false;
  }
}

async function submitLearningExercise(answerIndex) {
  const buttons = [...document.querySelectorAll("[data-exercise-answer]")];
  buttons.forEach(button => { button.disabled = true; });
  try {
    const result = await learningRequest(`/api/learning/lessons/${learningState.lesson.id}/exercise`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ answer_index: answerIndex }),
    });
    const feedback = document.getElementById("exerciseFeedback");
    feedback.textContent = `${result.feedback} ${result.explanation}`;
    learningState.lesson.exercise_completed = true;
    learningState.lesson.progress_percentage = Math.max(learningState.lesson.progress_percentage || 0, 80);
    const testButton = document.querySelector('.lesson-stage-nav [data-stage="test"]');
    if (testButton) testButton.disabled = false;
    const nextButton = document.getElementById("nextLessonStage");
    if (nextButton) nextButton.disabled = false;
    buttons[answerIndex]?.setAttribute("aria-pressed", "true");
  } catch (error) {
    buttons.forEach(button => { button.disabled = false; });
    setLearningError(error.message);
  }
}

async function startOrResumeQuiz(sessionId = null) {
  try {
    learningState.quiz = sessionId
      ? await learningRequest(`/api/learning/quiz/${encodeURIComponent(sessionId)}/resume`)
      : await learningRequest(`/api/learning/lessons/${learningState.lesson.id}/quiz/start`, { method: "POST" });
    learningState.answeredIds = new Set(learningState.quiz.answered_ids || []);
    learningState.questionIndex = learningState.quiz.questions.findIndex(question => !learningState.answeredIds.has(question.id));
    if (learningState.questionIndex < 0) learningState.questionIndex = learningState.quiz.questions.length;
    renderLearningStage();
  } catch (error) {
    setLearningError(error.message);
  }
}

function renderLearningQuiz(stage) {
  if (!learningState.quiz) {
    stage.innerHTML = `<h2>Lesson test</h2><p>Check what you understand. You need ${learningState.lesson.passing_score}% to pass; failed tests can be retried.</p><button type="button" class="action-btn start-btn" id="startLessonQuiz"><i data-lucide="clipboard-check" width="15" height="15"></i> ${learningState.lesson.active_quiz_session ? "Resume test" : "Start test"}</button>`;
    document.getElementById("startLessonQuiz").addEventListener("click", () => startOrResumeQuiz(learningState.lesson.active_quiz_session));
    return;
  }
  const questions = learningState.quiz.questions;
  if (learningState.questionIndex >= questions.length) {
    finishLearningQuiz();
    stage.innerHTML = "<p>Saving your result…</p>";
    return;
  }
  const question = questions[learningState.questionIndex];
  stage.innerHTML = `<div class="quiz-header"><h2>Question ${learningState.questionIndex + 1} of ${questions.length}</h2><span>${learningState.quiz.attempt_number ? `Attempt ${learningState.quiz.attempt_number}` : ""}</span></div><p>${learningEscape(question.prompt)}</p>${question.avatar_available ? '<div class="quiz-avatar-mount"></div><button type="button" id="playQuizSign" class="action-btn secondary-btn"><i data-lucide="play" width="15" height="15"></i> Play sign</button>' : ""}${question.type === "sign_recognition" ? '<div class="practice-camera"><video id="learningQuizVideo" autoplay playsinline muted aria-label="Quiz camera preview"></video><div class="practice-actions"><button type="button" class="action-btn start-btn" id="startQuizCamera"><i data-lucide="camera" width="15" height="15"></i> Start camera</button><button type="button" class="action-btn secondary-btn" id="checkQuizSign" disabled><i data-lucide="scan" width="15" height="15"></i> Check sign</button><button type="button" class="action-btn clear-btn" id="stopQuizCamera" disabled><i data-lucide="video-off" width="15" height="15"></i> Stop</button></div></div>' : `<div class="quiz-options">${question.options.map((option, index) => `<button type="button" data-quiz-answer="${index}">${learningEscape(option.label)}</button>`).join("")}</div>`}<p id="quizFeedback" class="quiz-feedback" role="status"></p>`;
  document.getElementById("playQuizSign")?.addEventListener("click", () => {
    learningRequest(`/api/learning/quiz/${encodeURIComponent(learningState.quiz.session_id)}/question/${question.id}/demo`)
      .then(data => data.asset ? playLearningAsset(data.asset) : setLearningError("No playable avatar asset is associated with this question."))
      .catch(error => setLearningError(error.message));
  });
  document.getElementById("startQuizCamera")?.addEventListener("click", startQuizCamera);
  document.getElementById("checkQuizSign")?.addEventListener("click", () => checkLearningQuizSign(question));
  document.getElementById("stopQuizCamera")?.addEventListener("click", stopLearningCamera);
  stage.querySelectorAll("[data-quiz-answer]").forEach(button => button.addEventListener("click", () => submitLearningQuizAnswer(question, Number(button.dataset.quizAnswer))));
  if (question.avatar_available) mountLearningAvatar(stage);
  if (typeof lucide !== "undefined") lucide.createIcons();
}

async function playLearningAsset(asset) {
  if (!asset?.path || !asset?.label) {
    setLearningError("No playable avatar asset is mapped to this question.");
    return;
  }
  const avatar = window.learningAvatar || window.BridgeSignAvatar?.init("learningAvatarHost", { globalName: "learningAvatar", loaderId: "learningAvatarLoading" });
  if (!avatar || typeof avatar.queueAsset !== "function") {
    setLearningError("The current avatar player cannot load this mapped asset type.");
    return;
  }
  try {
    if (avatar._readyPromise) await avatar._readyPromise;
    if (!avatar.queueAsset(asset.path, asset.label)) setLearningError("The mapped avatar path is not supported by CWASA.");
    else setLearningError(asset.validation_status === "VALIDATED" ? "" : "Avatar asset can play; sign content remains unverified.");
  } catch (error) {
    setLearningError(`Avatar unavailable: ${error.message}`);
  }
}

async function submitLearningQuizAnswer(question, optionIndex) {
  const buttons = [...document.querySelectorAll("[data-quiz-answer]")];
  buttons.forEach(button => { button.disabled = true; });
  try {
    const result = await learningRequest(`/api/learning/quiz/${encodeURIComponent(learningState.quiz.session_id)}/answer`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question_id: question.id, option_index: optionIndex }),
    });
    learningState.answeredIds.add(question.id);
    document.getElementById("quizFeedback").textContent = result.feedback;
    learningState.questionIndex += 1;
    window.setTimeout(renderLearningStage, 550);
  } catch (error) {
    buttons.forEach(button => { button.disabled = false; });
    setLearningError(error.message);
  }
}

async function finishLearningQuiz() {
  try {
    const result = await learningRequest(`/api/learning/quiz/${encodeURIComponent(learningState.quiz.session_id)}/complete`, { method: "POST" });
    learningState.lesson.active_quiz_session = null;
    learningState.lesson.status = result.passed ? "completed" : "review_required";
    learningState.lesson.current_section = "result";
    learningState.quiz = null;
    learningState.stage = "result";
    renderLearningResult(result);
  } catch (error) {
    setLearningError(error.message);
  }
}

function renderLearningResult(result) {
  const stage = document.getElementById("learningStage");
  document.querySelector(".lesson-footer").hidden = true;
  document.querySelector(".lesson-stage-nav").hidden = true;
  stage.innerHTML = `<h2>${result.passed ? "Lesson passed" : "Keep practicing"}</h2><div class="quiz-result-score">${result.score}/${result.total} · ${result.percentage}%</div><p>${result.passed ? "The next lesson is now unlocked." : `You need ${result.passing_score}% to pass. Your earlier attempts remain saved; review the lesson and try again.`}</p><div>${result.review.map(item => `<div class="quiz-review-row"><strong>${learningEscape(item.prompt)}</strong><p>${item.correct ? "Correct" : `Your answer: ${learningEscape(item.selected)} · Correct answer: ${learningEscape(item.correct_answer)}`}</p><p>${learningEscape(item.explanation)}</p></div>`).join("")}</div><div class="practice-actions"><button type="button" class="action-btn start-btn" id="resultCourseButton"><i data-lucide="list-checks" width="15" height="15"></i> Return to course</button>${result.passed ? "" : '<button type="button" class="action-btn secondary-btn" id="retryLessonQuiz">Try test again</button>'}</div>`;
  document.getElementById("resultCourseButton").addEventListener("click", closeLearningLesson);
  document.getElementById("retryLessonQuiz")?.addEventListener("click", () => {
    learningState.stage = "test";
    learningState.quiz = null;
    renderLearningLesson();
  });
  if (typeof lucide !== "undefined") lucide.createIcons();
}

async function startQuizCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setLearningError("Camera access is unavailable. Use HTTPS or localhost to allow camera practice.");
    return;
  }
  try {
    learningState.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    const video = document.getElementById("learningQuizVideo");
    video.srcObject = learningState.cameraStream;
    await video.play();
    document.getElementById("checkQuizSign").disabled = false;
    document.getElementById("stopQuizCamera").disabled = false;
    setLearningError();
  } catch (error) {
    setLearningError(error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access to continue." : `Camera unavailable: ${error.message}`);
  }
}

async function checkLearningQuizSign(question) {
  if (learningState.cameraBusy) return;
  const video = document.getElementById("learningQuizVideo");
  if (!video?.videoWidth) return;
  learningState.cameraBusy = true;
  const button = document.getElementById("checkQuizSign");
  button.disabled = true;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.82));
  const form = new FormData();
  form.append("frame", blob, "quiz-practice.jpg");
  try {
    const result = await learningRequest(`/api/learning/quiz/${encodeURIComponent(learningState.quiz.session_id)}/question/${question.id}/recognition`, { method: "POST", body: form });
    document.getElementById("quizFeedback").textContent = `${result.feedback} Prediction: ${result.prediction}; confidence ${Math.round(result.confidence * 100)}%.`;
    learningState.answeredIds.add(question.id);
    learningState.questionIndex += 1;
    stopLearningCamera();
    window.setTimeout(renderLearningStage, 900);
  } catch (error) {
    setLearningError(error.message);
    if (learningState.cameraStream) button.disabled = false;
  } finally {
    learningState.cameraBusy = false;
  }
}