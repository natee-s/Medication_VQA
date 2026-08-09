const OUTPUT_WIDTH = 1344;
const OUTPUT_HEIGHT = 1000;
const JPEG_QUALITY = 0.9;
const PDPA_MASK_RATIO = 0.25;
const CAMERA_IDEAL_WIDTH = 1920;
const CAMERA_IDEAL_HEIGHT = 1440;
const CAMERA_IDEAL_ASPECT_RATIO = 4 / 3;
const CAMERA_DEVICE_STORAGE_KEY = "medication_label_camera_device_id";
const CAMERA_PREFERENCE_STORAGE_KEY = "medication_label_camera_preference";

const video = document.getElementById("cameraPreview");
const canvas = document.getElementById("captureCanvas");
const guideFrame = document.getElementById("guideFrame");
const switchCameraButton = document.getElementById("switchCameraButton");
const captureButton = document.getElementById("captureButton");
const retakeButton = document.getElementById("retakeButton");
const uploadButton = document.getElementById("uploadButton");
const previewPanel = document.getElementById("previewPanel");
const capturedPreview = document.getElementById("capturedPreview");
const statusText = document.getElementById("statusText");
const processingOverlay = document.getElementById("processingOverlay");
const processingText = document.getElementById("processingText");
const cameraShell = document.querySelector(".camera-shell");

const FALLBACK_MESSAGES = {
  document_title: "Medication Label Camera",
  processing: "กำลังประมวลผล...",
  processing_captured: "ถ่ายรูปเสร็จแล้ว กำลังประมวลผล...",
  guide_header: "ส่วนหัวฉลาก",
  guide_body: "ชื่อยาและวิธีใช้",
  title: "ถ่ายฉลากยา",
  subtitle: "วางฉลากให้อยู่ในกรอบ และให้เส้นคั่นบนฉลากตรงกับเส้นกลางกรอบ",
  preview_instruction: "ตรวจรูปก่อนส่ง ถ้าไม่ชัดให้กดถ่ายใหม่",
  preview_alt: "รูปฉลากยาที่ถ่ายแล้ว",
  switch_camera_button: "สลับกล้อง",
  capture_button: "ถ่ายรูป",
  retake_button: "ถ่ายใหม่",
  upload_button: "ส่งรูป",
  status_camera_unsupported: "อุปกรณ์นี้ไม่รองรับการเปิดกล้องผ่านเว็บ",
  status_align_label: "จัดฉลากให้อยู่ในกรอบ แล้วกดถ่ายรูป",
  status_switching_camera: "กำลังสลับกล้อง...",
  status_camera_denied: "เปิดกล้องไม่ได้ กรุณาอนุญาตสิทธิ์กล้องแล้วลองใหม่",
  status_camera_not_ready: "กล้องยังไม่พร้อม กรุณารอสักครู่",
  status_create_failed: "สร้างรูปไม่สำเร็จ กรุณาถ่ายใหม่",
  status_no_image: "ยังไม่มีรูป กรุณาถ่ายรูปก่อน",
  status_upload_success: "ส่งรูปสำเร็จ กลับไปที่แชท LINE เพื่อรอผลลัพธ์",
  status_upload_unlinked: "ระบบได้รับรูปแล้ว แต่ยังไม่ได้เชื่อมกับบัญชี LINE",
  status_upload_failed: "ส่งรูปไม่สำเร็จ กรุณาลองใหม่",
};

let capturedBlob = null;
let stream = null;
let lineUserId = "";
let isCapturing = false;
let isSwitchingCamera = false;
let availableCameraDevices = [];
let activeCameraDeviceId = "";
let activeCameraPreference = null;
let uiMessages = { ...FALLBACK_MESSAGES };
let currentStatusKey = "";
const cameraDebugMode = new URLSearchParams(window.location.search).has("debug");

function t(key) {
  return uiMessages[key] || FALLBACK_MESSAGES[key] || key;
}

function setStatus(message) {
  currentStatusKey = "";
  statusText.textContent = message;
}

function setStatusKey(key) {
  currentStatusKey = key;
  statusText.textContent = t(key);
}

function setProcessingMode(enabled, message = t("processing")) {
  processingText.textContent = message;
  processingOverlay.hidden = !enabled;
}

function applyTranslations(messages, language = "th") {
  uiMessages = { ...FALLBACK_MESSAGES, ...(messages || {}) };
  document.documentElement.lang = language;
  document.title = t("document_title");

  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-alt]").forEach((element) => {
    element.setAttribute("alt", t(element.dataset.i18nAlt));
  });

  if (currentStatusKey) {
    statusText.textContent = t(currentStatusKey);
  }
}

async function loadLiffMessages() {
  const params = lineUserId ? `?line_user_id=${encodeURIComponent(lineUserId)}` : "";
  try {
    const response = await fetch(`/liff/messages${params}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Messages failed: ${response.status}`);
    }
    const result = await response.json();
    applyTranslations(result.messages, result.language);
  } catch (error) {
    console.warn("LIFF messages fallback to Thai", error);
    applyTranslations(FALLBACK_MESSAGES, "th");
  }
}

function setPreviewMode(enabled) {
  cameraShell.classList.toggle("preview-mode", enabled);
  previewPanel.hidden = !enabled;
  updateSwitchCameraVisibility();
  captureButton.hidden = enabled;
  retakeButton.hidden = !enabled;
  uploadButton.hidden = !enabled;
  if (enabled) {
    retakeButton.disabled = false;
    uploadButton.disabled = false;
  } else {
    captureButton.disabled = false;
  }
}

function closeLiffWindowSoon() {
  setTimeout(() => {
    if (window.liff?.isInClient?.()) {
      window.liff.closeWindow();
      return;
    }

    window.close();
  }, 900);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatusKey("status_camera_unsupported");
    captureButton.disabled = true;
    return;
  }

  captureButton.disabled = true;

  try {
    stream = await requestRearCameraStream();
    await activateCameraStream(stream, { saveDevice: false });
  } catch (error) {
    console.error(error);
    stopCameraStream();
    setStatusKey("status_camera_denied");
    captureButton.disabled = true;
  }
}

async function requestRearCameraStream() {
  const savedStream = await requestSavedCameraStream();
  if (savedStream) {
    return savedStream;
  }

  return requestCameraStreamWithFallbacks();
}

async function requestSavedCameraStream() {
  const savedPreference = getSavedCameraPreference();
  if (!savedPreference) {
    return null;
  }

  try {
    return await requestCameraStreamByPreference(savedPreference);
  } catch (error) {
    if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
      throw error;
    }

    clearSavedCameraPreference();
    console.warn("Saved camera unavailable; falling back to automatic selection", error);
    return null;
  }
}

function preferredVideoConstraints(extra = {}) {
  return {
    width: { ideal: CAMERA_IDEAL_WIDTH },
    height: { ideal: CAMERA_IDEAL_HEIGHT },
    aspectRatio: { ideal: CAMERA_IDEAL_ASPECT_RATIO },
    resizeMode: "none",
    ...extra,
  };
}

async function requestCameraStreamWithFallbacks() {
  const attempts = [
    {
      video: preferredVideoConstraints({
        facingMode: { ideal: "environment" },
      }),
      audio: false,
    },
    {
      video: {
        facingMode: { ideal: "environment" },
      },
      audio: false,
    },
    {
      video: true,
      audio: false,
    },
  ];

  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
        throw error;
      }
      console.warn("Camera constraint attempt failed", constraints, error);
    }
  }

  throw lastError || new Error("Camera stream unavailable");
}

async function requestCameraStreamByDeviceId(deviceId) {
  const attempts = [
    {
      video: preferredVideoConstraints({
        deviceId: { exact: deviceId },
      }),
      audio: false,
    },
    {
      video: {
        deviceId: { exact: deviceId },
      },
      audio: false,
    },
    {
      video: preferredVideoConstraints({
        deviceId: { ideal: deviceId },
      }),
      audio: false,
    },
    {
      video: {
        deviceId: { ideal: deviceId },
      },
      audio: false,
    },
  ];

  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
        throw error;
      }
    }
  }

  throw lastError || new Error("Camera device unavailable");
}

async function requestCameraStreamByPreference(preference) {
  if (preference?.deviceId) {
    try {
      return await requestCameraStreamByDeviceId(preference.deviceId);
    } catch (error) {
      if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
        throw error;
      }
      console.warn("Saved deviceId unavailable; trying saved camera metadata", error);
    }
  }

  const matchedDevice = await findPreferredCameraDevice(preference);
  if (matchedDevice?.deviceId) {
    return requestCameraStreamByDeviceId(matchedDevice.deviceId);
  }

  throw new Error("Saved camera preference could not be matched");
}

async function maybeSwitchToBetterRearCamera(initialStream) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return initialStream;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((device) => device.kind === "videoinput" && device.deviceId);
    availableCameraDevices = getCameraSwitchCandidates(videoInputs);
    if (cameraDebugMode) {
      console.info("LIFF available video inputs", videoInputs.map((device) => ({
        label: device.label || "unknown",
        deviceId: device.deviceId,
        score: scoreCameraLabel(device.label),
      })));
    }

    if (videoInputs.length < 2 || videoInputs.every((device) => !device.label)) {
      return initialStream;
    }

    const [currentTrack] = initialStream.getVideoTracks();
    const currentLabel = currentTrack?.label || "";
    const currentScore = scoreCameraLabel(currentLabel);
    const candidates = videoInputs
      .filter((device) => !isBadRearCameraLabel(device.label))
      .sort((a, b) => scoreCameraLabel(b.label) - scoreCameraLabel(a.label));
    const bestCandidate = candidates[0];

    if (!bestCandidate || scoreCameraLabel(bestCandidate.label) <= currentScore) {
      return initialStream;
    }

    const replacementStream = await navigator.mediaDevices.getUserMedia({
      video: preferredVideoConstraints({
        deviceId: { exact: bestCandidate.deviceId },
      }),
      audio: false,
    });

    stopSpecificStream(initialStream);
    return replacementStream;
  } catch (error) {
    if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
      throw error;
    }

    console.warn("Rear camera selection fallback kept initial stream", error);
    return initialStream;
  }
}

async function activateCameraStream(nextStream, { saveDevice = false, preferredDevice = null } = {}) {
  await optimizeCameraTrack(nextStream);
  video.srcObject = nextStream;
  await video.play().catch(() => {});

  const [track] = nextStream.getVideoTracks();
  activeCameraDeviceId = track?.getSettings?.().deviceId || "";
  await refreshCameraDevices();
  activeCameraPreference = buildCameraPreference(track, preferredDevice || findCameraDeviceById(activeCameraDeviceId));
  if (saveDevice && activeCameraPreference?.deviceId) {
    saveCameraPreference(activeCameraPreference);
  }
  updateSwitchCameraVisibility();
  logCameraInfo("LIFF camera ready", track);

  captureButton.disabled = false;
  if (cameraDebugMode) {
    setStatus(formatCameraDebugInfo(track));
  } else {
    setStatusKey("status_align_label");
  }
}

async function refreshCameraDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    availableCameraDevices = [];
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((device) => device.kind === "videoinput" && device.deviceId);
    availableCameraDevices = getCameraSwitchCandidates(videoInputs);
  } catch (error) {
    console.warn("Could not refresh camera list", error);
    availableCameraDevices = [];
  }
}

function getCameraSwitchCandidates(videoInputs) {
  return videoInputs.filter((device) => {
    if (!device.deviceId) {
      return false;
    }
    return !isFrontCameraLabel(device.label);
  });
}

function updateSwitchCameraVisibility() {
  if (!switchCameraButton) {
    return;
  }

  const hasAlternatives = availableCameraDevices.length > 1;
  switchCameraButton.hidden = !hasAlternatives || !previewPanel.hidden;
  switchCameraButton.disabled = isSwitchingCamera || isCapturing;
}

async function switchCamera() {
  if (isSwitchingCamera || availableCameraDevices.length < 2) {
    return;
  }

  isSwitchingCamera = true;
  updateSwitchCameraVisibility();
  captureButton.disabled = true;
  setStatusKey("status_switching_camera");

  try {
    await refreshCameraDevices();
    const currentIndex = getActiveCameraIndex();
    let replacementStream = null;
    let selectedDevice = null;

    for (let offset = 1; offset <= availableCameraDevices.length; offset += 1) {
      const candidate = availableCameraDevices[(currentIndex + offset) % availableCameraDevices.length];
      if (!candidate?.deviceId || candidate.deviceId === activeCameraDeviceId) {
        continue;
      }

      try {
        replacementStream = await requestCameraStreamByDeviceId(candidate.deviceId);
        selectedDevice = candidate;
        break;
      } catch (error) {
        console.warn("Camera switch candidate failed", candidate.label || candidate.deviceId, error);
      }
    }

    if (!replacementStream) {
      throw new Error("No switchable camera could be opened");
    }

    stopCameraStream();
    stream = replacementStream;
    activeCameraPreference = buildCameraPreference(stream.getVideoTracks()[0], selectedDevice);
    await activateCameraStream(stream, { saveDevice: true, preferredDevice: selectedDevice });
  } catch (error) {
    console.error(error);
    setStatusKey("status_camera_denied");
    captureButton.disabled = false;
  } finally {
    isSwitchingCamera = false;
    updateSwitchCameraVisibility();
  }
}

function getActiveCameraIndex() {
  const byDeviceId = availableCameraDevices.findIndex((device) => device.deviceId === activeCameraDeviceId);
  if (byDeviceId >= 0) {
    return byDeviceId;
  }

  const byPreference = findCameraIndexByPreference(activeCameraPreference);
  if (byPreference >= 0) {
    return byPreference;
  }

  return 0;
}

function scoreCameraLabel(label = "") {
  const normalized = label.toLowerCase();
  let score = 0;

  if (/(back|rear|environment|หลัง)/.test(normalized)) score += 40;
  if (/(wide|main|1x|standard|normal)/.test(normalized)) score += 18;
  if (/(front|user|หน้า|selfie)/.test(normalized)) score -= 80;
  if (/(tele|telephoto|zoom|portrait|depth|macro)/.test(normalized)) score -= 60;
  if (/(ultra|0\.5x)/.test(normalized)) score -= 10;

  return score;
}

function isFrontCameraLabel(label = "") {
  const normalized = label.toLowerCase();
  return /(front|user|หน้า|selfie)/.test(normalized);
}

function isBadRearCameraLabel(label = "") {
  const normalized = label.toLowerCase();
  return /(front|user|หน้า|selfie|tele|telephoto|zoom|portrait|depth|macro)/.test(normalized);
}

function getSavedCameraDeviceId() {
  try {
    return window.localStorage?.getItem(CAMERA_DEVICE_STORAGE_KEY) || "";
  } catch (error) {
    return "";
  }
}

function getSavedCameraPreference() {
  try {
    const rawPreference = window.localStorage?.getItem(CAMERA_PREFERENCE_STORAGE_KEY);
    if (rawPreference) {
      const parsed = JSON.parse(rawPreference);
      if (parsed && typeof parsed === "object") {
        return parsed;
      }
    }
  } catch (error) {
    console.warn("Could not read saved camera preference", error);
  }

  return null;
}

function saveCameraPreference(preference) {
  if (!preference?.deviceId) {
    return;
  }

  try {
    window.localStorage?.setItem(CAMERA_PREFERENCE_STORAGE_KEY, JSON.stringify(preference));
    window.localStorage?.setItem(CAMERA_DEVICE_STORAGE_KEY, preference.deviceId);
  } catch (error) {
    console.warn("Could not save camera device preference", error);
  }
}

function clearSavedCameraPreference() {
  try {
    window.localStorage?.removeItem(CAMERA_PREFERENCE_STORAGE_KEY);
    window.localStorage?.removeItem(CAMERA_DEVICE_STORAGE_KEY);
  } catch (error) {}
}

function buildCameraPreference(track, device = null) {
  const settings = track?.getSettings?.() || {};
  const label = track?.label || device?.label || "";
  const deviceId = settings.deviceId || device?.deviceId || "";
  return {
    deviceId,
    groupId: device?.groupId || "",
    label,
    labelKey: normalizeCameraLabel(label),
    index: availableCameraDevices.findIndex((item) => item.deviceId === deviceId),
    width: settings.width || 0,
    height: settings.height || 0,
    zoom: settings.zoom ?? null,
    savedAt: Date.now(),
  };
}

function normalizeCameraLabel(label = "") {
  return String(label)
    .toLowerCase()
    .replace(/\([^)]*\)/g, "")
    .replace(/[^a-z0-9ก-๙]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function findCameraDeviceById(deviceId) {
  if (!deviceId) {
    return null;
  }
  return availableCameraDevices.find((device) => device.deviceId === deviceId) || null;
}

async function findPreferredCameraDevice(preference) {
  if (!navigator.mediaDevices?.enumerateDevices || !preference) {
    return null;
  }

  await refreshCameraDevices();
  const matchIndex = findCameraIndexByPreference(preference);
  return matchIndex >= 0 ? availableCameraDevices[matchIndex] : null;
}

function findCameraIndexByPreference(preference) {
  if (!preference || !availableCameraDevices.length) {
    return -1;
  }

  if (preference.deviceId) {
    const byDeviceId = availableCameraDevices.findIndex((device) => device.deviceId === preference.deviceId);
    if (byDeviceId >= 0) {
      return byDeviceId;
    }
  }

  const preferredLabel = normalizeCameraLabel(preference.label || preference.labelKey || "");
  if (preference.groupId && preferredLabel) {
    const byGroupAndLabel = availableCameraDevices.findIndex((device) => (
      device.groupId === preference.groupId &&
      normalizeCameraLabel(device.label) === preferredLabel
    ));
    if (byGroupAndLabel >= 0) {
      return byGroupAndLabel;
    }
  }

  if (preferredLabel) {
    const byExactLabel = availableCameraDevices.findIndex((device) => normalizeCameraLabel(device.label) === preferredLabel);
    if (byExactLabel >= 0) {
      return byExactLabel;
    }

    const bySimilarLabel = availableCameraDevices.findIndex((device) => {
      const label = normalizeCameraLabel(device.label);
      return label && (label.includes(preferredLabel) || preferredLabel.includes(label));
    });
    if (bySimilarLabel >= 0) {
      return bySimilarLabel;
    }
  }

  const savedIndex = Number(preference.index);
  if (Number.isInteger(savedIndex) && availableCameraDevices[savedIndex]) {
    return savedIndex;
  }

  return -1;
}

async function optimizeCameraTrack(mediaStream) {
  const [track] = mediaStream?.getVideoTracks?.() || [];
  if (!track?.getCapabilities || !track?.getSettings || !track?.applyConstraints) {
    return;
  }

  await normalizeCameraZoom(track);
  await normalizeCameraFocus(track);
}

async function normalizeCameraZoom(track) {
  if (!track?.getCapabilities || !track?.getSettings || !track?.applyConstraints) {
    return;
  }

  const capabilities = track.getCapabilities();
  const settings = track.getSettings();
  const currentZoom = Number(settings.zoom);
  const minZoom = Number(capabilities.zoom?.min);
  const maxZoom = Number(capabilities.zoom?.max);

  if (
    !Number.isFinite(minZoom) ||
    !Number.isFinite(maxZoom) ||
    minZoom > maxZoom
  ) {
    return;
  }

  const targetZoom = Math.max(minZoom, Math.min(1, maxZoom));
  if (Number.isFinite(currentZoom) && Math.abs(currentZoom - targetZoom) < 0.01) {
    return;
  }

  try {
    await track.applyConstraints({ advanced: [{ zoom: targetZoom }] });
  } catch (error) {
    console.warn("Camera 1x normalization skipped", error);
  }
}

async function normalizeCameraFocus(track) {
  if (!track?.getCapabilities || !track?.applyConstraints) {
    return;
  }

  const capabilities = track.getCapabilities();
  const advanced = [];
  if (Array.isArray(capabilities.focusMode) && capabilities.focusMode.includes("continuous")) {
    advanced.push({ focusMode: "continuous" });
  }
  if (Array.isArray(capabilities.exposureMode) && capabilities.exposureMode.includes("continuous")) {
    advanced.push({ exposureMode: "continuous" });
  }
  if (Array.isArray(capabilities.whiteBalanceMode) && capabilities.whiteBalanceMode.includes("continuous")) {
    advanced.push({ whiteBalanceMode: "continuous" });
  }

  if (!advanced.length) {
    return;
  }

  try {
    await track.applyConstraints({ advanced });
  } catch (error) {
    console.warn("Camera focus/exposure normalization skipped", error);
  }
}

function stopSpecificStream(mediaStream) {
  mediaStream?.getTracks?.().forEach((track) => track.stop());
}

function stopCameraStream() {
  if (stream) {
    stopSpecificStream(stream);
    stream = null;
  }
  video.srcObject = null;
}

function logCameraInfo(label, track) {
  const settings = track?.getSettings?.() || {};
  const capabilities = track?.getCapabilities?.() || {};
  console.info(label, {
    label: track?.label || "unknown",
    facingMode: settings.facingMode || "unknown",
    width: settings.width || 0,
    height: settings.height || 0,
    aspectRatio: settings.aspectRatio || "not-reported",
    zoom: settings.zoom ?? "not-reported",
    zoomMin: capabilities.zoom?.min ?? "not-reported",
    zoomMax: capabilities.zoom?.max ?? "not-reported",
    focusMode: settings.focusMode || "not-reported",
  });
}

function formatCameraDebugInfo(track) {
  const settings = track?.getSettings?.() || {};
  return [
    track?.label || "camera",
    `${settings.width || 0}x${settings.height || 0}`,
    `zoom:${settings.zoom ?? "n/a"}`,
  ].join(" | ");
}

async function initializeLiff() {
  if (!window.liff) {
    return;
  }

  try {
    const response = await fetch("/liff/config", { cache: "no-store" });
    const config = await response.json();
    if (!config.liff_id) {
      return;
    }

    await window.liff.init({ liffId: config.liff_id });
    if (!window.liff.isLoggedIn()) {
      window.liff.login();
      return;
    }

    const profile = await window.liff.getProfile();
    lineUserId = profile.userId || "";
  } catch (error) {
    console.warn("LIFF initialization skipped", error);
  }
}

function getGuideSourceRect() {
  const videoRect = video.getBoundingClientRect();
  const guideRect = guideFrame.getBoundingClientRect();
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  const objectFit = getComputedStyle(video).objectFit || "contain";

  const scale =
    objectFit === "cover"
      ? Math.max(videoRect.width / videoWidth, videoRect.height / videoHeight)
      : Math.min(videoRect.width / videoWidth, videoRect.height / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;
  const offsetX =
    objectFit === "cover"
      ? (renderedWidth - videoRect.width) / 2
      : (videoRect.width - renderedWidth) / 2;
  const offsetY =
    objectFit === "cover"
      ? (renderedHeight - videoRect.height) / 2
      : (videoRect.height - renderedHeight) / 2;

  const sourceX =
    objectFit === "cover"
      ? (guideRect.left - videoRect.left + offsetX) / scale
      : (guideRect.left - videoRect.left - offsetX) / scale;
  const sourceY =
    objectFit === "cover"
      ? (guideRect.top - videoRect.top + offsetY) / scale
      : (guideRect.top - videoRect.top - offsetY) / scale;
  const sourceWidth = guideRect.width / scale;
  const sourceHeight = guideRect.height / scale;

  return {
    x: Math.max(0, Math.min(sourceX, videoWidth - 1)),
    y: Math.max(0, Math.min(sourceY, videoHeight - 1)),
    width: Math.min(sourceWidth, videoWidth - Math.max(0, sourceX)),
    height: Math.min(sourceHeight, videoHeight - Math.max(0, sourceY)),
  };
}

function applyGuidelinePdpaMask(context) {
  const maskHeight = Math.round(OUTPUT_HEIGHT * PDPA_MASK_RATIO);
  context.save();
  context.fillStyle = "#000000";
  context.fillRect(0, 0, OUTPUT_WIDTH, maskHeight);
  context.restore();
}

function canvasToJpegBlob() {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY);
  });
}

async function captureGuideFrame() {
  if (isCapturing) {
    return;
  }

  if (!video.videoWidth || !video.videoHeight) {
    setStatusKey("status_camera_not_ready");
    return;
  }

  isCapturing = true;
  capturedBlob = null;
  captureButton.disabled = true;
  updateSwitchCameraVisibility();
  setStatus("");
  setProcessingMode(true, t("processing_captured"));

  try {
    const source = getGuideSourceRect();
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(
      video,
      source.x,
      source.y,
      source.width,
      source.height,
      0,
      0,
      OUTPUT_WIDTH,
      OUTPUT_HEIGHT,
    );

    const previewBlob = await canvasToJpegBlob();
    if (!previewBlob) {
      throw new Error("Could not create preview image");
    }

    if (capturedPreview.src) {
      URL.revokeObjectURL(capturedPreview.src);
    }
    capturedPreview.src = URL.createObjectURL(previewBlob);
    setPreviewMode(true);
    retakeButton.disabled = true;
    uploadButton.disabled = true;

    applyGuidelinePdpaMask(context);
    const maskedBlob = await canvasToJpegBlob();
    if (!maskedBlob) {
      throw new Error("Could not create masked image");
    }

    capturedBlob = maskedBlob;
    retakeButton.disabled = false;
    uploadButton.disabled = false;
    setStatus("");
  } catch (error) {
    console.warn("Capture failed", error);
    capturedBlob = null;
    if (capturedPreview.src) {
      URL.revokeObjectURL(capturedPreview.src);
    }
    capturedPreview.removeAttribute("src");
    setPreviewMode(false);
    setStatusKey("status_create_failed");
  } finally {
    isCapturing = false;
    captureButton.disabled = false;
    updateSwitchCameraVisibility();
    setProcessingMode(false);
  }
}

function retake() {
  capturedBlob = null;
  isCapturing = false;
  setProcessingMode(false);
  if (capturedPreview.src) {
    URL.revokeObjectURL(capturedPreview.src);
  }
  capturedPreview.removeAttribute("src");
  retakeButton.disabled = false;
  uploadButton.disabled = false;
  captureButton.disabled = false;
  updateSwitchCameraVisibility();
  setPreviewMode(false);
  setStatusKey("status_align_label");
}

async function uploadCapture() {
  if (!capturedBlob) {
    setStatusKey("status_no_image");
    return;
  }

  uploadButton.disabled = true;
  retakeButton.disabled = true;
  setStatus("");
  setProcessingMode(true);

  try {
    const response = await fetch("/liff/upload-label", {
      method: "POST",
      headers: {
        "content-type": "image/jpeg",
        "x-line-user-id": lineUserId,
      },
      body: capturedBlob,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    const result = await response.json();
    if (result.processing_queued) {
      setStatusKey("status_upload_success");
      closeLiffWindowSoon();
    } else {
      setStatusKey("status_upload_unlinked");
    }
    setProcessingMode(false);
    retakeButton.disabled = false;
  } catch (error) {
    console.error(error);
    setProcessingMode(false);
    setStatusKey("status_upload_failed");
    uploadButton.disabled = false;
    retakeButton.disabled = false;
  }
}

captureButton.addEventListener("click", captureGuideFrame);
switchCameraButton?.addEventListener("click", switchCamera);
retakeButton.addEventListener("click", retake);
uploadButton.addEventListener("click", uploadCapture);

async function bootstrap() {
  await initializeLiff();
  await loadLiffMessages();
  await startCamera();
}

window.addEventListener("pagehide", stopCameraStream);

bootstrap();
