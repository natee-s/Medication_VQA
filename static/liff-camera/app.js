const OUTPUT_WIDTH = 1344;
const OUTPUT_HEIGHT = 1000;
const JPEG_QUALITY = 0.9;

const video = document.getElementById("cameraPreview");
const canvas = document.getElementById("captureCanvas");
const guideFrame = document.getElementById("guideFrame");
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
  guide_header: "ส่วนหัวฉลาก",
  guide_body: "ชื่อยาและวิธีใช้",
  title: "ถ่ายฉลากยา",
  subtitle: "วางฉลากให้อยู่ในกรอบ และให้เส้นคั่นบนฉลากตรงกับเส้นกลางกรอบ",
  preview_instruction: "ตรวจรูปก่อนส่ง ถ้าไม่ชัดให้กดถ่ายใหม่",
  preview_alt: "รูปฉลากยาที่ถ่ายแล้ว",
  capture_button: "ถ่ายรูป",
  retake_button: "ถ่ายใหม่",
  upload_button: "ส่งรูป",
  status_camera_unsupported: "อุปกรณ์นี้ไม่รองรับการเปิดกล้องผ่านเว็บ",
  status_align_label: "จัดฉลากให้อยู่ในกรอบ แล้วกดถ่ายรูป",
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
let uiMessages = { ...FALLBACK_MESSAGES };
let currentStatusKey = "";

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

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    video.srcObject = stream;
    setStatusKey("status_align_label");
  } catch (error) {
    console.error(error);
    setStatusKey("status_camera_denied");
    captureButton.disabled = true;
  }
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

  const scale = Math.max(videoRect.width / videoWidth, videoRect.height / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;
  const offsetX = (renderedWidth - videoRect.width) / 2;
  const offsetY = (renderedHeight - videoRect.height) / 2;

  const sourceX = (guideRect.left - videoRect.left + offsetX) / scale;
  const sourceY = (guideRect.top - videoRect.top + offsetY) / scale;
  const sourceWidth = guideRect.width / scale;
  const sourceHeight = guideRect.height / scale;

  return {
    x: Math.max(0, Math.min(sourceX, videoWidth - 1)),
    y: Math.max(0, Math.min(sourceY, videoHeight - 1)),
    width: Math.min(sourceWidth, videoWidth - Math.max(0, sourceX)),
    height: Math.min(sourceHeight, videoHeight - Math.max(0, sourceY)),
  };
}

function captureGuideFrame() {
  if (isCapturing) {
    return;
  }

  if (!video.videoWidth || !video.videoHeight) {
    setStatusKey("status_camera_not_ready");
    return;
  }

  isCapturing = true;
  captureButton.disabled = true;
  setStatus("");
  setProcessingMode(true);

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

  canvas.toBlob(
    (blob) => {
      isCapturing = false;
      captureButton.disabled = false;
      setProcessingMode(false);

      if (!blob) {
        setStatusKey("status_create_failed");
        return;
      }

      capturedBlob = blob;
      capturedPreview.src = URL.createObjectURL(blob);
      setPreviewMode(true);
      setStatus("");
    },
    "image/jpeg",
    JPEG_QUALITY,
  );
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
retakeButton.addEventListener("click", retake);
uploadButton.addEventListener("click", uploadCapture);

async function bootstrap() {
  await initializeLiff();
  await loadLiffMessages();
  startCamera();
}

bootstrap();
