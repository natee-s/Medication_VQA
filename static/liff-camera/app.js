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

let capturedBlob = null;
let stream = null;
let lineUserId = "";
let isCapturing = false;

function setStatus(message) {
  statusText.textContent = message;
}

function setPreviewMode(enabled) {
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

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("อุปกรณ์นี้ไม่รองรับการเปิดกล้องผ่านเว็บ");
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
    setStatus("จัดฉลากให้อยู่ในกรอบ แล้วกดถ่ายรูป");
  } catch (error) {
    console.error(error);
    setStatus("เปิดกล้องไม่ได้ กรุณาอนุญาตสิทธิ์กล้องแล้วลองใหม่");
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
    setStatus("กล้องยังไม่พร้อม กรุณารอสักครู่");
    return;
  }

  isCapturing = true;
  captureButton.disabled = true;
  setStatus("กำลังสร้างรูป...");

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

      if (!blob) {
        setStatus("สร้างรูปไม่สำเร็จ กรุณาถ่ายใหม่");
        return;
      }

      capturedBlob = blob;
      capturedPreview.src = URL.createObjectURL(blob);
      setPreviewMode(true);
      setStatus("ตรวจรูปก่อนส่ง ถ้าไม่ชัดให้กดถ่ายใหม่");
    },
    "image/jpeg",
    JPEG_QUALITY,
  );
}

function retake() {
  capturedBlob = null;
  isCapturing = false;
  if (capturedPreview.src) {
    URL.revokeObjectURL(capturedPreview.src);
  }
  capturedPreview.removeAttribute("src");
  retakeButton.disabled = false;
  uploadButton.disabled = false;
  captureButton.disabled = false;
  setPreviewMode(false);
  setStatus("จัดฉลากให้อยู่ในกรอบ แล้วกดถ่ายรูป");
}

async function uploadCapture() {
  if (!capturedBlob) {
    setStatus("ยังไม่มีรูป กรุณาถ่ายรูปก่อน");
    return;
  }

  uploadButton.disabled = true;
  retakeButton.disabled = true;
  setStatus("กำลังส่งรูป...");

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

    setStatus("ระบบได้รับรูปแล้ว หากต้องการถ่ายใหม่ยังกดถ่ายใหม่ได้");
    retakeButton.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("ส่งรูปไม่สำเร็จ กรุณาลองใหม่");
    uploadButton.disabled = false;
    retakeButton.disabled = false;
  }
}

captureButton.addEventListener("click", captureGuideFrame);
retakeButton.addEventListener("click", retake);
uploadButton.addEventListener("click", uploadCapture);

initializeLiff();
startCamera();
