export async function initCamera(onFrame) {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 320, height: 320, facingMode: "environment" },
    audio: false,
  });

  video.srcObject = stream;

  await video.play();

  canvas.width = 320;
  canvas.height = 320;

  function loop() {
    if (!video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, 320, 320);
      onFrame(ctx, canvas);
      requestAnimationFrame(loop);
    }
  }

  requestAnimationFrame(loop);
}
