const CONF_THRESHOLD = 0.3;

function imageToTensor(img, size = MODEL_SIZE) {
  let off = document.createElement("canvas");
  off.width = off.height = size;
  const ctxOff = off.getContext("2d");
  ctxOff.fillStyle = "black";
  ctxOff.fillRect(0, 0, size, size);

  const scale = Math.min(size / img.width, size / img.height);
  const newW = img.width * scale;
  const newH = img.height * scale;
  const dx = (size - newW) / 2;
  const dy = (size - newH) / 2;
  ctxOff.drawImage(img, dx, dy, newW, newH);

  const imgData = ctxOff.getImageData(0, 0, size, size).data;
  const data = new Float32Array(3 * size * size);
  let r = 0,
    g = size * size,
    b = 2 * size * size;
  for (let i = 0; i < imgData.length; i += 4) {
    data[r++] = imgData[i] / 255;
    data[g++] = imgData[i + 1] / 255;
    data[b++] = imgData[i + 2] / 255;
  }
  off.width = 0;
  off.height = 0;
  off = null; // Explicitly release the canvas reference
  return {
    tensor: new ort.Tensor("float32", data, [1, 3, size, size]),
    scale,
    dx,
    dy,
  };
}

function mapBoxes(boxes, scale, dx, dy, imgW, imgH) {
  const results = [];
  for (let i = 0; i < boxes.length / 6; i++) {
    const o = i * 6;
    let x1 = boxes[o],
      y1 = boxes[o + 1],
      x2 = boxes[o + 2],
      y2 = boxes[o + 3];
    const score = boxes[o + 4];
    if (!score || score < CONF_THRESHOLD) continue;

    x1 = Math.max(0, (x1 - dx) / scale);
    y1 = Math.max(0, (y1 - dy) / scale);
    x2 = Math.min(imgW, (x2 - dx) / scale);
    y2 = Math.min(imgH, (y2 - dy) / scale);
    results.push({ x1, y1, x2, y2, score });
  }
  return results;
}

export { imageToTensor, mapBoxes };
