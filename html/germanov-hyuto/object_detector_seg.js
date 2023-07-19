const video = document.getElementById("cam")

//duplicated in worker
const inputShape = [1, 3, 640, 640]
const [modelWidth, modelHeight] = inputShape.slice(2)
const maxSize = Math.max(modelWidth, modelHeight) // max size in input model

const worker = new Worker("worker.js") // Should make this a module worker?

let maskCanvas,
    videoCanvas,
    offscreenCanvas,
    maskContext,
    videoContext,
    offscreenContext
    
addEventListener("load", (event) => {
    videoCanvas = document.getElementById("videoCanvas")
    videoContext = videoCanvas.getContext("2d", { willReadFrequently: true })
    maskCanvas = document.getElementById("maskCanvas")
    maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })
    offscreenCanvas = document.getElementById("offscreenCanvas")
    offscreenContext = offscreenCanvas.getContext("2d", {
        willReadFrequently: true,
    })
    worker.postMessage({
        type: "init",
        data: "",
    })
})
let boxes = []
let interval

video.addEventListener("play", () => {
    videoCanvas.width = 640
    videoCanvas.height = 640
    maskCanvas.width = videoCanvas.width
    maskCanvas.height = videoCanvas.height
    offscreenCanvas.width = videoCanvas.width
    offscreenCanvas.height = videoCanvas.height

    interval = setInterval(() => {
        videoContext.drawImage(video, 0, 0)
        const inputArgs = preprocessing(
            videoContext.getImageData(
                0,
                0,
                videoContext.canvas.width,
                videoContext.canvas.height,
            ).data,
            videoContext.canvas.width,
            videoContext.canvas.height,
            modelWidth,
            modelHeight,
        ) // preprocess frame
        worker.postMessage({ type: "image", data: { inputArgs: inputArgs } })
    }, 750) /// WHY...
})

worker.onmessage = function (e) {
    switch (e.data.type) {
        case "message":
            console.log("worker message:", e.data.data)
            break
        case "mask":
            //console.log('got worker mask', e.data.data);
            if (e.data.data.idx == 0) {
                offscreenContext.clearRect(
                    0,
                    0,
                    offscreenContext.canvas.width,
                    offscreenContext.canvas.height,
                ) // clean canvas
            }
            if (typeof e.data.data !== "undefined") {
                renderMask(e.data.data.mask, offscreenContext)
            }
            break
        case "boxes":
            //console.log('got worker boxes', e.data.data);
            if (typeof e.data.data !== "undefined") {
                renderBoxes(offscreenContext, e.data.data)

                maskContext.putImageData(
                    offscreenContext.getImageData(
                        0,
                        0,
                        offscreenContext.canvas.width,
                        offscreenContext.canvas.height,
                    ),
                    0,
                    0,
                )
            }
            break
        default:
            console.log("unknown worker message", e.data)
            break
    }
}

// Set up an error listener for the worker
worker.onerror = function (e) {
    console.error(e)
}

// HYUTO STUFF

const preprocessing = (
    imagedata,
    width,
    height,
    modelWidth,
    modelHeight,
    stride = 32,
) => {
    let mat = new cv.Mat(height, width, cv.CV_8UC4)
    mat.data.set(imagedata)

    const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3) // new image matrix
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR) // RGBA to BGR

    const [w, h] = divStride(stride, matC3.cols, matC3.rows)
    cv.resize(matC3, matC3, new cv.Size(w, h))

    // padding image to [n x n] dim
    const maxSize = Math.max(matC3.rows, matC3.cols) // get max size from width and height
    const xPad = maxSize - matC3.cols, // set xPadding
        xRatio = maxSize / matC3.cols // set xRatio
    const yPad = maxSize - matC3.rows, // set yPadding
        yRatio = maxSize / matC3.rows // set yRatio
    const matPad = new cv.Mat() // new mat for padded image
    cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT) // padding black

    const input32F = new Float32Array(
        cv.blobFromImage(
            matPad,
            1 / 255.0, // normalize
            new cv.Size(modelWidth, modelHeight), // resize to model input size
            new cv.Scalar(0, 0, 0),
            true, // swapRB
            false, // crop
        ).data32F, // preprocessing image matrix
    )
    // release mat opencv
    mat.delete()
    matC3.delete()
    matPad.delete()

    return [input32F, xRatio, yRatio]
}

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
    if (width % stride !== 0) {
        if (width % stride >= stride / 2)
            width = (Math.floor(width / stride) + 1) * stride
        else width = Math.floor(width / stride) * stride
    }
    if (height % stride !== 0) {
        if (height % stride >= stride / 2)
            height = (Math.floor(height / stride) + 1) * stride
        else height = Math.floor(height / stride) * stride
    }
    return [width, height]
}

const renderBoxes = (ctx, boxes) => {
    // font configs
    const font = `${Math.max(
        Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
        14,
    )}px Arial`
    ctx.font = font
    ctx.textBaseline = "top"

    boxes.forEach((box) => {
        const klass = box.label
        const color = box.color
        const score = (box.probability * 100).toFixed(1)
        const [x1, y1, width, height] = box.bounding

        // draw border box
        ctx.strokeStyle = color
        ctx.lineWidth = Math.max(
            Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
            2.5,
        )
        ctx.strokeRect(x1, y1, width, height)

        // draw the label background.
        ctx.fillStyle = color
        const textWidth = ctx.measureText(klass + " - " + score + "%").width
        const textHeight = parseInt(font, 10) // base 10
        const yText = y1 - (textHeight + ctx.lineWidth)
        ctx.fillRect(
            x1 - 1,
            yText < 0 ? 0 : yText,
            textWidth + ctx.lineWidth,
            textHeight + ctx.lineWidth,
        )

        // Draw labels
        ctx.fillStyle = "#ffffff"
        ctx.fillText(
            klass + " - " + score + "%",
            x1 - 1,
            yText < 0 ? 1 : yText + 1,
        )
    })
}

const renderMask = async (overlay, ctx) => {
    const mask_img = new ImageData(
        new Uint8ClampedArray(overlay.data),
        modelHeight,
        modelWidth,
    ) // create image data from mask overlay
    ctx.putImageData(mask_img, 0, 0) // put overlay to canvas
}
