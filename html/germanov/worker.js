// importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js");

// WASM file has to be in the webserver root.  WTF?

// which WASM files may be needed
// 
let busy = false;
onmessage = async(event) => {
    if (busy) {
        return
    }
    busy = true;
    const input = event.data;
    const output = await run_model(input);
    postMessage(output);
    busy = false;
}

async function run_model(input) {
    const model = await ort.InferenceSession.create("./yolov8n.onnx");
    input = new ort.Tensor(Float32Array.from(input),[1, 3, 640, 640]);
    const outputs = await model.run({images:input});
    return outputs["output0"].data;
}
