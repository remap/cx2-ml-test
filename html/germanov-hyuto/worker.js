//importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"); 
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js");

importScripts("opencv.js"); // TODO: Pass from script tag import? 
importScripts("download.js");
importScripts("renderBox.js");
const labels = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

// configs
const modelName = "yolov8n-seg.onnx";
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.25;
  
// Worker main
console.log("Worker starts");


let busy = false;
let init = false; 
let session;
let imageRef, canvasRef;

self.onmessage = async(event) => {
  switch (event.data.type) {
  case "init":
	session = await workerInit();
	console.log("session", session);
  	init = true; 
  	self.postMessage({type:"message", data:"workerInit complete"});
  	break;
  case "message":
  	console.log(`worker message ${event.data.data}`);
    break;
  case "image":
    if (!init || busy) break; 
    busy = true;
    //console.log(`worker image`, session);
    const inputArgs = event.data.data.inputArgs;
	const selected = 
		await detectImage(
		  inputArgs,
		  //output,
		  session,
		  topk,
		  iouThreshold,
		  scoreThreshold,
		  modelInputShape);
	//self.postMessage({type: "output", data:selected}) 
	busy = false;  
    break;
  default:
    console.log(`worker message unknown type ${event.data}`);
    break
  } 
}; 

async function workerInit() {
	  // wait until opencv.js initialized
	  let session; 
	  
	  return new Promise( async(resolve,reject) => {
	  	try{ 
		  cv["onRuntimeInitialized"] = async () => {

			const baseModelURL = `model`;

			// create session
			const arrBufNet = await download( `${baseModelURL}/${modelName}`, console.log(`${modelName}`));
			const yolov8 = await ort.InferenceSession.create(arrBufNet);
		
			const arrBufNMS = await download( `${baseModelURL}/nms-yolov8.onnx`, console.log(`nms-yolov8.onnx`));
			const nms = await ort.InferenceSession.create(arrBufNMS);
		
			const arrBufMask = await download( `${baseModelURL}/mask-yolov8-seg.onnx`, console.log(`mask-yolov8-seg.onnx`));
			const mask = await ort.InferenceSession.create(arrBufMask);

			// warmup main model
			console.log("warming up model");
			const tensor = new ort.Tensor(
			  "float32",
			  new Float32Array(modelInputShape.reduce((a, b) => a * b)),
			  modelInputShape
			);
			await yolov8.run({ images: tensor });
			session = { net: yolov8, nms:nms, mask: mask } 
			resolve(session); 
		};
		} catch (error) {
			reject(error);
		};
	});
	return session; 
};



//   
//           onLoad={() => {
//             detectImage(
//               imageRef.current,
//               canvasRef.current,
//               session,
//               topk,
//               iouThreshold,
//               scoreThreshold,
//               modelInputShape
//             );
//           }}
//         />
//         <canvas
//           id="canvas"
//           width={modelInputShape[2]}
//           height={modelInputShape[3]}
//           ref={canvasRef}
//         />
//       </div>
// 



// Hyoto Detection


/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
 const detectImage = async (
  inputArgs,
//  output,
//   numClass, 
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {


 const colors = new Colors();
const numClass = labels.length;


  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight); // max size in input model

 // move since we can't mess with the dom
 //
 // const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight); // preprocess frame
  const [input32F, xRatio, yRatio] = inputArgs; 


  //const tensor = new ort.Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const tensor = new ort.Tensor("float32", input32F, inputShape); // to ort.Tensor
  const config = new ort.Tensor(
    "float32",
    new Float32Array([
      numClass, // num class
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor }); // run session and get output layer. out1: detect layer, out2: seg layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes


	
  const boxes = []; // ready to draw boxes
  let overlay = new ort.Tensor("uint8", new Uint8Array(modelHeight * modelWidth * 4), [
    modelHeight,
    modelWidth,
    4,
  ]); // create overlay to draw segmentation object

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    let box = data.slice(0, 4); // det boxes
    const scores = data.slice(4, 4 + numClass); // det classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    const color = colors.get(label); // get color

    box = overflowBoxes(
      [
        box[0] - 0.5 * box[2], // before upscale x
        box[1] - 0.5 * box[3], // before upscale y
        box[2], // before upscale w
        box[3], // before upscale h
      ],
      maxSize
    ); // keep boxes in maxSize range

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio), // upscale left
        Math.floor(box[1] * yRatio), // upscale top
        Math.floor(box[2] * xRatio), // upscale width
        Math.floor(box[3] * yRatio), // upscale height
      ],
      maxSize
    ); // upscale boxes

    boxes.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later

    const mask = new ort.Tensor(
      "float32",
      new Float32Array([
        ...box, // original scale box
        ...data.slice(4 + numClass), // mask data
      ])
    ); // mask input
    const maskConfig = new ort.Tensor(
      "float32",
      new Float32Array([
        maxSize,
        x, // upscale x
        y, // upscale y
        w, // upscale width
        h, // upscale height
        ...Colors.hexToRgba(color, 120), // color in RGBA
      ])
    ); // mask config
    const { mask_filter } = await session.mask.run({
      detection: mask,
      mask: output1,
      config: maskConfig,
      overlay: overlay,
    }); // perform post-process to get mask

    overlay = mask_filter; // update overlay with the new one
   // console.log("posting mask", idx); 
    self.postMessage({type:"mask", data:{mask:overlay, idx:idx}});
    
  }
self.postMessage({type:"boxes", data:boxes});


//  input.delete(); // delete unused Mat
};








// // Here's the original Germanov code
// 
// let busy = false;
// self.onmessage = async(event) => {
// 
// 
// // 
// //     if (busy) {
// //         return
// //     }
// //     busy = true;
// //     const input = event.data;
// //     const output = await run_model(input);
// //     postMessage(output);
// //     busy = false;
// }
// 
// async function run_model(input) {
//     const model = await ort.InferenceSession.create("./yolov8n.onnx");
//     input = new ort.Tensor(Float32Array.from(input),[1, 3, 640, 640]);
//     const outputs = await model.run({images:input});
//     return outputs["output0"].data;
// }
