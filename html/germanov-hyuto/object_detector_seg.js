

const video = document.getElementById("cam");


//duplicated in worker
const inputShape = [1, 3, 640, 640];
const [modelWidth, modelHeight] = inputShape.slice(2);
const maxSize = Math.max(modelWidth, modelHeight); // max size in input model

const worker = new Worker("worker.js");   // Should make this a module worker? 

let maskCanvas, videoCanvas, offscreenCanvas, maskContext, videoContext, offscreenContext
addEventListener("load", (event) => {
	videoCanvas = document.getElementById("videoCanvas");
	videoContext = videoCanvas.getContext("2d", {willReadFrequently:true});
	maskCanvas = document.getElementById("maskCanvas");
	maskContext = maskCanvas.getContext("2d",{willReadFrequently:true});
	offscreenCanvas = document.getElementById("offscreenCanvas");
	offscreenContext = offscreenCanvas.getContext("2d",{willReadFrequently:true});
	worker.postMessage( {
		type: "init", 
		data: ""
		} );
}); 
let boxes = [];
let interval

    
video.addEventListener("play", () => {
    videoCanvas.width = 640;
    videoCanvas.height = 640;
    maskCanvas.width = videoCanvas.width;
    maskCanvas.height = videoCanvas.height;
    offscreenCanvas.width = videoCanvas.width;
    offscreenCanvas.height = videoCanvas.height; 
    
    interval = setInterval(() => {
        videoContext.drawImage(video,0,0);
        //context.clearRect(0, 0, context.canvas.width, context.canvas.height); // clean canvas

        //draw_boxes(canvas, boxes);
        //const input = prepare_input(canvas);    
        //[input, xRatio, yRatio]
        
        const inputArgs = preprocessing(
        	videoContext.getImageData(0,0,videoContext.canvas.width,videoContext.canvas.height).data, 
        	videoContext.canvas.width, videoContext.canvas.height, 
        	modelWidth, modelHeight); // preprocess frame
        //console.log("Call worker");
        worker.postMessage({type:"image", data: {inputArgs: inputArgs} });
    },750) /// WHY... 
});

// Set up a listener for messages from the worker
// const canvas = document.getElementById("canvas"); 
// const ctx = canvas.getContext("2d");

let drawbusy = false; 
worker.onmessage = function(e) {
	switch(e.data.type) { 
    case "message":
    	console.log('worker message:', e.data.data);
    	break;
    case "mask":
    	//console.log('got worker mask', e.data.data);
    	if (e.data.data.idx == 0) {
    	        offscreenContext.clearRect(0, 0, offscreenContext.canvas.width, offscreenContext.canvas.height); // clean canvas
		}
    	if (typeof e.data.data !== 'undefined') {
    		renderMask(e.data.data.mask, offscreenContext);
    		};
    	break;
    case "boxes":
    	//console.log('got worker boxes', e.data.data);
    	if (typeof e.data.data !== 'undefined') {
    		renderBoxes(offscreenContext, e.data.data);
    	
    		maskContext.putImageData(offscreenContext.getImageData(0,0,offscreenContext.canvas.width, offscreenContext.canvas.height), 0, 0,); 
    		};
    	break;
    default:
    	console.log('unknown worker message', e.data); 
    	break; 
    }
}
// Set up an error listener for the worker
worker.onerror = function(e) {
    console.error(e)
}


// 
// worker.onmessage = (event) => {
//     const output = event.data;
//     const canvas = document.querySelector("canvas");
//     boxes =  process_output(output, canvas.width, canvas.height);
// };

// video.addEventListener("pause", () => {
//     clearInterval(interval);
// });

// const playBtn = document.getElementById("play");
// const pauseBtn = document.getElementById("pause");
// playBtn.addEventListener("click", () => {
//     video.play();
// });
// pauseBtn.addEventListener("click", () => {
//     video.pause();
// });



// HYUTO STUFF


/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Number} stride model stride
 * @return preprocessed image and configs
 */
const preprocessing = (imagedata, width, height, modelWidth, modelHeight, stride = 32) => {
  //const mat = cv.imread(source); // read from img tag
  
  //console.log(width,height);
  let mat = new cv.Mat(height, width, cv.CV_8UC4);
  mat.data.set(imagedata);
  
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input32F = new Float32Array (
  	cv.blobFromImage(
		matPad,
		1 / 255.0, // normalize
		new cv.Size(modelWidth, modelHeight), // resize to model input size
		new cv.Scalar(0, 0, 0),
		true, // swapRB
		false // crop
	  ).data32F // preprocessing image matrix
 	); 
  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input32F, xRatio, yRatio];
};


/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};


 const renderBoxes = (ctx, boxes) => {
  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = box.label;
    const color = box.color;
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(klass + " - " + score + "%", x1 - 1, yText < 0 ? 1 : yText + 1);
  });
};


const renderMask = async(overlay, ctx) => {

  


  const mask_img = new ImageData(new Uint8ClampedArray(overlay.data), modelHeight, modelWidth); // create image data from mask overlay
  ctx.putImageData(mask_img, 0, 0); // put overlay to canvas

  //renderBoxes(ctx, boxes); // draw boxes after overlay added to canvas
  
}; 



// --------------------

// GERMANOV  STUFF
//
function prepare_input(img) {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 640;
    const context = canvas.getContext("2d");
    context.drawImage(img, 0, 0, 640, 640);
    const data = context.getImageData(0,0,640,640).data;
    const red = [], green = [], blue = [];
    for (let index=0;index<data.length;index+=4) {
        red.push(data[index]/255);
        green.push(data[index+1]/255);
        blue.push(data[index+2]/255);
    }
    return [...red, ...green, ...blue];
}

function process_output(output, img_width, img_height) {
    let boxes = [];
    for (let index=0;index<8400;index++) {
        const [class_id,prob] = [...Array(yolo_classes.length).keys()]
            .map(col => [col, output[8400*(col+4)+index]])
            .reduce((accum, item) => item[1]>accum[1] ? item : accum,[0,0]);
        if (prob < 0.5) {
            continue;
        }
        const label = yolo_classes[class_id];
        const xc = output[index];
        const yc = output[8400+index];
        const w = output[2*8400+index];
        const h = output[3*8400+index];
        const x1 = (xc-w/2)/640*img_width;
        const y1 = (yc-h/2)/640*img_height;
        const x2 = (xc+w/2)/640*img_width;
        const y2 = (yc+h/2)/640*img_height;
        boxes.push([x1,y1,x2,y2,label,prob]);
    }
    boxes = boxes.sort((box1,box2) => box2[5]-box1[5])
    const result = [];
    while (boxes.length>0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0],box)<0.7 || boxes[0][4] !== box[4]);
    }
    return result;
}

function iou(box1,box2) {
    return intersection(box1,box2)/union(box1,box2);
}

function union(box1,box2) {
    const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
    const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
    const box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    const box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)
}

function intersection(box1,box2) {
    const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
    const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
    const x1 = Math.max(box1_x1,box2_x1);
    const y1 = Math.max(box1_y1,box2_y1);
    const x2 = Math.min(box1_x2,box2_x2);
    const y2 = Math.min(box1_y2,box2_y2);
    return (x2-x1)*(y2-y1)
}

function draw_boxes(canvas,boxes) {
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    ctx.font = "18px serif";
    boxes.forEach(([x1,y1,x2,y2,label]) => {
        ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        ctx.fillStyle = "#00ff00";
        const width = ctx.measureText(label).width;
        ctx.fillRect(x1,y1,width+10,25);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x1, y1+18);
    });
}

const yolo_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];
