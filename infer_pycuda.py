import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time

class BaseEngine(object):
    def __init__(self, engine_path, imgsz=(640,640)):
        self.imgsz = imgsz
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})        

    def predict(self, img,threshold):
        self.img = self.preprocess(img)
        self.inputs[0]['host'] = np.ravel(self.img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        results = self.postprocess(data,threshold)
        return results

    def letterbox(self,im,color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img,self.r,self.dw,self.dh

    def preprocess(self,image):
        self.img,self.r,self.dw,self.dh = self.letterbox(image)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.img = self.img.astype(np.float32)
        self.img = self.img / 255.
        self.img -= np.array([0.485, 0.456, 0.406])[None, None, :]
        self.img /= np.array([0.229, 0.224, 0.225])[None, None, :]
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img,0)
        return self.img

    def postprocess(self,pred,threshold):
        new_bboxes = []
        num =int(pred[0][0])
        bboxes = pred[1].reshape(-1,4)
        scores = pred[2]
        clas = pred[3]
        for i in range(num):
            if(scores[i] < threshold):
                continue
            xmin = (bboxes[i][0] - self.dw)/self.r
            ymin = (bboxes[i][1] - self.dh)/self.r
            xmax = (bboxes[i][2] - self.dw)/self.r
            ymax = (bboxes[i][3] - self.dh)/self.r
            new_bboxes.append([clas[i],scores[i],xmin,ymin,xmax,ymax])
        return new_bboxes


def visualize(img,bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), (105, 237, 249), 2)
        img = cv2.putText(img, "class:"+str(clas)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img

Predictor = BaseEngine("./trt_model/ppyoloex_nms_fp16.engine")
img1 = cv2.imread("./pictures/bus.jpg")
results = Predictor.predict(img1,threshold=0.5)
img = visualize(img1,results)
cv2.imshow("img",img)
cv2.waitKey(0)
