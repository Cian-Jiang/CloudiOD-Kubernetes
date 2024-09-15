import base64
import json
import numpy as np
import os
import time
from flask import Flask, request, jsonify
import cv2
from functools import lru_cache

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1
yolo_path = "yolo_tiny_configs"


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath=os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

# Use the cache decorator to ensure that the model is only loaded once 使用缓存装饰器确保仅加载一次模型
@lru_cache(maxsize=1)
def get_model():
    nets = load_model(CFG, Weights)
    return nets

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image,net,LABELS):

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # TODO Prepare the output as required to the assignment specification
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            print("detected item:{}, accuracy:{}, X:{}, Y:{}, width:{}, height:{}".format(LABELS[classIDs[i]],
                                                                                             confidences[i],
                                                                                             boxes[i][0],
                                                                                             boxes[i][1],
                                                                                             boxes[i][2],
                                                                                             boxes[i][3]))

    detected_objects = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            obj_data = {
                "label": LABELS[classIDs[i]],
                "accuracy": confidences[i],
                "rectangle": {
                    "height": boxes[i][3],
                    "left": boxes[i][0],
                    "top": boxes[i][1],
                    "width": boxes[i][2],
                },
            }
            detected_objects.append(obj_data)

    return detected_objects

# ## argument
# if len(sys.argv) != 3:
#     raise ValueError("Argument list is wrong. Please use the following format:  {} {} {}".
#                      format("python iWebLens_server.py", "<yolo_config_folder>", "<Image file path>"))
#
# yolo_path  = str(sys.argv[1])

## Yolov3-tiny versrion
labelsPath= "coco.names"
cfgpath= "yolov3-tiny.cfg"
wpath= "yolov3-tiny.weights"

Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)


# #TODO, you should  make this console script into webservice using Flask
# def main():
#     try:
#         imagefile = str(sys.argv[2])
#         img = cv2.imread(imagefile)
#         npimg=np.array(img)
#         image=npimg.copy()
#         image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         # load the neural net.  Should be local to this method as its multi-threaded endpoint
#         nets = load_model(CFG, Weights)
#         do_prediction(image, nets, Lables)
#
#
#     except Exception as e:
#
#         print("Exception  {}".format(e))
#
# if __name__ == '__main__':
#     main()


# Define a new POST request handler 定义新的POST请求处理函数
def detect_objects():
#    data = request.get_json(force=True)
    data = json.loads(request.get_json(force=True))
    image_id = data['id']
    image_data = data['image']
    image_base64_decoded = base64.b64decode(image_data)

    image = np.frombuffer(image_base64_decoded, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Get the model from the cache instead of reloading it every time 从缓存获取模型，而不是每次都重新加载
    nets = get_model()

    #nets = load_model(CFG, Weights)

    detected_objects = do_prediction(image, nets, Lables)

    response_data = {"id": image_id, "objects": detected_objects}
    return jsonify(response_data)



app = Flask(__name__)

# Associate the new POST request handler with a route 关联新的POST请求处理函数与一个路由
app.add_url_rule('/api/object_detection', 'detect_objects', detect_objects, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", threaded=True, port=5000)
