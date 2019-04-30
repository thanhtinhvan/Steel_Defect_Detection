#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np

# ap = argparse.ArgumentParser()
# ap.add_argument('--image', default="dog.jpg",
#                 help = 'path to input image')
# ap.add_argument('--config', default="yolo3.cfg",
#                 help = 'path to yolo config file')
# ap.add_argument('--weights', default="yolov3.weights",
#                 help = 'path to yolo pre-trained weights')
# ap.add_argument('--classes', default="yolov3.txt",
#                 help = 'path to text file containing class names')
# args = ap.parse_args()
imageName="a.jpg"
backgrnd = "backg2.jpg"
config = "yolov3.cfg"
weights = "yolov3.weights"
classesTXT = "yolov3.txt"
videoOut = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (4032,3024), True)
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(imageName)
background = cv2.imread(backgrnd)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(classesTXT, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
new_img = image

videoOut.write(new_img)
trans = 0
while True:
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0]-100)
        y = round(box[1]-100)
        w = round(box[2]+200)
        h = round(box[3]+200)
        label = str(classes[class_ids[i]])
        print(label)
        if label == "book" or label == "bottle":
            tmp = image[y:y+h, x:x+w]
            backgroundTMP = background[y:y+h, x:x+w]
            new_item = cv2.addWeighted(tmp, 1-trans, backgroundTMP, trans, 0)
            
            new_img[y:y+h, x:x+w] = new_item
    trans = trans + 0.01
    print(trans)
    videoOut.write(new_img)
    # cv2.imwrite("disa.jpg", new_img)
    if trans>=1:
        break
        # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
# cv2.imshow("object detection", image)
# cv2.waitKey()
videoOut.release()
cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()
