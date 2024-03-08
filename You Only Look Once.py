import cv2

def detect_objects_yolo(image_path, config_path, weights_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    return outputs, classes

# Example usage:
image_path = "image.jpg"
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
classes_path = "coco.names"
objects, classes = detect_objects_yolo(image_path, config_path, weights_path, classes_path)
print("Detected objects:", objects)
print("Object classes:", classes)
