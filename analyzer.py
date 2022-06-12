import cv2
import numpy as np

from tracker import SpeedDetector

MODEL_INPUT_SIZE = 320
MODEL_CONFIDENCE_THRESHOLD = 0.2
MODEL_NMS_THRESHOLD = 0.2

classes_filename = "coco.names"
with open(classes_filename) as f:
    class_names = f.read().strip().split('\n')

# required classes
required_class_index = [2, 3, 5, 7]


model_configuration = 'yolov3-320.cfg'
model_weights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')


def post_process(outputs, img, speed_detector):
    height, width, _ = img.shape
    boxes = []
    class_ids = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in required_class_index:
                if confidence > MODEL_CONFIDENCE_THRESHOLD:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w/2), int((det[1] * height) - h/2)
                    boxes.append([x, y, w, h])
                    class_ids.append(int(class_id))
                    confidence_scores.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, MODEL_CONFIDENCE_THRESHOLD, MODEL_NMS_THRESHOLD)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_id = class_ids[i]

        color = colors[class_id].tolist()
        name = class_names[class_id]
        cv2.putText(
            img,
            f'{name.upper()} {int(confidence_scores[i]*100)}%',
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(class_id)])

    speed_detector.update(detection, img)


def real_time_processing(cap, show_output: bool, speed_limit: int, zone_length: int):

    original_w = int(cap.get(3))
    original_h = int(cap.get(4))
    original_fps = int(cap.get(5))
    print(original_w, original_h, original_fps)

    up_line_position = 0.50
    down_line_position = 0.83

    speed_detector = SpeedDetector(
        start_line_y=int(original_h * up_line_position),
        end_line_y=int(original_h * down_line_position),
        zone_length_m=zone_length,
        speed_limit=speed_limit,
        fps=original_fps,
    )

    while True:
        _, img = cap.read()
        # print(img.shape)  # 720 1080 3
        # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        # print(img.shape)
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        layers_names = net.getLayerNames()
        output_names = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_names)

        post_process(outputs, img, speed_detector)

        ih, iw, _ = img.shape
        cv2.line(img, (0, int(ih*up_line_position)), (iw, int(ih*up_line_position)), (0, 0, 255), 2)
        cv2.line(img, (0, int(ih*down_line_position)), (iw, int(ih*down_line_position)), (0, 0, 255), 2)

        if show_output:
            cv2.imshow('Output', img)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def process_video(source: str, show_output: bool, speed_limit: int, zone_length: int):
    cap = cv2.VideoCapture(source)

    real_time_processing(cap, show_output, speed_limit, zone_length)
