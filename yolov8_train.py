from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8l.yaml")  # build a new model from YAML
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':


    model.train(data="my-polyps.yaml", wokers=1, epochs=200, batch=16)