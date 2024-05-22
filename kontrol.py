from ultralytics import YOLO

# Load a model
model = YOLO('runs/pose/train2/weights/best.pt')  # load a tiger-pose trained model

# Run inference
results = model.predict(source="VID-20240304-WA0008_online-video-cutter.com.mp4" ,show=True)