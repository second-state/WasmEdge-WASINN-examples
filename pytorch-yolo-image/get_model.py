# requires ultralytics to be installed
from ultralytics import YOLO

# Load a model, saving it to disk
# Note there are different net sizes, n, m, l, x
# Larger models are more accurate, but slower
model = YOLO('yolov8n.pt')

# Export the model as a torchscript file
model.export(format='torchscript')
