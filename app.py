from flask import Flask, request, jsonify
import torch
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Tải mô hình YOLOv5 đã huấn luyện (best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_model/best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    img = Image.open(image_file.stream)

    # Dự đoán với YOLOv5
    results = model(img)
    predictions = results.pred[0].cpu().numpy()

    # Lấy kết quả đầu tiên (nếu có)
    prediction_text = "Healthy"  # Default prediction
    bounding_box = None

    if len(predictions) > 0:
        prediction_text = "Disease"
        x1, y1, x2, y2 = predictions[0][:4]
        bounding_box = {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}

    return jsonify({
        "prediction": prediction_text,
        "x": bounding_box['x'] if bounding_box else None,
        "y": bounding_box['y'] if bounding_box else None,
        "width": bounding_box['width'] if bounding_box else None,
        "height": bounding_box['height'] if bounding_box else None
    })

if __name__ == '__main__':
    app.run(debug=True)
