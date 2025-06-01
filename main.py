from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: list[float]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2

# Đọc mô hình từ file JSON
try:
    with open('nn_info.json', 'r') as f:
        model_data = json.load(f)
    # Chọn bộ trọng số đầu tiên
    W1 = np.array(model_data['weights'][0], dtype=np.float32)  # [784, 128]
    W2 = np.array(model_data['weights'][1], dtype=np.float32)  # [128, 10]
    b1 = np.array(model_data['biases'][0], dtype=np.float32).reshape(1, -1)
    b2 = np.array(model_data['biases'][1], dtype=np.float32).reshape(1, -1)
    
    # Kiểm tra kích thước
    assert W1.shape == (784, 128), f"W1 shape {W1.shape} != (784, 128)"
    assert b1.shape == (1, 128), f"b1 shape {b1.shape} != (1, 128)"
    assert W2.shape == (128, 10), f"W2 shape {W2.shape} != (128, 10)"
    assert b2.shape == (1, 10), f"b2 shape {b2.shape} != (1, 10)"
except FileNotFoundError:
    raise Exception("Không tìm thấy tệp nn_info.json")
except KeyError as e:
    raise Exception(f"Thiếu khóa trong nn_info.json: {str(e)}")
except AssertionError as e:
    raise Exception(f"Kích thước trọng số không hợp lệ: {str(e)}")

@app.post("/predict")
async def predict(data: ImageData):
    try:
        image = np.array(data.image, dtype=np.float32)
        if len(image) != 784:
            raise ValueError(f"Kích thước đầu vào không hợp lệ: {len(image)}, cần 784")
        # image = image.reshape(1, -1)
        prediction = forward(image, W1, b1, W2, b2)
        digit = int(np.argmax(prediction, axis=1)[0])
        return {"digit": digit, "probabilities": prediction[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.get("/weights")
async def get_weights():
    try:
        # Lấy các trọng số cần thiết
        # W1: Lấy 5 hàng đầu và 5 hàng cuối (input layer), cột 0-2 và 125-127 (hidden layer)
        w1_subset = np.concatenate((W1[:5, :], W1[-5:, :]), axis=0)  # (10, 128)
        w1_subset = np.concatenate((w1_subset[:, :3], w1_subset[:, -3:]), axis=1)  # (10, 6)
        # W2: Lấy hàng 0-2 và 125-127 (hidden layer), tất cả cột (output layer)
        w2_subset = np.concatenate((W2[:3, :], W2[-3:, :]), axis=0)  # (6, 10)
        return {
            "W1": w1_subset.tolist(),
            "W2": w2_subset.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy trọng số: {str(e)}")