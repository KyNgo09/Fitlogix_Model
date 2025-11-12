import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Khởi tạo ứng dụng Flask
app = Flask(__name__)

# 2. Tải mô hình đã được huấn luyện (từ Bước 1)
# Đảm bảo tệp 'workout_model.joblib' ở cùng thư mục
try:
    model_pipeline = joblib.load('workout_model.joblib')
    print("Tải mô hình thành công!")
except FileNotFoundError:
    print("LỖI: Không tìm thấy tệp 'workout_model.joblib'.")
    model_pipeline = None

# 3. Định nghĩa "cửa" (endpoint) cho việc dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Mô hình chưa được tải."}), 500

    try:
        # 4. Lấy dữ liệu JSON từ app mobile gửi lên
        data = request.json
        print(f"Nhận được dữ liệu: {data}")

        # 5. Chuyển đổi JSON thành DataFrame (quan trọng)
        # Dữ liệu JSON là một object, cần bọc nó trong list []
        # để DataFrame hiểu đây là 1 hàng duy nhất
        input_df = pd.DataFrame([data])
        
        # Đảm bảo các cột số được chuyển đúng kiểu (vì JSON có thể gửi số dưới dạng text)
        # Lấy danh sách cột số từ pipeline (nếu cần, nhưng RF thường khá linh hoạt)
        # Tạm thời tin tưởng dữ liệu đầu vào là đúng kiểu

        # 6. Thực hiện dự đoán
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        # 7. Lấy tên các nhãn
        classes = model_pipeline.classes_
        # Tạo dict xác suất
        proba_dict = dict(zip(classes, prediction_proba[0]))

        # 8. Trả kết quả về cho app mobile
        return jsonify({
            'du_doan': prediction[0],
            'chi_tiet_xac_suat': proba_dict
        })

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return jsonify({"error": str(e)}), 400

# 9. Chạy server
if __name__ == '__main__':
    # Chạy ở chế độ debug để dễ sửa lỗi
    # Khi "lên sóng", bạn sẽ dùng một server thật như Gunicorn
    app.run(debug=True, use_reloader=False, port=5000)