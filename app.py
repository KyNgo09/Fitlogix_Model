import joblib
import pandas as pd
from flask import Flask, request, jsonify
from recommender_system import WorkoutRecommender

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã được huấn luyện (từ Bước 1)
try:
    model_pipeline = joblib.load('workout_model.joblib')
    print("Tải mô hình thành công!")
except FileNotFoundError:
    print("LỖI: Không tìm thấy tệp 'workout_model.joblib'.")
    model_pipeline = None

# Load Hệ thống gợi ý (Chỉ load 1 lần khi server khởi động)
try:
    recommender_engine = WorkoutRecommender('workouts_rows.csv')
    print("Tải hệ thống gợi ý thành công!")
except Exception as e:
    print(f"Lỗi tải hệ thống gợi ý: {e}")
    recommender_engine = None

# Định nghĩa endpoint dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Mô hình chưa được tải."}), 500

    try:
        # Lấy dữ liệu JSON từ app mobile gửi lên
        data = request.json
        print(f"Nhận được dữ liệu: {data}")

        # Chuyển đổi JSON thành DataFrame 
        # Dữ liệu JSON là một object, cần bọc nó trong list []
        # để DataFrame hiểu đây là 1 hàng duy nhất
        input_df = pd.DataFrame([data])
        
        # Đảm bảo các cột số được chuyển đúng kiểu (vì JSON có thể gửi số dưới dạng text)
        # Lấy danh sách cột số từ pipeline (nếu cần, nhưng RF thường khá linh hoạt)
        # Tạm thời tin tưởng dữ liệu đầu vào là đúng kiểu

        # Thực hiện dự đoán
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        # Lấy tên các nhãn
        classes = model_pipeline.classes_
        # Tạo dict xác suất
        proba_dict = dict(zip(classes, prediction_proba[0]))

        # Trả kết quả về cho app mobile
        return jsonify({
            'du_doan': prediction[0],
            'chi_tiet_xac_suat': proba_dict
        })

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return jsonify({"error": str(e)}), 400

# Định nghĩa endpoint gợi ý bài tập
@app.route('/recommend', methods=['POST'])
def recommend_workout():
    if recommender_engine is None:
        return jsonify({"error": "Hệ thống gợi ý chưa sẵn sàng."}), 500

    try:
        # 1. Nhận dữ liệu JSON từ App
        # App cần gửi lên: thông tin user + kết quả dự đoán (level)
        input_data = request.json
        print(f"📩 Nhận yêu cầu gợi ý cho: {input_data.get('muc_tieu_chinh')}")

        # 2. Chạy thuật toán gợi ý
        # Hàm này trả về DataFrame
        result_df = recommender_engine.recommend_from_api_json(input_data)

        # 3. Kiểm tra kết quả
        if result_df.empty:
            return jsonify({
                "message": "Không tìm thấy bài tập phù hợp.",
                "data": []
            }), 200

        # 4. Chuyển đổi DataFrame sang JSON list
        result_list = result_df.to_dict('records')

        return jsonify({
            "message": "Success",
            "count": len(result_list),
            "data": result_list
        })

    except Exception as e:
        print(f"Lỗi gợi ý: {e}")
        return jsonify({"error": str(e)}), 400

# Chạy server
if __name__ == '__main__':
    # Chạy ở chế độ debug để dễ sửa lỗi
    # Khi "lên sóng", bạn sẽ dùng một server thật như Gunicorn
    app.run(debug=True, use_reloader=False, port=5000)