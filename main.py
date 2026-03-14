# -*- coding: utf-8 -*-
import os
from pathlib import Path

from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# ========================
# Flask設定
# ========================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "aidemy_car_classifier_secret_key"

# uploadsフォルダを作成
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


# ========================
# クラス定義
# indices:
# {'SUV_car': 0, 'coupe_car': 1, 'hatchback_car': 2,
#  'kei_car_japan': 3, 'minivan_car': 4, 'sedan_car': 5,
#  'station_wagon_car': 6}
# ========================
classes = [
    "SUV_car",            # 0
    "coupe_car",          # 1
    "hatchback_car",      # 2
    "kei_car_japan",      # 3
    "minivan_car",        # 4
    "sedan_car",          # 5
    "station_wagon_car"   # 6
]

# 画面表示用の日本語名
class_labels_ja = {
    "SUV_car": "SUV",
    "coupe_car": "クーペ",
    "hatchback_car": "ハッチバック",
    "kei_car_japan": "軽自動車",
    "minivan_car": "ミニバン",
    "sedan_car": "セダン",
    "station_wagon_car": "ステーションワゴン"
}


# ========================
# 学習済みモデル読み込み
# ========================
model = load_model("./car_classifier.h5")

# モデルの入力shapeを確認
# 例: (None, 224, 224, 3) など
input_shape = model.input_shape

if len(input_shape) != 4:
    raise ValueError(
        f"想定外のinput_shapeです: {input_shape}。"
        "通常は (None, height, width, channels) の4次元を想定しています。"
    )

_, image_height, image_width, image_channels = input_shape

if image_height is None or image_width is None or image_channels is None:
    raise ValueError(
        f"モデル入力shapeにNoneが含まれています: {input_shape}。"
        "画像サイズ・チャネル数を明示的に扱えるモデルであることを確認してください。"
    )

# PILの読み込みモードをモデルに合わせる
if image_channels == 1:
    PIL_MODE = "L"      # grayscale
elif image_channels == 3:
    PIL_MODE = "RGB"    # color
else:
    raise ValueError(
        f"未対応のチャネル数です: {image_channels}。"
        "1(ch=grayscale) または 3(ch=RGB) のモデルを想定しています。"
    )


# ========================
# ユーティリティ関数
# ========================
def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(filepath: str) -> np.ndarray:
    """
    画像をモデル入力用に前処理して返す
    戻り値 shape: (1, H, W, C)
    """
    img = Image.open(filepath).convert(PIL_MODE)
    img = img.resize((image_width, image_height))

    img_array = np.array(img, dtype=np.float32)

    # grayscaleの場合は (H, W) -> (H, W, 1)
    if image_channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    # 正規化: 0-255 -> 0-1
    img_array = img_array / 255.0

    # バッチ次元追加: (H, W, C) -> (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(filepath: str):
    """
    画像を予測して、
    predicted_index, predicted_class_name, predicted_label_ja, confidence, raw_result
    を返す
    """
    data = preprocess_image(filepath)

    result = model.predict(data, verbose=0)[0]

    predicted_index = int(np.argmax(result))
    predicted_class_name = classes[predicted_index]
    predicted_label_ja = class_labels_ja.get(predicted_class_name, predicted_class_name)
    confidence = float(result[predicted_index])

    return predicted_index, predicted_class_name, predicted_label_ja, confidence, result


# ========================
# ルーティング
# ========================
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # ファイル項目の存在確認
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)

        file = request.files["file"]

        # ファイル名が空の場合
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)

        # 拡張子チェック
        if not allowed_file(file.filename):
            flash("対応していないファイル形式です。png / jpg / jpeg / gif を使用してください。")
            return redirect(request.url)

        # 安全なファイル名に変換して保存
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            predicted_index, predicted_class_name, predicted_label_ja, confidence, raw_result = predict_image(filepath)

            pred_answer = (
                f"これは {predicted_label_ja} です "
                f"(class index: {predicted_index}, class name: {predicted_class_name}, "
                f"confidence: {confidence:.4f})"
            )

            # 各クラスの確率もテンプレートに渡したい場合
            class_probs = []
            for i, class_name in enumerate(classes):
                class_probs.append({
                    "index": i,
                    "class_name": class_name,
                    "label_ja": class_labels_ja.get(class_name, class_name),
                    "probability": float(raw_result[i])
                })

            return render_template(
                "index.html",
                answer=pred_answer,
                class_probs=class_probs
            )

        except Exception as e:
            flash(f"予測中にエラーが発生しました: {str(e)}")
            return redirect(request.url)

    return render_template("index.html", answer="", class_probs=[])


# ========================
# エントリポイント
# ========================
if __name__ == "__main__":
    # ローカル動作確認用
    app.run(host="0.0.0.0", port=8080, debug=True)