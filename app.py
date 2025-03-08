import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

# MediaPipe の Pose モジュールをセットアップ
mp_pose = mp.solutions.pose

# **関節名の日本語マッピング**
joint_names_jp = {
    mp_pose.PoseLandmark.NOSE: "鼻",
    mp_pose.PoseLandmark.LEFT_SHOULDER: "左肩",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "右肩",
    mp_pose.PoseLandmark.LEFT_ELBOW: "左肘",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "右肘",
    mp_pose.PoseLandmark.LEFT_WRIST: "左手首",
    mp_pose.PoseLandmark.RIGHT_WRIST: "右手首",
    mp_pose.PoseLandmark.LEFT_HIP: "左股関節",
    mp_pose.PoseLandmark.RIGHT_HIP: "右股関節",
    mp_pose.PoseLandmark.LEFT_KNEE: "左膝",
    mp_pose.PoseLandmark.RIGHT_KNEE: "右膝",
    mp_pose.PoseLandmark.LEFT_ANKLE: "左足首",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "右足首"
}

# Streamlit の設定
st.title("📷 画像のモーションキャプチャ解析アプリ")

# **カスタマイズ設定**
with st.sidebar:
    st.header("🔧 カスタマイズ設定")
    
    # **マーカーと線のデフォルトカラー（シンプル化）**
    marker_color = st.color_picker("マーカーの色", "#FFFFFF")  # 白
    line_color = st.color_picker("線の色", "#C8C8C8")  # グレー
    
    # **線の太さ選択**
    line_thickness = st.slider("線の太さ", 1, 5, 2)
    
    # **表示するマーカーの選択**
    st.subheader("表示する関節の選択")
    selected_joints = {
        "首": st.checkbox("首（頭部）", value=True),
        "上肢": st.checkbox("上肢（肩・肘・手）", value=True),
        "下肢": st.checkbox("下肢（股関節・膝・足）", value=True),
        "体幹": st.checkbox("体幹（両肩-両股関節の線）", value=True),
    }

# **関節グループの定義**
joints_map = {
    "首": [mp_pose.PoseLandmark.NOSE],
    "上肢": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
             mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
             mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST],
    "下肢": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
             mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
             mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
}

# **画像のアップロード**
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # OpenCV 形式に変換（BGR に変更）
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # **MediaPipe Pose を実行**
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.85) as pose:
        results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        h, w, _ = img_bgr.shape
        landmarks = {}

        # **ユーザーが選択した関節のみリストアップ**
        selected_landmarks = []
        for key, enabled in selected_joints.items():
            if enabled:
                selected_landmarks.extend(joints_map.get(key, []))

        # **体幹の線（両肩-両股関節）を表示するかどうか**
        show_trunk_line = selected_joints["体幹"]

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in selected_landmarks or (
                    show_trunk_line and idx in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
                ):
                    if landmark.visibility > 0.1:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        landmarks[idx] = [x, y]

        if not landmarks:
            st.warning("⚠️ 関節マーカーが検出されませんでした。")
        else:
            # **スライダーで手動調整（関節名付き）**
            st.sidebar.subheader("🔧 マーカー調整")
            adjusted_landmarks = {}
            for idx, (x, y) in landmarks.items():
                joint_name = joint_names_jp.get(mp_pose.PoseLandmark(idx), f"関節 {idx}")
                new_x = st.sidebar.slider(f"{joint_name} のX座標", 0, w, x, key=f"x_{idx}")
                new_y = st.sidebar.slider(f"{joint_name} のY座標", 0, h, y, key=f"y_{idx}")
                adjusted_landmarks[idx] = (new_x, new_y)

            # **画像にマーカーと線を描画**
            for idx, (x, y) in adjusted_landmarks.items():
                cv2.circle(img_bgr, (x, y), 7, tuple(int(marker_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)), -1)

            # **関節を線でつなぐ**
            for connection in mp_pose.POSE_CONNECTIONS:
                p1, p2 = connection
                if p1 in adjusted_landmarks and p2 in adjusted_landmarks:
                    # **体幹が非表示なら、両肩-両股関節の線だけを消す**
                    if not show_trunk_line and (
                        (p1 == mp_pose.PoseLandmark.LEFT_SHOULDER and p2 == mp_pose.PoseLandmark.RIGHT_SHOULDER) or
                        (p1 == mp_pose.PoseLandmark.LEFT_HIP and p2 == mp_pose.PoseLandmark.RIGHT_HIP) or
                        (p1 == mp_pose.PoseLandmark.LEFT_SHOULDER and p2 == mp_pose.PoseLandmark.LEFT_HIP) or
                        (p1 == mp_pose.PoseLandmark.RIGHT_SHOULDER and p2 == mp_pose.PoseLandmark.RIGHT_HIP)
                    ):
                        continue  # **体幹の線だけを消す**

                    cv2.line(
                        img_bgr, adjusted_landmarks[p1], adjusted_landmarks[p2],
                        tuple(int(line_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)), line_thickness
                    )

            # **結果画像を表示**
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="解析結果", use_column_width=True)

