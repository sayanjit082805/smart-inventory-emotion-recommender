import streamlit as st
import cv2
import sqlite3
import json
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Load product catalog
@st.cache_data
def load_catalog():
    with open("catalog.json", "r") as f:
        return json.load(f)

# DB functions
def get_all_products():
    with sqlite3.connect("inventory.db") as conn:
        return conn.execute("SELECT * FROM products").fetchall()

def get_logs():
    with sqlite3.connect("inventory.db") as conn:
        return conn.execute("SELECT * FROM logs ORDER BY log_id DESC").fetchall()

def update_inventory(product_id, direction):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect("inventory.db") as conn:
        if direction == "in":
            conn.execute("UPDATE products SET stock = stock + 1 WHERE product_id = ?", (product_id,))
            conn.execute("INSERT INTO logs (product_id, in_time) VALUES (?, ?)", (product_id, now))
        else:
            conn.execute("UPDATE products SET stock = stock - 1 WHERE product_id = ?", (product_id,))
            conn.execute("INSERT INTO logs (product_id, out_time) VALUES (?, ?)", (product_id, now))
        conn.commit()

# Initialize session state
if "scanning" not in st.session_state:
    st.session_state.scanning = False

# Page setup
st.set_page_config(page_title="🧠 Smart Inventory System", layout="wide")
st.title("📦 Smart Inventory + Emotion Recommender")

# 🎭 Emotion Detection
if st.button("🎭 Detect Emotion & Suggest Products"):
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
            .detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                st.success(f"Detected Emotion: {emotion}")
                st.markdown("### 🛍️ Recommended Products")
                catalog = load_catalog()
                for item in catalog:
                    if item["emotion"].lower() == emotion.lower():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(item["image"], width=100)
                        with col2:
                            st.markdown(f"**[{item['name']}]({item['link']})**  \nCategory: _{item['category']}_")
            except Exception as e:
                st.error(f"Error: {e}")
    cam.release()

st.markdown("---")

# 📸 Scan Products
st.subheader("📸 Scan Product to Update Inventory")

# Toggle scanner state
if not st.session_state.scanning:
    if st.button("📦 Start Product Scanning", key="start_button"):
        st.session_state.scanning = True
else:
    if st.button("🛑 Stop Scanning", key="stop_button"):
        st.session_state.scanning = False

# Run scanning logic
if st.session_state.scanning:
    cap = cv2.VideoCapture(0)
    detected = []

    stframe = st.empty()  # Streamlit placeholder

    while st.session_state.scanning:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        names = model.names
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for cls in classes:
            name = names[cls]
            if name not in detected:
                detected.append(name)
                with sqlite3.connect("inventory.db") as conn:
                    row = conn.execute("SELECT * FROM products WHERE LOWER(name) = ?", (name.lower(),)).fetchone()
                    if row:
                        update_inventory(row[0], "in")
                        st.success(f"✅ {name} stock increased.")
                    else:
                        st.warning(f"⚠️ {name} not found. Add manually.")

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")

# 📊 Inventory Dashboard
st.subheader("📊 Inventory Overview")
if st.button("🔁 Refresh"):
    st.rerun()

products = get_all_products()
low_stock = []

for pid, name, stock, cat, threshold in products:
    st.markdown(f"**{name}** ({cat}) — Stock: `{stock}`")
    if stock <= threshold:
        low_stock.append(name)

if low_stock:
    st.error(f"🚨 Low stock: {', '.join(low_stock)}")

# 🛠️ Manual Stock Editor
st.subheader("🛠️ Manual Stock Update")
product_names = [p[1] for p in products]
selected = st.selectbox("Select Product", product_names)

pid = next((p[0] for p in products if p[1] == selected), None)
col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Add 1"):
        update_inventory(pid, "in")
        st.success(f"{selected} stock increased")
with col2:
    if st.button("➖ Remove 1"):
        update_inventory(pid, "out")
        st.success(f"{selected} stock decreased")

# 📜 Logs & Export
st.subheader("📜 Recent Logs")
logs = get_logs()
log_df = pd.DataFrame(logs, columns=["Log ID", "Product ID", "In Time", "Out Time"])
st.dataframe(log_df)

if st.button("⬇ Export Logs to CSV"):
    log_df.to_csv("logs_export.csv", index=False)
    st.success("Exported as logs_export.csv")
