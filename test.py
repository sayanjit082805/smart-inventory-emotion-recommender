import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import pandas as pd
import streamlit as st
from deepface import DeepFace
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Page setup
st.set_page_config(page_title="Inventory Dashboard", layout="wide")
st.title("Dashboard")


# Load initial sample data
def load_data():
    return pd.DataFrame(
        {
            "Product ID": ["P001", "P002", "P003", "P004"],
            "Product Name": ["Milk", "Eggs", "Bread", "Juice"],
            "Category": ["Dairy", "Poultry", "Bakery", "Beverage"],
            "Stock": [45, 12, 3, 25],
            "Reorder Level": [20, 10, 5, 15],
        }
    )


# Initialize session state
if "inventory" not in st.session_state:
    st.session_state.inventory = load_data()

inventory = st.session_state.inventory

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Inventory", "Add/Update", "Scan", "Analytics", "Upload/Download"],
)


# Row highlight for low stock
def highlight_low_stock(row):
    style = [""] * len(row)
    if row["Stock"] < row["Reorder Level"]:
        style[row.index.get_loc("Stock")] = "background-color: #ffcccc"
    return style


# Section: Overview
if section == "Overview":
    st.subheader("Overview")
    total_products = len(inventory)
    low_stock = (inventory["Stock"] < inventory["Reorder Level"]).sum()

    col1, col2 = st.columns(2)
    col1.metric("Total Products", total_products)
    col2.metric("Low Stock Items", low_stock)

    st.dataframe(
        inventory.style.apply(highlight_low_stock, axis=1), use_container_width=True
    )

# Section: Inventory
elif section == "Inventory":
    st.subheader("Present State")

    category_filter = st.selectbox(
        "Filter by Category", options=["All"] + inventory["Category"].unique().tolist()
    )
    if category_filter != "All":
        filtered = inventory[inventory["Category"] == category_filter]
    else:
        filtered = inventory

    st.dataframe(
        filtered.style.apply(highlight_low_stock, axis=1), use_container_width=True
    )

# Section: Add or Update
elif section == "Add/Update":
    st.subheader("Add or Update Product")

    with st.form("add_update_form"):
        product_id = st.text_input("Product ID")
        product_name = st.text_input("Product Name")
        category = st.text_input("Category")
        stock = st.number_input("Stock", min_value=0, step=1)
        reorder = st.number_input("Reorder Level", min_value=0, step=1)

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not product_id:
                st.error("Product ID is required.")
            else:
                idx = inventory[inventory["Product ID"] == product_id].index
                if len(idx) > 0:
                    st.session_state.inventory.loc[
                        idx, ["Product Name", "Category", "Stock", "Reorder Level"]
                    ] = [product_name, category, stock, reorder]
                    st.success("Product updated!")
                else:
                    new_row = pd.DataFrame(
                        {
                            "Product ID": [product_id],
                            "Product Name": [product_name],
                            "Category": [category],
                            "Stock": [stock],
                            "Reorder Level": [reorder],
                        }
                    )
                    st.session_state.inventory = pd.concat(
                        [inventory, new_row], ignore_index=True
                    )
                    st.success("Product added!")

elif section == "Scan":
    st.subheader("Scan Inventory")

    if "scanning" not in st.session_state:
        st.session_state.scanning = False
    if "detected" not in st.session_state:
        st.session_state.detected = set()
    if "cap" not in st.session_state:
        st.session_state.cap = None

    start_scan = st.button("▶️ Start Scanning")
    stop_scan = st.button("🛑 Stop Scanning")

    if start_scan and not st.session_state.scanning:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.scanning = True
        st.session_state.detected = set()
        st.success("Camera started. Scanning...")

    if stop_scan and st.session_state.cap:
        st.session_state.cap.release()
        cv2.destroyAllWindows()
        st.session_state.scanning = False
        st.session_state.cap = None
        st.success("Camera stopped.")

    if st.session_state.scanning:
        cap = st.session_state.cap
        ret, frame = cap.read()

        if ret:
            results = model.predict(frame, verbose=False)
            names = model.names
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            new_detections = set()

            for cls in classes:
                product_name = names[cls]

                if product_name not in st.session_state.detected:
                    new_detections.add(product_name)
                    st.session_state.detected.add(product_name)

                    matched_idx = inventory[inventory["Product Name"].str.lower() == product_name.lower()].index

                    if len(matched_idx) > 0:
                        row = matched_idx[0]
                        current_stock = inventory.at[row, "Stock"]
                        reorder_level = inventory.at[row, "Reorder Level"]

                        st.success(f"Detected: **{product_name}** (Current Stock: {current_stock})")

                        if current_stock < reorder_level:
                            st.warning(f"Stock is **below reorder level** ({current_stock} < {reorder_level}). Consider reordering.")
                        else:
                            st.info("Stock is sufficient.")
                    else:
                        st.error(f"**{product_name.capitalize()}** not found in inventory.")
                        st.markdown("Manually add from the **Add/Update** section.")

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Camera Feed")

        else:
            st.error("Failed to read from camera.")



# Section: Analytics
elif section == "Analytics":
    st.subheader("Inventory Analytics")

    fig_bar = px.bar(
        inventory,
        x="Product Name",
        y="Stock",
        color="Category",
        title="Stock by Product",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(inventory, names="Category", title="Category Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# Section: Upload/Download
elif section == "Upload/Download":
    st.subheader("Download or Upload Inventory")

    st.download_button(
        "Download Inventory",
        data=inventory.to_csv(index=False),
        file_name="inventory.csv",
    )

    uploaded = st.file_uploader("Upload New Inventory", type="csv")
    if uploaded:
        try:
            new_data = pd.read_csv(uploaded)
            required_cols = {
                "Product ID",
                "Product Name",
                "Category",
                "Stock",
                "Reorder Level",
            }
            if not required_cols.issubset(new_data.columns):
                st.error("Uploaded file is missing required columns.")
            else:
                st.session_state.inventory = new_data
                st.success("Inventory updated from uploaded file!")
        except Exception as e:
            st.error(f"Error: {e}")


# Section: Analytics
elif section == "Analytics":
    st.subheader("Inventory Analytics")

    fig_bar = px.bar(
        inventory,
        x="Product Name",
        y="Stock",
        color="Category",
        title="Stock by Product",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(inventory, names="Category", title="Category Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# Section: Upload/Download
elif section == "Upload/Download":
    st.subheader("Download or Upload Inventory")

    st.download_button(
        "Download Inventory",
        data=inventory.to_csv(index=False),
        file_name="inventory.csv",
    )

    uploaded = st.file_uploader("Upload New Inventory", type="csv")
    if uploaded:
        try:
            new_data = pd.read_csv(uploaded)
            required_cols = {
                "Product ID",
                "Product Name",
                "Category",
                "Stock",
                "Reorder Level",
            }
            if not required_cols.issubset(new_data.columns):
                st.error("Uploaded file is missing required columns.")
            else:
                st.session_state.inventory = new_data
                st.success("Inventory updated from uploaded file!")
        except Exception as e:
            st.error(f"Error: {e}")
