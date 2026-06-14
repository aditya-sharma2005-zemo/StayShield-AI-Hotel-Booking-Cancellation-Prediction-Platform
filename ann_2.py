import os
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="StayShield AI",
    page_icon="🏨",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#0f172a;
}

.main-title{
    font-size:3rem;
    font-weight:700;
    text-align:center;
    color:white;
}

.sub-title{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.card{
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(15px);
    padding:20px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:20px;
}

.result-card{
    background:rgba(255,255,255,0.08);
    padding:25px;
    border-radius:20px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ENCODERS
# --------------------------------------------------

CITY_MAP = {
    "Boston":0,
    "Chicago":1,
    "Dallas":2,
    "Las Vegas":3,
    "Los Angeles":4,
    "Miami":5,
    "New York":6,
    "Orlando":7,
    "San Francisco":8,
    "Seattle":9
}

ROOM_MAP = {
    "Deluxe":0,
    "Standard":1,
    "Suite":2
}

STAY_MAP = {
    "Business":0,
    "Leisure":1
}

BOOKING_CHANNEL_MAP = {
    "Mobile App":0,
    "Travel Agent":1,
    "Web":2
}

PAYMENT_MAP = {
    "Bank Transfer":0,
    "Credit Card":1,
    "Debit Card":2,
    "PayPal":3
}

CHANNEL_MAP = {
    "Android":0,
    "Web":1,
    "iOS":2
}

COUPON_MAP = {
    "No":0,
    "Yes":1
}

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        "hotel_cancellation_ann.h5"
    )

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_assets()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    "<div class='main-title'>🏨 StayShield AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Intelligent Booking Stability Assessment Platform</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🏨 StayShield AI")

    st.markdown("---")

    st.write(
        """
        Analyze reservation reliability
        using Deep Learning.
        """
    )

# --------------------------------------------------
# INPUT SECTIONS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Guest Profile")

    city = st.selectbox(
        "City",
        list(CITY_MAP.keys())
    )

    star_rating = st.slider(
        "Star Rating",
        1,
        5,
        3
    )

    room_type = st.selectbox(
        "Room Type",
        list(ROOM_MAP.keys())
    )

    stay_type = st.selectbox(
        "Stay Type",
        list(STAY_MAP.keys())
    )

with col2:

    st.subheader("Reservation Details")

    num_rooms_booked = st.number_input(
        "Rooms Booked",
        1,
        20,
        1
    )

    length_of_stay = st.number_input(
        "Length Of Stay",
        1,
        30,
        2
    )

    lead_time = st.number_input(
        "Lead Time",
        0,
        365,
        30
    )

    is_weekend_checkin = st.selectbox(
        "Weekend Check-In",
        ["No", "Yes"]
    )

st.divider()

col3, col4 = st.columns(2)

with col3:

    st.subheader("Financial Information")

    booking_value = st.number_input(
        "Booking Value",
        min_value=0.0,
        value=10000.0
    )

    costprice = st.number_input(
        "Cost Price",
        min_value=0.0,
        value=7000.0
    )

    markup = st.number_input(
        "Markup",
        min_value=0.0,
        value=3000.0
    )

    selling_price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=10000.0
    )

    cashback = st.number_input(
        "Cashback",
        min_value=0.0,
        value=0.0
    )

    coupon = st.selectbox(
        "Coupon Used",
        ["No", "Yes"]
    )

with col4:

    st.subheader("Distribution Channel")

    booking_channel = st.selectbox(
        "Booking Channel",
        list(BOOKING_CHANNEL_MAP.keys())
    )

    payment_method = st.selectbox(
        "Payment Method",
        list(PAYMENT_MAP.keys())
    )

    channel_of_booking = st.selectbox(
        "Platform",
        list(CHANNEL_MAP.keys())
    )

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

st.divider()

st.subheader("Booking Summary")

st.info(
    f"""
📍 {city}

🏨 {star_rating} Star Hotel

🛏 {room_type}

📅 {length_of_stay} Night Stay

💰 Booking Value: ₹{booking_value:,.0f}
"""
)

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

predict_btn = st.button(
    "🔍 Analyze Booking Stability",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_btn:

    features = np.array([[
        CITY_MAP[city],
        star_rating,
        ROOM_MAP[room_type],
        num_rooms_booked,
        STAY_MAP[stay_type],
        BOOKING_CHANNEL_MAP[booking_channel],
        booking_value,
        costprice,
        markup,
        selling_price,
        PAYMENT_MAP[payment_method],
        CHANNEL_MAP[channel_of_booking],
        cashback,
        COUPON_MAP[coupon],
        length_of_stay,
        1 if is_weekend_checkin == "Yes" else 0,
        lead_time
    ]])

    scaled = scaler.transform(features)

    probability = float(
        model.predict(
            scaled,
            verbose=0
        )[0][0]
    )

    prediction = (
        "Cancelled"
        if probability > 0.5
        else "Confirmed"
    )

    st.divider()

    if probability > 0.8:

        st.error(
            "⚠ High Cancellation Risk"
        )

    elif probability > 0.5:

        st.warning(
            "🟡 Moderate Cancellation Risk"
        )

    else:

        st.success(
            "🟢 Stable Reservation"
        )

    st.metric(
        "Cancellation Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(probability)

    st.subheader("Recommended Actions")

    if probability > 0.8:

        st.markdown("""
        ✅ Contact guest directly

        ✅ Send reminder email

        ✅ Request advance payment

        ✅ Verify reservation details
        """)

    elif probability > 0.5:

        st.markdown("""
        ✅ Monitor reservation

        ✅ Send confirmation reminder

        ✅ Offer loyalty incentives
        """)

    else:

        st.markdown("""
        ✅ Reservation appears healthy

        ✅ Standard workflow recommended
        """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
"""
Model: Artificial Neural Network (ANN)

Framework: TensorFlow / Keras

Task: Hotel Booking Cancellation Prediction
"""
)