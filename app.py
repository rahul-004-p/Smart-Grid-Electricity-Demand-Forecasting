import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Grid Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0b0f17;
        color: #f5f7fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111722;
        border-right: 1px solid #252c38;
    }

    section[data-testid="stSidebar"] * {
        color: #f5f7fa !important;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #f8fafc !important;
    }

    /* Normal text */
    p, label, span {
        color: #d7dce5;
    }

    /* KPI cards */
    .kpi-card {
        background: #171d28;
        border: 1px solid #2b3442;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.20);
    }

    .kpi-title {
        color: #9ca7b8 !important;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #ffffff !important;
        font-size: 27px;
        font-weight: 700;
    }

    /* Prediction card - IMPORTANT CONTRAST FIX */
    .prediction-card {
        background: #172554 !important;
        border: 2px solid #60a5fa !important;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.30);
    }

    .prediction-label {
        color: #bfdbfe !important;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .prediction-value {
        color: #ffffff !important;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.2;
    }

    .prediction-unit {
        color: #dbeafe !important;
        font-size: 15px;
        margin-top: 8px;
    }

    /* Section cards */
    .section-card {
        background: #111722;
        border: 1px solid #252e3c;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    /* Info box */
    .info-box {
        background: #172033;
        border-left: 4px solid #60a5fa;
        padding: 15px 18px;
        border-radius: 8px;
        margin: 12px 0;
    }

    .info-box p {
        color: #dbeafe !important;
        margin: 0;
    }

    /* Streamlit metric */
    div[data-testid="stMetric"] {
        background: #171d28;
        border: 1px solid #2b3442;
        padding: 15px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aeb8c7 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "England_Wales_Electricity_2021_2025_for_prediction.csv"

# Also support your earlier dataset name
DATA_FILE_ALTERNATIVE = "England_Wales_Electricity_2021_2025_final.csv"

MODEL_FILE = "elacricty_final_model.pkl"
FEATURE_FILE = "elacricty_feature_cols.pkl"


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Demand_Previous_24hr",
    "Demand_Previous_7days",
    "DoW_sin",
    "DoW_cos",
    "Solar_Capacity",
    "Hour_cos",
    "Hour_sin",
    "Month_cos",
    "Month_sin",
    "Wind_Capacity",
    "Is_Weekend",
    "Quarter",
    "Year"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_feature_list():
    if not os.path.exists(FEATURE_FILE):
        return EXPECTED_FEATURES

    return joblib.load(FEATURE_FILE)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    filename = None

    if os.path.exists(DATA_FILE):
        filename = DATA_FILE

    elif os.path.exists(DATA_FILE_ALTERNATIVE):
        filename = DATA_FILE_ALTERNATIVE

    if filename is None:
        return None

    try:

        df = pd.read_csv(filename)

        # Convert date columns if available
        for col in ["Date", "Datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# ============================================================
# CREATE TIME FEATURES
# ============================================================

def create_time_features(date_value, hour):

    timestamp = pd.Timestamp(date_value)

    day_of_week = timestamp.dayofweek
    month = timestamp.month
    quarter = timestamp.quarter
    year = timestamp.year

    # Day of week cyclical encoding
    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # Hour cyclical encoding
    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    # Month cyclical encoding
    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    is_weekend = int(day_of_week >= 5)

    return {
        "DoW_sin": dow_sin,
        "DoW_cos": dow_cos,
        "Hour_cos": hour_cos,
        "Hour_sin": hour_sin,
        "Month_cos": month_cos,
        "Month_sin": month_sin,
        "Is_Weekend": is_weekend,
        "Quarter": quarter,
        "Year": year
    }


# ============================================================
# CREATE PREDICTION INPUT
# ============================================================

def create_prediction_dataframe(
    date_value,
    hour,
    wind_capacity,
    solar_capacity,
    previous_24hr,
    previous_7days
):

    time_features = create_time_features(
        date_value,
        hour
    )

    row = {

        "Demand_Previous_24hr":
            previous_24hr,

        "Demand_Previous_7days":
            previous_7days,

        "DoW_sin":
            time_features["DoW_sin"],

        "DoW_cos":
            time_features["DoW_cos"],

        "Solar_Capacity":
            solar_capacity,

        "Hour_cos":
            time_features["Hour_cos"],

        "Hour_sin":
            time_features["Hour_sin"],

        "Month_cos":
            time_features["Month_cos"],

        "Month_sin":
            time_features["Month_sin"],

        "Wind_Capacity":
            wind_capacity,

        "Is_Weekend":
            time_features["Is_Weekend"],

        "Quarter":
            time_features["Quarter"],

        "Year":
            time_features["Year"]
    }

    return pd.DataFrame([row])


# ============================================================
# FORMAT NUMBER
# ============================================================

def format_number(value):

    try:
        return f"{value:,.2f}"
    except:
        return str(value)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 25px 0;">
            <div style="font-size:42px;">⚡</div>
            <h2 style="margin:0;">Smart Grid</h2>
            <p style="color:#9ca7b8;">Electricity Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Dataset Explorer",
            "Demand Analysis",
            "Exploratory Data Analysis",
            "Machine Learning",
            "Future Forecast",
            "Feature Importance",
            "Smart Grid Insights",
            "About Project"
        ]
    )

    st.markdown("---")

    st.caption(
        "Smart Grid Electricity Demand Forecasting"
    )


# ============================================================
# LOAD DATA / MODEL
# ============================================================

df = load_data()
model = load_model()
feature_cols = load_feature_list()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="font-size:38px; margin-bottom:0;">
        Smart Grid Electricity Demand Forecasting
    </h1>
    <p style="font-size:17px; color:#9ca7b8;">
        Machine Learning • Electricity Demand Analysis • Forecasting
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA MISSING CHECK
# ============================================================

if df is None:

    st.error(
        "Dataset file not found. Please place "
        "`England_Wales_Electricity_2021_2025_for_prediction.csv` "
        "or `England_Wales_Electricity_2021_2025_final.csv` "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Project Overview")

    st.markdown(
        """
        <div class="info-box">
        <p>
        This application analyzes electricity demand patterns in
        England and Wales and uses Machine Learning to forecast
        future electricity demand.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_records = len(df)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    target_candidates = [
        "Demand",
        "Electricity_Demand",
        "Demand_MW",
        "demand",
        "Actual_Demand"
    ]

    target_col = None

    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    if target_col is None and numeric_cols:
        # Try to identify a demand column
        demand_cols = [
            c for c in numeric_cols
            if "demand" in c.lower()
        ]

        if demand_cols:
            target_col = demand_cols[0]

    if target_col:

        avg_demand = df[target_col].mean()
        max_demand = df[target_col].max()
        min_demand = df[target_col].min()

    else:

        avg_demand = 0
        max_demand = 0
        min_demand = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{total_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Demand</div>
                <div class="kpi-value">{format_number(avg_demand)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Maximum Demand</div>
                <div class="kpi-value">{format_number(max_demand)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Minimum Demand</div>
                <div class="kpi-value">{format_number(min_demand)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    rows = st.slider(
        "Rows to display",
        min_value=5,
        max_value=min(100, len(df)),
        value=min(10, len(df))
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Unique Values": [
            int(df[c].nunique())
            for c in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ============================================================
# DEMAND ANALYSIS
# ============================================================

elif page == "Demand Analysis":

    st.subheader("Electricity Demand Analysis")

    demand_candidates = [
        c for c in df.columns
        if "demand" in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not demand_candidates:

        st.warning(
            "No numerical demand column was automatically detected."
        )

    else:

        demand_col = st.selectbox(
            "Select Demand Column",
            demand_candidates
        )

        chart_df = df.copy()

        datetime_col = None

        if "Datetime" in chart_df.columns:
            datetime_col = "Datetime"

        elif "Date" in chart_df.columns:
            datetime_col = "Date"

        if datetime_col:

            chart_df[datetime_col] = pd.to_datetime(
                chart_df[datetime_col],
                errors="coerce"
            )

            chart_df = chart_df.dropna(
                subset=[datetime_col]
            )

            fig = px.line(
                chart_df,
                x=datetime_col,
                y=demand_col,
                title="Electricity Demand Over Time"
            )

            fig.update_layout(
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            temp = chart_df.copy()

            temp["Hour"] = temp[
                datetime_col
            ].dt.hour

            hourly = (
                temp.groupby("Hour")[demand_col]
                .mean()
                .reset_index()
            )

            fig2 = px.line(
                hourly,
                x="Hour",
                y=demand_col,
                markers=True,
                title="Average Demand by Hour"
            )

            fig2.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# ============================================================
# EDA
# ============================================================

elif page == "Exploratory Data Analysis":

    st.subheader("Exploratory Data Analysis")

    demand_candidates = [
        c for c in df.columns
        if "demand" in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if demand_candidates:

        demand_col = st.selectbox(
            "Demand Column",
            demand_candidates
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.histogram(
                df,
                x=demand_col,
                nbins=50,
                title="Demand Distribution"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.box(
                df,
                y=demand_col,
                title="Demand Box Plot"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Numerical Feature Correlation"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.subheader("Machine Learning Model")

    if model is None:

        st.error(
            f"Model file `{MODEL_FILE}` was not found."
        )

    else:

        st.success(
            "Trained Machine Learning model loaded successfully."
        )

        st.write(
            "**Model:** Random Forest / Selected trained model"
        )

        st.write(
            "**Number of features:**",
            len(feature_cols)
        )

        st.subheader("Features Used by Model")

        st.dataframe(
            pd.DataFrame({
                "Feature": feature_cols
            }),
            use_container_width=True
        )

        st.markdown(
            """
            <div class="info-box">
            <p>
            The prediction interface does not require the user to
            manually enter every model feature. Time-based features
            are automatically calculated from the selected date and hour.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FUTURE FORECAST
# ============================================================

elif page == "Future Forecast":

    st.subheader("Forecast Demand")

    st.write(
        "Predict electricity demand for a selected future time."
    )

    if model is None:

        st.error(
            f"Model file `{MODEL_FILE}` was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        selected_date = st.date_input(
            "Date",
            value=pd.Timestamp.today().date()
        )

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

        wind_capacity = st.number_input(
            "Wind Capacity (MW)",
            min_value=0.0,
            value=15000.0,
            step=100.0
        )

        previous_24hr = st.number_input(
            "Demand 24 Hours Ago (MW)",
            min_value=0.0,
            value=25000.0,
            step=100.0
        )

    with right:

        solar_capacity = st.number_input(
            "Solar Capacity (MW)",
            min_value=0.0,
            value=16000.0,
            step=100.0
        )

        previous_7days = st.number_input(
            "Demand Same Time Last Week (MW)",
            min_value=0.0,
            value=25000.0,
            step=100.0
        )

        # Automatically calculated weekend status
        selected_day = pd.Timestamp(
            selected_date
        ).day_name()

        is_weekend = (
            pd.Timestamp(selected_date).dayofweek >= 5
        )

        st.markdown(
            f"""
            <div class="info-box">
                <p>
                <b>Day:</b> {selected_day}<br>
                <b>Weekend:</b> {"Yes" if is_weekend else "No"}<br>
                <b>Weekend value:</b> {int(is_weekend)}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict = st.button(
        "⚡ Predict Electricity Demand",
        type="primary",
        use_container_width=True
    )

    if predict:

        try:

            # Create complete 13-feature dataframe
            input_df = create_prediction_dataframe(
                date_value=selected_date,
                hour=hour,
                wind_capacity=wind_capacity,
                solar_capacity=solar_capacity,
                previous_24hr=previous_24hr,
                previous_7days=previous_7days
            )

            # ------------------------------------------------
            # EXACT FEATURE ORDER
            # ------------------------------------------------

            missing_features = [
                col for col in feature_cols
                if col not in input_df.columns
            ]

            if missing_features:

                st.error(
                    "Missing model features: "
                    + ", ".join(missing_features)
                )

                st.stop()

            # Select EXACT feature order
            X_prediction = input_df[
                feature_cols
            ].copy()

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                X_prediction
            )[0]

            prediction = float(prediction)

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        FORECASTED ELECTRICITY DEMAND
                    </div>

                    <div class="prediction-value">
                        {prediction:,.2f}
                    </div>

                    <div class="prediction-unit">
                        MW • Machine Learning Prediction
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PREDICTION SUMMARY
            # ------------------------------------------------

            st.subheader("Prediction Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Selected Date",
                    str(selected_date)
                )

            with c2:
                st.metric(
                    "Hour",
                    f"{hour:02d}:00"
                )

            with c3:
                st.metric(
                    "Predicted Demand",
                    f"{prediction:,.2f} MW"
                )

            # ------------------------------------------------
            # SHOW MODEL INPUTS
            # ------------------------------------------------

            with st.expander(
                "View 13 Features Sent to the Model"
            ):

                display_df = X_prediction.T.reset_index()

                display_df.columns = [
                    "Feature",
                    "Value"
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True
                )

            # ------------------------------------------------
            # AUTOMATIC FEATURES
            # ------------------------------------------------

            with st.expander(
                "View Automatically Calculated Features"
            ):

                auto_features = [
                    "DoW_sin",
                    "DoW_cos",
                    "Hour_cos",
                    "Hour_sin",
                    "Month_cos",
                    "Month_sin",
                    "Is_Weekend",
                    "Quarter",
                    "Year"
                ]

                st.dataframe(
                    X_prediction[
                        auto_features
                    ].T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.subheader("Feature Importance")

    if model is None:

        st.error(
            f"Model file `{MODEL_FILE}` was not found."
        )

    elif hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance — Random Forest"
        )

        fig.update_layout(
            template="plotly_dark",
            height=650
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Feature Importance Table")

        st.dataframe(
            importance_df.sort_values(
                "Importance",
                ascending=False
            ),
            use_container_width=True
        )

    elif hasattr(model, "coef_"):

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": model.coef_
        })

        st.dataframe(
            importance_df,
            use_container_width=True
        )

    else:

        st.info(
            "This model does not expose direct feature importance."
        )


# ============================================================
# SMART GRID INSIGHTS
# ============================================================

elif page == "Smart Grid Insights":

    st.subheader("Smart Grid Insights")

    demand_candidates = [
        c for c in df.columns
        if "demand" in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not demand_candidates:

        st.warning(
            "No numerical demand column detected."
        )

    else:

        demand_col = st.selectbox(
            "Demand Column",
            demand_candidates
        )

        avg = df[demand_col].mean()
        maximum = df[demand_col].max()
        minimum = df[demand_col].min()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">
                        Average Demand
                    </div>
                    <div class="kpi-value">
                        {avg:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">
                        Peak Demand
                    </div>
                    <div class="kpi-value">
                        {maximum:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">
                        Minimum Demand
                    </div>
                    <div class="kpi-value">
                        {minimum:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Hourly insight
        if "Datetime" in df.columns:

            temp = df.copy()

            temp["Datetime"] = pd.to_datetime(
                temp["Datetime"],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=["Datetime"]
            )

            temp["Hour"] = (
                temp["Datetime"].dt.hour
            )

            hourly = (
                temp.groupby("Hour")[demand_col]
                .mean()
            )

            peak_hour = hourly.idxmax()

            st.markdown(
                f"""
                <div class="info-box">
                <p>
                <b>Peak Average Demand Hour:</b>
                {peak_hour:02d}:00
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":

    st.subheader("About the Project")

    st.markdown(
        """
        <div class="section-card">

        <h2>Smart Grid Electricity Demand Forecasting</h2>

        <p>
        A Machine Learning based electricity demand analysis and
        forecasting project designed to understand electricity
        consumption patterns and predict future demand.
        </p>

        <h3>Technologies</h3>

        <p>
        Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit
        </p>

        <h3>Machine Learning</h3>

        <p>
        The application uses a trained Machine Learning model with
        historical demand, renewable capacity and time-based
        engineered features.
        </p>

        <h3>Key Features</h3>

        <p>
        • Dataset exploration<br>
        • Electricity demand analysis<br>
        • Exploratory data analysis<br>
        • Machine Learning prediction<br>
        • Feature importance<br>
        • Future demand forecasting<br>
        • Smart Grid insights
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )