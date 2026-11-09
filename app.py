
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="WattWise | Smart Energy",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f5f8fc;}
[data-testid="stMetric"] {
    background-color: #1b263b;
    color: white;
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: white !important;
}
h1, h2, h3 {color: #102a43;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ WattWise")
st.subheader("Smart Electricity Intelligence & Energy Saver")
st.caption("Forecast • Detect unusual usage • Estimate cost • Save energy")

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


@st.cache_data(show_spinner=False)
def load_uci_data():
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        filename = next(
            name for name in archive.namelist()
            if name.lower().endswith(".txt")
        )

        with archive.open(filename) as file:
            df = pd.read_csv(
                file,
                sep=";",
                low_memory=False,
                na_values=["?"]
            )

    return df


def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )
        df = df.drop(columns=["Date", "Time"])

    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"], errors="coerce"
        )

    elif "Datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Datetime"], errors="coerce"
        )
        df = df.drop(columns=["Datetime"])

    else:
        raise ValueError(
            "Dataset must contain Date and Time columns "
            "or a datetime column."
        )

    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    if "Global_active_power" in df.columns:
        df["power_kw"] = df["Global_active_power"]

    elif "power_kw" not in df.columns:
        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_cols:
            raise ValueError("No numeric electricity column found.")

        df["power_kw"] = df[numeric_cols[0]]

    df["date"] = df["datetime"].dt.floor("D")

    daily_data = df.groupby("date").agg(
        avg_power_kw=("power_kw", "mean"),
        max_power_kw=("power_kw", "max"),
        min_power_kw=("power_kw", "min"),
        readings=("power_kw", "count")
    ).reset_index()

    daily_data["energy_kwh"] = daily_data["avg_power_kw"] * 24
    daily_data["day_of_week"] = daily_data["date"].dt.dayofweek
    daily_data["month"] = daily_data["date"].dt.month
    daily_data["day_of_year"] = daily_data["date"].dt.dayofyear
    daily_data["is_weekend"] = (
        daily_data["day_of_week"] >= 5
    ).astype(int)

    daily_data = daily_data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["energy_kwh"])

    return daily_data


# SIDEBAR SETTINGS
with st.sidebar:
    st.header("⚙️ Project Settings")

    source = st.radio(
        "Choose dataset source",
        ["Download UCI dataset", "Upload CSV"]
    )

    tariff = st.number_input(
        "Electricity rate (₹ per kWh)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )

    test_percent = st.slider(
        "Test data (%)", 10, 30, 20, 5
    )

    st.info("Bill amounts are estimates for academic demonstration.")


# LOAD DATASET
raw = None

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload electricity CSV",
        type=["csv", "txt"]
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".txt"):
                raw = pd.read_csv(
                    uploaded,
                    sep=";",
                    na_values=["?"],
                    low_memory=False
                )
            else:
                raw = pd.read_csv(uploaded, na_values=["?"])

        except Exception as error:
            st.error(f"Could not read file: {error}")

else:
    if st.button("Load open-access UCI dataset", type="primary"):
        try:
            with st.spinner("Downloading dataset..."):
                st.session_state["raw_data"] = load_uci_data()

        except Exception as error:
            st.error(
                "Download failed. Check your internet connection "
                "or choose Upload CSV."
            )
            st.caption(f"Technical detail: {error}")

    if "raw_data" in st.session_state:
        raw = st.session_state["raw_data"]


# SHOW WELCOME SCREEN
if raw is None:
    st.markdown("### Welcome to WattWise")

    st.write(
        "Load the UCI dataset or upload a CSV containing Date, "
        "Time and Global_active_power columns."
    )

    st.markdown("""
    **Project workflow**

    1. Load and clean electricity data.
    2. Calculate daily energy usage.
    3. Train machine learning models.
    4. Forecast next-day consumption.
    5. Detect unusual usage.
    6. Estimate bills and suggest energy savings.
    """)

    st.stop()


# PREPARE DATA
try:
    daily = prepare_data(raw)

except Exception as error:
    st.error(f"Dataset preparation problem: {error}")
    st.stop()

if daily.empty:
    st.error("No valid daily records found in the dataset.")
    st.stop()

if len(daily) > 2500:
    daily = daily.tail(2500).reset_index(drop=True)

st.success(
    f"Dataset ready: {len(daily):,} daily records | "
    f"{daily['date'].min().date()} to "
    f"{daily['date'].max().date()}"
)

if len(daily) < 30:
    st.warning(
        "At least 30 daily records are recommended "
        "for a meaningful forecasting demonstration."
    )


# DASHBOARD METRICS
c1, c2, c3, c4 = st.columns(4)

c1.metric("Daily Records", f"{len(daily):,}")
c2.metric(
    "Average Daily Energy",
    f"{daily['energy_kwh'].mean():.2f} kWh"
)
c3.metric(
    "Peak Average Power",
    f"{daily['avg_power_kw'].max():.2f} kW"
)
c4.metric(
    "Missing Energy Values",
    f"{daily['energy_kwh'].isna().sum()}"
)


# DASHBOARD TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast & Models",
    "🚨 Unusual Usage",
    "₹ Bill Estimator",
    "🌱 Saving Advisor"
])


# TAB 1: FORECAST AND MODELS
with tab1:
    st.header("Daily Electricity Consumption")

    chart = px.line(
        daily,
        x="date",
        y="energy_kwh",
        title="Estimated Daily Energy Use",
        labels={
            "date": "Date",
            "energy_kwh": "Energy (kWh)"
        }
    )

    st.plotly_chart(chart, use_container_width=True)

    model_df = daily.copy().sort_values(
        "date"
    ).reset_index(drop=True)

    model_df["lag_1"] = model_df["energy_kwh"].shift(1)
    model_df["lag_7"] = model_df["energy_kwh"].shift(7)

    model_df["rolling_mean_7"] = (
        model_df["energy_kwh"].shift(1).rolling(7).mean()
    )

    model_df["rolling_mean_14"] = (
        model_df["energy_kwh"].shift(1).rolling(14).mean()
    )

    feature_cols = [
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_mean_14",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend"
    ]

    model_df = model_df.dropna(
        subset=feature_cols + ["energy_kwh"]
    ).copy()

    if len(model_df) < 2:
        st.error(
            "Not enough historical data to train the models. "
            "Upload more daily records."
        )
        st.stop()

    X = model_df[feature_cols]
    y = model_df["energy_kwh"]

    split_idx = int(
        len(model_df) * (1 - test_percent / 100)
    )

    split_idx = max(1, min(split_idx, len(model_df) - 1))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fitted[name] = model

        results.append({
            "Model": name,
            "MSE": mean_squared_error(y_test, predictions),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "MAE": mean_absolute_error(y_test, predictions),
            "R²": (
                r2_score(y_test, predictions)
                if len(y_test) > 1 else np.nan
            )
        })

    result_df = pd.DataFrame(results).sort_values("RMSE")

    st.subheader("Model Performance")

    st.dataframe(
        result_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Lower RMSE and MAE generally indicate more accurate predictions. "
        "The test data comes from a later time period."
    )

    chosen = st.selectbox(
        "Choose a model for prediction",
        list(models.keys()),
        index=list(models.keys()).index(
            result_df.iloc[0]["Model"]
        )
    )

    model = fitted[chosen]

    latest = daily.iloc[-1]
    next_date = latest["date"] + pd.Timedelta(days=1)

    energy_series = daily["energy_kwh"].reset_index(drop=True)

    next_input = pd.DataFrame([{
        "lag_1": float(energy_series.iloc[-1]),
        "lag_7": float(
            energy_series.iloc[-7]
            if len(energy_series) >= 7
            else energy_series.iloc[-1]
        ),
        "rolling_mean_7": float(energy_series.tail(7).mean()),
        "rolling_mean_14": float(energy_series.tail(14).mean()),
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "day_of_year": next_date.dayofyear,
        "is_weekend": int(next_date.dayofweek >= 5)
    }])[feature_cols]

    forecast = max(
        0.0,
        float(model.predict(next_input)[0])
    )

    st.subheader("Next-Day Consumption Forecast")

    a, b = st.columns(2)

    a.metric(
        f"Predicted Energy for {next_date.date()}",
        f"{forecast:.2f} kWh"
    )

    b.metric(
        "Estimated Energy Cost",
        f"₹{forecast * tariff:.2f}"
    )

    test_predictions = model.predict(X_test)

    pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": test_predictions,
        "Test date": model_df["date"].iloc[split_idx:].values
    })

    fig = px.line(
        pred_df,
        x="Test date",
        y=["Actual", "Predicted"],
        title=f"Actual vs Predicted — {chosen}"
    )

    st.plotly_chart(fig, use_container_width=True)


# TAB 2: UNUSUAL USAGE
with tab2:
    st.header("Unusual Electricity Usage")

    st.write(
        "Isolation Forest identifies daily energy readings "
        "that differ from the usual pattern."
    )

    contamination = st.slider(
        "Expected unusual days (%)",
        1, 10, 3
    ) / 100

    if len(daily) >= 2:
        detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        detector.fit(daily[["energy_kwh"]])

        scores = detector.predict(
            daily[["energy_kwh"]]
        )

        anomaly_df = daily[
            ["date", "energy_kwh", "avg_power_kw"]
        ].copy()

        anomaly_df["status"] = np.where(
            scores == -1,
            "Unusual Usage",
            "Typical Usage"
        )

        fig = px.scatter(
            anomaly_df,
            x="date",
            y="energy_kwh",
            color="status",
            title="Typical vs Unusual Usage",
            labels={
                "date": "Date",
                "energy_kwh": "Energy (kWh)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Unusual Usage Records")

        st.dataframe(
            anomaly_df[
                anomaly_df["status"] == "Unusual Usage"
            ].tail(30).round(3),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("At least two daily records are required.")


# TAB 3: BILL ESTIMATOR
with tab3:
    st.header("Electricity Bill Estimator")

    col1, col2 = st.columns(2)

    avg_daily = float(
        daily["energy_kwh"].tail(30).mean()
    )

    days = col1.slider(
        "Billing Period (days)",
        7, 60, 30
    )

    custom_rate = col2.number_input(
        "Tariff (₹/kWh)",
        min_value=0.0,
        value=float(tariff),
        step=0.5,
        key="bill_tariff"
    )

    estimated_units = avg_daily * days
    estimated_bill = estimated_units * custom_rate

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Average Daily Energy",
        f"{avg_daily:.2f} kWh"
    )

    m2.metric(
        f"Estimated Units ({days} days)",
        f"{estimated_units:.1f} kWh"
    )

    m3.metric(
        "Estimated Energy Cost",
        f"₹{estimated_bill:,.2f}"
    )

    st.caption(
        "This is a simplified estimate. Actual electricity bills "
        "may include fixed charges, taxes, duties and tariff slabs."
    )


# TAB 4: ENERGY-SAVING ADVISOR
with tab4:
    st.header("Personalized Energy-Saving Advisor")

    recent = daily.tail(30)

    baseline = float(
        daily["energy_kwh"].median()
    )

    recent_avg = float(
        recent["energy_kwh"].mean()
    )

    st.metric(
        "Recent 30-Day Average",
        f"{recent_avg:.2f} kWh/day",
        delta=f"{recent_avg - baseline:+.2f} vs dataset median"
    )

    tips = []

    if recent_avg > baseline * 1.10:
        tips.append(
            "Your recent average is more than 10% above the "
            "dataset median. Review high-use days and check "
            "whether heavy appliances ran longer than usual."
        )
    else:
        tips.append(
            "Your recent average is close to the dataset median. "
            "Continue monitoring daily usage."
        )

    tips.extend([
        "Run washing machines with full loads where practical.",
        "Switch off lights and devices when they are not needed.",
        "Use LED lights and energy-efficient appliances.",
        "Compare similar weekdays before deciding that a usage "
        "spike is abnormal."
    ])

    for number, tip in enumerate(tips, 1):
        st.markdown(f"**{number}.** {tip}")

    st.caption(
        "Advice is based on general usage patterns. "
        "The app does not directly measure individual appliances."
    )
