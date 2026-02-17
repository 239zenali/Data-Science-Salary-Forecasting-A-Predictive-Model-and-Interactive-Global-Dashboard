import streamlit as st
import pandas as pd
import pickle

# ======================================================
# CUSTOM CSS TO REDUCE PADDING & SPACING
# ======================================================

st.markdown(
    """
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Reduce spacing between elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }

    /* Reduce metric padding */
    div[data-testid="metric-container"] {
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# 1. LOAD MODEL, DATA, METADATA
# ======================================================

@st.cache_resource
def load_model():
    return pickle.load(open("final_lgb_model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("recommendation_data.csv")

@st.cache_data
def load_metadata():
    numeric_features = pickle.load(open("numeric_features.pkl", "rb"))
    categorical_features = pickle.load(open("categorical_features.pkl", "rb"))
    X_columns = pickle.load(open("X_columns.pkl", "rb"))
    return numeric_features, categorical_features, X_columns

model = load_model()
data = load_data()
numeric_features, categorical_features, X_columns = load_metadata()

# ======================================================
# 2. PAGE SETUP (DESIGN CHANGE ONLY)
# ======================================================

st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide"
)

st.title("🎯Salary Prediction & Company Recommendation")
st.caption("Predict salary based on user's choices and view closest matching companies")

st.divider()

# ======================================================
# 3. MAIN LAYOUT (LEFT = INPUTS | RIGHT = OUTPUTS)
# ======================================================

left_col, right_col = st.columns([1, 2], gap="large")

# ======================================================
# LEFT SIDE — USER INPUTS (NO LOGIC CHANGE)
# ======================================================

with left_col:
    st.subheader("📌 Enter Your Details")

    user_input = {}

    def select_with_placeholder(label, options):
        return st.selectbox(label, ["-- Select --"] + list(options))

    user_input["job_title"] = select_with_placeholder(
        "Job Title", sorted(data["job_title"].dropna().unique())
    )

    user_input["experience_level"] = select_with_placeholder(
        "Experience Level", sorted(data["experience_level"].dropna().unique())
    )

    user_input["employment_type"] = select_with_placeholder(
        "Employment Type", sorted(data["employment_type"].dropna().unique())
    )

    user_input["employee_residence"] = select_with_placeholder(
        "Employee Residence", sorted(data["employee_residence"].dropna().unique())
    )

    user_input["company_location"] = select_with_placeholder(
        "Company Location", sorted(data["company_location"].dropna().unique())
    )

    user_input["company_size"] = select_with_placeholder(
        "Company Size", sorted(data["company_size"].dropna().unique())
    )

    user_input["remote_ratio"] = select_with_placeholder(
        "Remote Ratio (%)", [0, 50, 100]
    )

    predict_clicked = st.button(
        "🔮 Predict Salary",
        use_container_width=True
    )

# ======================================================
# 4. PREDICTION & RECOMMENDATION LOGIC (UNCHANGED)
# ======================================================

def predict_salary_and_suggest_companies(user_features, model, original_data, top_n=5):
    user_df = pd.DataFrame([user_features])

    for col in X_columns:
        if col not in user_df.columns:
            if col in categorical_features:
                user_df[col] = original_data[col].mode()[0]
            else:
                user_df[col] = original_data[col].median()

    user_df = user_df[X_columns]

    predicted_salary = model.predict(user_df)[0]

    company_salary = (
        original_data[original_data["company_name"].notna()]
        .groupby("company_name")["salary_in_usd"]
        .mean()
        .reset_index()
        .rename(columns={"salary_in_usd": "avg_salary_usd"})
    )

    company_salary["diff"] = (
        company_salary["avg_salary_usd"] - predicted_salary
    ).abs()

    closest_companies = (
        company_salary
        .sort_values("diff")
        .head(top_n)
    )

    closest_companies = (
        closest_companies
        .sort_values("avg_salary_usd", ascending=False)
        .reset_index(drop=True)
    )

    closest_companies.insert(
        0, "Rank", range(1, len(closest_companies) + 1)
    )

    return predicted_salary, closest_companies

# ======================================================
# RIGHT SIDE — OUTPUTS (DESIGN CHANGE ONLY)
# ======================================================

with right_col:
    st.subheader("📊 Prediction Results")

    if predict_clicked:

        if "-- Select --" in user_input.values():
            st.warning("⚠️ Please select all fields on the left.")
        else:
            salary, companies = predict_salary_and_suggest_companies(
                user_input, model, data
            )

            st.metric(
                label="💰 Predicted Salary (USD)",
                value=f"${salary:,.2f}"
            )

            st.divider()

            st.subheader("🏢 Recommended Companies")
            # st.dataframe(
            #     companies[["Rank", "company_name", "avg_salary_usd"]],
            #     use_container_width=True,
            #     hide_index=True
            # )
            st.dataframe(
            companies[["Rank", "company_name", "avg_salary_usd"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank",
                    width="small"
                ),
                "company_name": st.column_config.TextColumn(
                    "Company Name",
                    width="medium"
                ),
                "avg_salary_usd": st.column_config.NumberColumn(
                    "Average Salary (USD)",
                    width="small",
                    format="$%d"
                ),
            }
        )


# ======================================================
# 6. FOOTER
# ======================================================

st.divider()
st.caption("📊 Powered by LightGBM | Salary Prediction System")
