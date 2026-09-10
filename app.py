import streamlit as st
import pandas as pd
import joblib
import os
import base64
import textwrap

# --- 1. ASSET LOADING ---
_this_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_this_dir, 'best_attrition_model.pkl')
columns_path = os.path.join(_this_dir, 'best_model_columns.pkl')
logo_path = os.path.join(_this_dir, 'assets', 'logo.svg')

try:
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
except FileNotFoundError:
    st.error("Error: Model assets not found. Ensure .pkl files are in the same folder.")
    st.stop()

logo_data_uri = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as logo_file:
        encoded = base64.b64encode(logo_file.read()).decode("utf-8")
        logo_data_uri = f"data:image/svg+xml;base64,{encoded}"

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide", initial_sidebar_state="expanded")

# --- 2. REFINED CSS FOR ORGANIZED CARDS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        :root {
            --bg: #0b0f1a;
            --panel: rgba(18, 24, 38, 0.7);
            --panel-strong: rgba(18, 24, 38, 0.9);
            --border: rgba(255, 255, 255, 0.08);
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #7c3aed;
            --accent-2: #4f46e5;
        }

        .stApp {
            background: radial-gradient(1200px 600px at 15% -10%, rgba(124, 58, 237, 0.25), transparent 60%),
                        radial-gradient(900px 500px at 95% 0%, rgba(79, 70, 229, 0.25), transparent 55%),
                        var(--bg);
            font-family: 'Inter', sans-serif;
            color: var(--text);
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 20, 32, 0.95) 0%, rgba(16, 20, 32, 0.8) 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] label {
            color: var(--text);
        }

        /* LOGO & HEADER */
        .header-container {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 1.5rem;
        }
        .logo-box {
            width: 56px;
            height: 56px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--border);
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.35);
        }
        .logo-img {
            width: 36px;
            height: 36px;
        }

        /* GLASS CARDS */
        .glass-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            height: 100%;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
        }

        .card-title {
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 14px;
            border-bottom: 1px solid rgba(124, 58, 237, 0.2);
            padding-bottom: 8px;
        }
        .title-box {
            display: inline-flex;
            align-items: center;
            padding: 10px 18px;
            border: 1px solid rgba(124, 58, 237, 0.35);
            border-radius: 10px;
            background: rgba(124, 58, 237, 0.06);
            color: #c4b5fd;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            margin-bottom: 16px;
        }

        /* OVERVIEW TABLE */
        .overview-table2 {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(15, 19, 32, 0.5);
        }
        .overview-table2 th {
            text-align: left;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--muted);
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
        }
        .overview-table2 td {
            padding: 12px 12px;
            color: var(--text);
            font-weight: 600;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .overview-table2 tr:last-child td {
            border-bottom: none;
        }

        /* BUTTON STYLE */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            width: 100%;
            transition: transform 0.15s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 22px rgba(79, 70, 229, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(79, 70, 229, 0.4);
        }

        /* METRICS + TEXT */
        .stMarkdown p {
            color: var(--text);
        }
        .stMarkdown small,
        .stMarkdown span {
            color: var(--muted);
        }

        /* INFO BOX */
        .stAlert {
            background: var(--panel-strong) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='title-box'>Employee Details</div>", unsafe_allow_html=True)
    u_age = st.slider('Age', 18, 60, 35)
    u_income = st.slider('Monthly Income ($)', 1000, 20000, 6500, step=250)
    u_level = st.slider('Job Level', 1, 5, 2)
    u_years = st.slider('Total Working Years', 0, 40, 10)
    u_ot = st.selectbox('Works OverTime?', ('No', 'Yes'))
    u_role = st.selectbox('Job Role', (
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
        'Manufacturing Director', 'Healthcare Representative', 'Manager', 
        'Sales Representative', 'Research Director', 'Human Resources'
    ))

# --- 4. ORGANIZED DASHBOARD ---

# Header
logo_html = ""
if logo_data_uri:
    logo_html = f"<img src=\"{logo_data_uri}\" class=\"logo-img\" alt=\"EA\" />"
else:
    logo_html = "EA"

st.markdown(f"""
    <div class="header-container">
        <div class="logo-box">{logo_html}</div>
        <h1 style='margin:0; font-weight:800; color:white;'>Employee Attrition Prediction</h1>
    </div>
""", unsafe_allow_html=True)

# Two-Column Layout
main_col1, main_col2 = st.columns([1.5, 1], gap="medium")

with main_col1:
    st.markdown("<div class='title-box'>Employee Overview</div>", unsafe_allow_html=True)
    
    overview_html = textwrap.dedent(f"""
        <table class="overview-table2">
            <thead>
                <tr>
                    <th>Age</th>
                    <th>MonthlyIncome</th>
                    <th>JobLevel</th>
                    <th>TotalWorkingYears</th>
                    <th>OverTime</th>
                    <th>JobRole</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{u_age}</td>
                    <td>{u_income}</td>
                    <td>{u_level}</td>
                    <td>{u_years}</td>
                    <td>{u_ot}</td>
                    <td>{u_role}</td>
                </tr>
            </tbody>
        </table>
    """)
    st.markdown(overview_html, unsafe_allow_html=True)

with main_col2:
    st.markdown("<div class='title-box'>Analysis Engine</div>", unsafe_allow_html=True)
    
    if st.button("Predict Attrition"):
        # Match CSV Feature names
        input_data = pd.DataFrame({
            'Age': [u_age],
            'MonthlyIncome': [u_income],
            'JobLevel': [u_level],
            'TotalWorkingYears': [u_years],
            'OverTime': [u_ot],
            'JobRole': [u_role]
        })

        input_encoded = pd.get_dummies(input_data)
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        proba = model.predict_proba(input_final)[0]
        risk = proba[1] * 100

        if risk < 45:
            st.markdown(f"<h1 style='color: #10b981; font-size: 3.5rem; margin:10px 0;'>{risk:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<span style='color:#10b981; font-weight:700;'>LOW ATTRITION RISK</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: #ef4444; font-size: 3.5rem; margin:10px 0;'>{risk:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<span style='color:#ef4444; font-weight:700;'>HIGH ATTRITION RISK</span>", unsafe_allow_html=True)
    else:
        st.write("\n\n")
        st.write("Awaiting analysis trigger...")
        st.write("\n\n")

st.info("The model analyzes factors like Monthly Income and Overtime to forecast probability.")