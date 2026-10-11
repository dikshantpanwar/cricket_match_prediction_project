                       
"""
Cricket Match Winner Prediction - Streamlit App
Run: python -m streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

                                                               
                                                               
                                                               
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "models", "best_model.pkl")
ENCODERS_PATH    = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
TEAM_STATS_PATH  = os.path.join(BASE_DIR, "models", "team_stats.pkl")

                                               
CAT_COLS = ["team1", "team2", "toss_winner", "toss_decision", "venue"]

                                                                               
FEATURE_COLS = [
    "team1", "team2", "toss_winner", "toss_decision", "venue",
    "team1_win_rate", "team2_win_rate", "win_rate_diff",
    "h2h_win_rate", "home_adv",
]

                                                           
HOME_VENUES = {
    "Chennai Super Kings"        : "MA Chidambaram Stadium, Chepauk",
    "Deccan Chargers"            : "Rajiv Gandhi International Stadium, Uppal",
    "Delhi Capitals"             : "Arun Jaitley Stadium",
    "Delhi Daredevils"           : "Feroz Shah Kotla",
    "Gujarat Lions"              : "Saurashtra Cricket Association Stadium",
    "Gujarat Titans"             : "Narendra Modi Stadium, Ahmedabad",
    "Kings XI Punjab"            : "Punjab Cricket Association Stadium, Mohali",
    "Kochi Tuskers Kerala"       : "Nehru Stadium",
    "Kolkata Knight Riders"      : "Eden Gardens",
    "Lucknow Super Giants"       : "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Mumbai Indians"             : "Wankhede Stadium",
    "Pune Warriors"              : "Subrata Roy Sahara Stadium",
    "Punjab Kings"               : "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Rajasthan Royals"           : "Sawai Mansingh Stadium",
    "Rising Pune Supergiant"     : "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiants"    : "Maharashtra Cricket Association Stadium",
    "Royal Challengers Bangalore": "M Chinnaswamy Stadium",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Sunrisers Hyderabad"        : "Rajiv Gandhi International Stadium, Uppal",
}


                                                               
                                                         
                                                               
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading encoders...")
def load_encoders():
    return joblib.load(ENCODERS_PATH)


@st.cache_resource(show_spinner="Loading team stats...")
def load_team_stats():
    return joblib.load(TEAM_STATS_PATH)


                                                               
                    
                                                               
def encode_input(raw: dict, encoders: dict) -> pd.DataFrame:
    """
    Build the model input row:
    - Label-encode the 5 categorical columns
    - Pass numeric features (win rates, h2h, home_adv) through as-is
    """
    row = {}
                                         
    for col in CAT_COLS:
        le  = encoders[col]
        val = str(raw[col])
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else -1
                                      
    for col in ["team1_win_rate", "team2_win_rate", "win_rate_diff", "h2h_win_rate", "home_adv"]:
        row[col] = raw[col]
    return pd.DataFrame([row])[FEATURE_COLS]                              


def build_raw_input(team1, team2, toss_winner, toss_decision, venue, team_stats):
    """Compute all numeric features from saved team_stats and return a raw dict."""
    win_rates = team_stats.get("win_rates", {})
    h2h_wins  = team_stats.get("h2h_wins",  {})
    h2h_total = team_stats.get("h2h_total", {})

    t1_wr  = win_rates.get(team1, 0.5)
    t2_wr  = win_rates.get(team2, 0.5)
    total  = h2h_total.get((team1, team2), 0)
    h2h_wr = h2h_wins.get((team1, team2), 0) / total if total > 0 else 0.5
    home   = 1 if HOME_VENUES.get(team1) == venue else 0

    return {
        "team1"          : team1,
        "team2"          : team2,
        "toss_winner"    : toss_winner,
        "toss_decision"  : toss_decision,
        "venue"          : venue,
        "team1_win_rate" : t1_wr,
        "team2_win_rate" : t2_wr,
        "win_rate_diff"  : t1_wr - t2_wr,
        "h2h_win_rate"   : h2h_wr,
        "home_adv"       : home,
    }


def predict(raw: dict, model, encoders: dict, team1: str, team2: str):
    """
    Run the model and return a head-to-head result for ONLY team1 vs team2.

    Steps:
    1. Encode the input and call predict_proba() to get probabilities for
       all 19 classes.
    2. Pull out only the probabilities for team1 and team2.
    3. Normalize: each team's probability / (team1_prob + team2_prob)
       so they add up to 100%.
    4. Winner = whichever team has the higher normalized probability.

    Returns
    -------
    winner     : str   -- team1 or team2 name
    t1_pct     : float -- normalized probability for team1 [0.0, 1.0]
    t2_pct     : float -- normalized probability for team2 [0.0, 1.0]
    """
    X = encode_input(raw, encoders)

                                                  
    proba      = model.predict_proba(X)[0]
    all_teams  = encoders["winner"].inverse_transform(model.classes_)
    proba_map  = dict(zip(all_teams, proba))

                                                          
                                                                  
    t1_raw = proba_map.get(team1, 0.0)
    t2_raw = proba_map.get(team2, 0.0)

                                                                          
    total  = t1_raw + t2_raw
    if total == 0:                                                        
        t1_pct, t2_pct = 0.5, 0.5
    else:
        t1_pct = t1_raw / total
        t2_pct = t2_raw / total

    winner = team1 if t1_pct >= t2_pct else team2
    return winner, t1_pct, t2_pct


                                                               
              
                                                               
st.set_page_config(
    page_title = "Cricket Match Winner Prediction",
    page_icon  = "🏏",
    layout     = "centered",
)

                                                               
                                  
                                                               
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f9d423, #ff4e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.hero p {
    color: #a0a8c8;
    font-size: 1.05rem;
    margin-top: 0;
}

/* ── Card (form container) ── */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* ── Section labels inside card ── */
.section-label {
    color: #f9d423;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(249,212,35,0.15), rgba(255,78,80,0.15));
    border: 1px solid rgba(249,212,35,0.4);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

.result-label {
    color: #a0a8c8;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-winner {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f9d423, #ff4e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.result-confidence {
    font-size: 1.1rem;
    color: #c8d0f0;
    font-weight: 500;
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 12px;
    margin: 1rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f9d423, #ff4e50);
    transition: width 0.8s ease;
}

/* ── Top-5 prob rows ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.prob-team {
    color: #c8d0f0;
    font-size: 0.88rem;
    width: 220px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}
.prob-pct {
    color: #f9d423;
    font-size: 0.82rem;
    font-weight: 700;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Warning / error boxes ── */
.warn-box {
    background: rgba(255, 193, 7, 0.15);
    border: 1px solid rgba(255, 193, 7, 0.5);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #ffe082;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}
.err-box {
    background: rgba(255, 78, 80, 0.15);
    border: 1px solid rgba(255, 78, 80, 0.5);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #ff8a8a;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    color: #c8d0f0 !important;
    font-weight: 500;
    font-size: 0.95rem;
}
div[data-testid="stSelectbox"] > div:first-child > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
.stButton > button {
    width: 100%;
    padding: 0.85rem 1.5rem;
    border-radius: 12px;
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f9d423, #ff4e50) !important;
    color: #0f0c29 !important;
    border: none !important;
    cursor: pointer;
    transition: opacity 0.2s ease, transform 0.15s ease;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


                                                               
                                                        
                                                               
try:
    model      = load_model()
    encoders   = load_encoders()
    team_stats = load_team_stats()
except FileNotFoundError as e:
    st.markdown("""
    <div class='err-box'>
        &#9888; <strong>Model files not found!</strong><br>
        Please run <code>python src/train.py</code> first to train the model,
        then re-launch this app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
except Exception as e:
    st.markdown(f"<div class='err-box'>&#10060; Failed to load model: <code>{e}</code></div>",
                unsafe_allow_html=True)
    st.stop()


                                                               
                                 
                                                               
teams          = sorted(encoders["team1"].classes_.tolist())
venues         = sorted(encoders["venue"].classes_.tolist())
toss_decisions = sorted(encoders["toss_decision"].classes_.tolist())


                                                               
              
                                                               
st.markdown("""
<div class="hero">
    <h1>🏏 Cricket Match Winner Prediction</h1>
    <p>Powered by Machine Learning · Select match details below to predict the winner</p>
</div>
""", unsafe_allow_html=True)


                                                               
                     
                                                               
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='section-label'>⚔️ Teams</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, key="team1")
with col2:
    team2 = st.selectbox("Team 2", teams,
                          index=1 if len(teams) > 1 else 0,
                          key="team2")

st.markdown("<div class='section-label'>🪙 Toss</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox("Toss Winner", [team1, team2],
                                key="toss_winner",
                                help="Which team won the toss?")
with col4:
    toss_decision = st.selectbox("Toss Decision", toss_decisions,
                                  key="toss_decision",
                                  help="Did the toss winner choose to bat or field?")

st.markdown("<div class='section-label'>🏟️ Venue</div>", unsafe_allow_html=True)
venue = st.selectbox("Venue", venues, key="venue")

st.markdown("</div>", unsafe_allow_html=True)


                                                               
                 
                                                               
predict_btn = st.button("🔮 Predict Winner", use_container_width=True)

if predict_btn:
                                   
    if team1 == team2:
        st.markdown("""
        <div class='warn-box'>
            ⚠️ <strong>Team 1 and Team 2 cannot be the same!</strong>
            Please select two different teams.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

                                                                  
    with st.spinner("Analysing match conditions..."):
        try:
            raw = build_raw_input(
                team1, team2, toss_winner, toss_decision, venue, team_stats
            )
                                                                               
            winner, t1_pct, t2_pct = predict(raw, model, encoders, team1, team2)

        except Exception as e:
            st.markdown(f"""
            <div class='err-box'>
                &#10060; <strong>Prediction failed:</strong> {e}<br>
                Make sure all model files are up to date (run <code>python src/train.py</code>).
            </div>
            """, unsafe_allow_html=True)
            st.stop()

                                                                  
    loser      = team2 if winner == team1 else team1
    conf_pct   = round((t1_pct if winner == team1 else t2_pct) * 100, 1)
    bar_w      = min(int(conf_pct), 100)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">&#127942; Predicted Winner</div>
        <div class="result-winner">{winner}</div>
        <div class="result-confidence">Win Probability: {conf_pct}%</div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{bar_w}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

                                                                  
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>&#9878; Head-to-Head Win Probability</div>",
                unsafe_allow_html=True)

    t1_pct_show = round(t1_pct * 100, 1)
    t2_pct_show = round(t2_pct * 100, 1)

                                                           
    t1_color  = "#f9d423, #ff4e50" if winner == team1 else "#4facfe, #00f2fe"
    t2_color  = "#f9d423, #ff4e50" if winner == team2 else "#4facfe, #00f2fe"
    t1_bold   = "font-weight:800; color:#ffffff;" if winner == team1 else "color:#c8d0f0;"
    t2_bold   = "font-weight:800; color:#ffffff;" if winner == team2 else "color:#c8d0f0;"
    t1_trophy = "&#127942; " if winner == team1 else ""
    t2_trophy = "&#127942; " if winner == team2 else ""

    h2h_html = f"""
    <div style="margin-bottom:1.2rem;">
        <!-- Team 1 row -->
        <div class="prob-row">
            <div class="prob-team" style="{t1_bold}">{t1_trophy}{team1}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{int(t1_pct_show)}%; background:linear-gradient(90deg,{t1_color});"></div>
            </div>
            <div class="prob-pct">{t1_pct_show}%</div>
        </div>
        <!-- Team 2 row -->
        <div class="prob-row">
            <div class="prob-team" style="{t2_bold}">{t2_trophy}{team2}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{int(t2_pct_show)}%; background:linear-gradient(90deg,{t2_color});"></div>
            </div>
            <div class="prob-pct">{t2_pct_show}%</div>
        </div>
    </div>
    """
    st.markdown(h2h_html, unsafe_allow_html=True)

                                                                  
    with st.expander("&#128203; Match Details Summary"):
        summary_data = {
            "Field" : ["Team 1", "Team 2", "Toss Winner", "Toss Decision",
                       "Venue", "Predicted Winner", "Confidence"],
            "Value" : [team1, team2, toss_winner, toss_decision,
                       venue, winner, f"{conf_pct}%"],
        }
        st.table(pd.DataFrame(summary_data))


                                                               
         
                                                               
st.markdown("""
<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin: 2rem 0 1rem;">
<p style="text-align:center; color:#606880; font-size:0.82rem;">
    Cricket Match Winner Prediction &nbsp;·&nbsp; Gradient Boosting Model
    &nbsp;·&nbsp; Built with Streamlit
</p>
""", unsafe_allow_html=True)