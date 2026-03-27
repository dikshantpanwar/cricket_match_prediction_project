# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Fix Windows console encoding so all print() calls work without crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -------------------------------------------------------------
#  PATHS  (works no matter where you run the script from)
# -------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "matches.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
#  COLUMNS
#
#  CAT_COLS  : go through LabelEncoder (strings -> integers)
#  FEATURE_COLS: everything the model sees (cat + numeric)
#  Numeric features are added BEFORE encoding and do NOT
#  pass through LabelEncoder.
# -------------------------------------------------------------
CAT_COLS = ["team1", "team2", "toss_winner", "toss_decision", "venue"]

FEATURE_COLS = [
    # -- categorical (label-encoded) --
    "team1", "team2", "toss_winner", "toss_decision", "venue",
    # -- numeric (passed through as-is) --
    "team1_win_rate",   # team1 overall win rate  [0.0, 1.0]
    "team2_win_rate",   # team2 overall win rate  [0.0, 1.0]
    "win_rate_diff",    # team1_win_rate - team2_win_rate  [-1.0, 1.0]
    "h2h_win_rate",     # team1 wins / total h2h matches   [0.0, 1.0]
    "home_adv",         # 1 if team1 plays at home, else 0
]

TARGET_COL = "winner"

# Global dict that holds one LabelEncoder per column
encoders = {}

# Global dict for precomputed numeric stats (saved/loaded like encoders)
team_stats = {}

# -------------------------------------------------------------
#  HOME GROUND MAPPING -- team -> primary home venue
# -------------------------------------------------------------
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


# -------------------------------------------------------------
#  STEP 1 - Load & clean
# -------------------------------------------------------------
def load_and_clean(path=DATA_PATH):
    """
    Read matches.csv, keep only the columns we need,
    drop rows where 'winner' is null (abandoned / no-result matches).
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")

    df = df[["team1", "team2", "toss_winner", "toss_decision", "venue", TARGET_COL]].copy()

    before = len(df)
    df.dropna(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} null rows  ->  {len(df)} rows remain")

    return df


# -------------------------------------------------------------
#  STEP 1b - Compute team stats (call ONCE on the full dataset)
#
#  We compute these stats from ALL available data, then save them
#  to disk so predict.py can load and reuse the exact same values.
#  This keeps training and prediction perfectly consistent.
# -------------------------------------------------------------
def compute_team_stats(df):
    """
    Compute and return a stats dict with:

        win_rates : { team_name: overall_win_rate }
                    How often each team wins across all its matches.

        h2h_wins  : { (team_a, team_b): wins_by_team_a }
                    Head-to-head raw win counts.

        h2h_total : { (team_a, team_b): total_h2h_matches }
                    Total matches between each pair (both orderings
                    share the same total, stored under both keys).

    These are float values in [0, 1] so they're meaningful and
    comparable across teams that have played very different numbers
    of matches.
    """
    global team_stats

    # ── Overall win rate per team ───────────────────────────────
    all_teams = set(df["team1"].tolist() + df["team2"].tolist())
    match_count = {t: 0 for t in all_teams}
    win_count   = {t: 0 for t in all_teams}

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        match_count[t1] = match_count.get(t1, 0) + 1
        match_count[t2] = match_count.get(t2, 0) + 1
        if winner in match_count:
            win_count[winner] = win_count.get(winner, 0) + 1

    win_rates = {
        t: win_count[t] / match_count[t] if match_count[t] > 0 else 0.5
        for t in all_teams
    }

    # ── Head-to-head win counts & total matches ─────────────────
    h2h_wins  = {}   # (winner_team, loser_team) -> count
    h2h_total = {}   # (team_a, team_b) -> total matches (unordered pair stored both ways)

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        # Store total under both orderings so lookup is direction-agnostic
        for pair in [(t1, t2), (t2, t1)]:
            h2h_total[pair] = h2h_total.get(pair, 0) + 1

        if winner == t1:
            h2h_wins[(t1, t2)] = h2h_wins.get((t1, t2), 0) + 1
        elif winner == t2:
            h2h_wins[(t2, t1)] = h2h_wins.get((t2, t1), 0) + 1

    team_stats = {
        "win_rates" : win_rates,
        "h2h_wins"  : h2h_wins,
        "h2h_total" : h2h_total,
    }

    print(f"[INFO] Win rates computed for {len(win_rates)} teams")
    print(f"[INFO] H2H records computed for {len(h2h_total)//2} unique matchups")
    return team_stats


# -------------------------------------------------------------
#  STEP 1c - Add engineered features using precomputed stats
# -------------------------------------------------------------
def add_features(df, stats=None):
    """
    Add all numeric feature columns to the cleaned DataFrame.
    Must be called AFTER compute_team_stats() and BEFORE encode().

    If 'stats' is None, uses the global team_stats dict
    (which is populated either by compute_team_stats() or load_team_stats()).

    Features added
    --------------
    team1_win_rate : team1's overall historical win rate
    team2_win_rate : team2's overall historical win rate
    win_rate_diff  : team1_win_rate - team2_win_rate
                     Positive = team1 historically stronger
    h2h_win_rate   : team1's win rate specifically against team2
                     Uses 0.5 (neutral) when no prior h2h history exists
    home_adv       : 1 if venue == team1's home ground, else 0
    """
    if stats is None:
        stats = team_stats   # use global

    win_rates = stats["win_rates"]
    h2h_wins  = stats["h2h_wins"]
    h2h_total = stats["h2h_total"]

    df = df.copy()

    # ── Feature 1 & 2: Overall win rates ───────────────────────
    df["team1_win_rate"] = df["team1"].map(lambda t: win_rates.get(t, 0.5))
    df["team2_win_rate"] = df["team2"].map(lambda t: win_rates.get(t, 0.5))

    # ── Feature 3: Win-rate difference ─────────────────────────
    df["win_rate_diff"] = df["team1_win_rate"] - df["team2_win_rate"]

    # ── Feature 4: Head-to-head win rate ───────────────────────
    # h2h_wins[(t1,t2)] = how many times t1 beat t2
    # h2h_total[(t1,t2)] = total matches between them
    # Use 0.5 as default when teams have never met
    def h2h_rate(row):
        t1, t2 = row["team1"], row["team2"]
        total = h2h_total.get((t1, t2), 0)
        if total == 0:
            return 0.5
        wins = h2h_wins.get((t1, t2), 0)
        return wins / total

    df["h2h_win_rate"] = df.apply(h2h_rate, axis=1)

    # ── Feature 5: Home ground advantage ───────────────────────
    df["home_adv"] = df.apply(
        lambda r: 1 if HOME_VENUES.get(r["team1"]) == r["venue"] else 0,
        axis=1
    )

    # -- Summary stats
    print(f"[INFO] team1_win_rate range  : [{df['team1_win_rate'].min():.3f}, {df['team1_win_rate'].max():.3f}]")
    print(f"[INFO] win_rate_diff range   : [{df['win_rate_diff'].min():.3f}, {df['win_rate_diff'].max():.3f}]")
    print(f"[INFO] h2h_win_rate range    : [{df['h2h_win_rate'].min():.3f}, {df['h2h_win_rate'].max():.3f}]")
    print(f"[INFO] home_adv (team1 home) : {df['home_adv'].sum()} / {len(df)} rows")

    return df


# -------------------------------------------------------------
#  Save / load team stats (so predict.py uses identical values)
# -------------------------------------------------------------
def save_team_stats():
    path = os.path.join(MODELS_DIR, "team_stats.pkl")
    joblib.dump(team_stats, path)
    print(f"[INFO] Team stats saved  ->  {path}")


def load_team_stats():
    global team_stats
    path = os.path.join(MODELS_DIR, "team_stats.pkl")
    team_stats = joblib.load(path)
    print(f"[INFO] Team stats loaded from {path}")
    return team_stats


# -------------------------------------------------------------
#  STEP 2 - Encode categories -> integers
#  Only CAT_COLS are label-encoded; all numeric features pass through.
# -------------------------------------------------------------
def encode(df, fit=True):
    """
    Convert categorical columns to integers.

    fit=True  -> fit new LabelEncoders (call during training)
    fit=False -> use already-saved encoders (call during prediction)

    All numeric columns (win_rates, h2h, home_adv) are NOT touched.
    Returns encoded DataFrame; original df is NOT modified.
    """
    df_enc = df.copy()

    for col in CAT_COLS + [TARGET_COL]:
        if fit:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df_enc[col] = df_enc[col].astype(str).map(
                lambda val, le=le: (
                    le.transform([val])[0]
                    if val in le.classes_
                    else -1
                )
            )
    return df_enc


# -------------------------------------------------------------
#  STEP 3 - Split into X (features) and y (target)
# -------------------------------------------------------------
def get_features_and_target(df_encoded):
    X = df_encoded[FEATURE_COLS]
    y = df_encoded[TARGET_COL]
    return X, y


# -------------------------------------------------------------
#  Save / load label encoders
# -------------------------------------------------------------
def save_encoders():
    path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    joblib.dump(encoders, path)
    print(f"[INFO] Encoders saved  ->  {path}")


def load_encoders():
    global encoders
    path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    encoders = joblib.load(path)
    print(f"[INFO] Encoders loaded from {path}")


# -------------------------------------------------------------
#  Save processed CSV  (pre-encoding, with all new columns)
# -------------------------------------------------------------
def save_processed_csv(df_clean):
    """Save cleaned + feature-engineered data for inspection."""
    out_path = os.path.join(BASE_DIR, "data", "processed.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"[INFO] Processed CSV saved  ->  {out_path}")


# -------------------------------------------------------------
#  Quick self-test  ->  python src/preprocess.py
# -------------------------------------------------------------
if __name__ == "__main__":
    df_clean    = load_and_clean()
    stats       = compute_team_stats(df_clean)   # compute stats first
    df_featured = add_features(df_clean, stats)  # use stats to add features
    save_processed_csv(df_featured)
    save_team_stats()                            # save for predict.py to load

    df_encoded = encode(df_featured, fit=True)
    save_encoders()

    X, y = get_features_and_target(df_encoded)

    print("\n-- First 5 encoded rows --")
    print(df_encoded[FEATURE_COLS].head().to_string())
    print(f"\nX shape : {X.shape}   (rows, features)")
    print(f"y unique: {y.nunique()} different winners")
    print("\n[OK] preprocess.py")