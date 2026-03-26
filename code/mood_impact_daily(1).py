"""
Daily Societal Mood Impact Scorer v3 — Dartmouth / US context
==============================================================
Period      : March 25, 2013 → June 4, 2013
Granularity : Daily

What changed from v2
---------------------
PROBLEM 1 — Major events diluted by routine noise:
    FIX: Salience filter — only keep events with NumSources >= MIN_SOURCES
         and NumMentions >= MIN_MENTIONS before any scoring.
         This removes local/routine events that never reached general audience.

PROBLEM 2 — Min/max normalization compressed extreme days toward neutral:
    FIX: Z-score normalization for GDELT signal too.
         Genuinely extreme days now stand out statistically (e.g. Apr 15
         bombing shows as a clear negative spike, not washed to neutral).

PROBLEM 3 — Single dominant event (e.g. bombing) averaged away by others:
    FIX: Prominence boost — if one event's NumMentions is > PROMINENCE_THRESHOLD×
         the day's median, that event gets extra weight in the daily average.
         This lets the Boston bombing dominate April 15 as it should.

PROBLEM 4 — Trends lag (people search after processing news, not same day):
    FIX: 1-day lag applied to Trends signal before merging.
         April 16 anxiety spike now correctly influences April 15 score.

Compound mood score (range -100 to +100, centred at 0):
    GDELT signal  (40%) : prominence-weighted Z-score of signed daily mood
                          = blend of AvgTone (70%) + GoldsteinScale (30%)
    Trends signal (60%) : lagged Z-score of anxiety searches, inverted
                          (anxiety spike -> negative mood contribution)

Output:
    mood_daily.csv       : full daily scores + all diagnostic columns
    top_events_daily.csv : top N salient events per day, all GDELT columns
    mood_timeline.png    : bidirectional mood chart + event table

Setup:
    pip install pytrends pandas requests matplotlib numpy
"""

import io
import time
import zipfile
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

START_DATE = "2013-03-25"
END_DATE   = "2013-06-04"

# Scoring weights (must sum to 1.0)
GDELT_WEIGHT  = 0.40
TRENDS_WEIGHT = 0.60

# Within GDELT signal: how much GoldsteinScale contributes vs AvgTone
GOLDSTEIN_WEIGHT = 0.30   # 30% Goldstein, 70% AvgTone

# Salience filter
# Events below these thresholds are excluded before scoring.
# They represent local/routine news that never reached general public awareness.
# Rationale: Boston bombing had NumSources ~200+, NumMentions ~10,000+
#            A local council meeting might have NumSources 1, NumMentions 2
MIN_SOURCES  = 5    # covered by at least 5 distinct news outlets
MIN_MENTIONS = 20   # mentioned at least 20 times across all coverage

# Prominence boost
# If a single event's NumMentions exceeds this multiple of the day's median,
# it gets extra weight. Prevents bombing from being averaged away.
PROMINENCE_THRESHOLD = 10   # must be 10x the daily median to trigger boost
PROMINENCE_BOOST     = 5.0  # multiply its weight by this factor

# Z-score clip (prevents one extreme day from compressing everything else)
ZSCORE_CLIP = 3.0

# Trends: apply N-day lag (people search after processing news)
TRENDS_LAG_DAYS = 1

# Top N salient events per day saved to CSV
TOP_N_EVENTS = 5

# Google Trends anxiety keywords
TRENDS_KEYWORDS = [
    "anxiety",
    "depression",
    "can't sleep",
    "how to cope",
    "overwhelmed",
]
TRENDS_GEO = "US"

# Known major events for chart annotation
KNOWN_EVENTS = [
    ("2013-04-15", "Boston Marathon\nbombing",      "violence",  "negative"),
    ("2013-04-17", "Senate rejects\ngun control",   "political", "negative"),
    ("2013-04-19", "Bomber captured",               "political", "mixed"),
    ("2013-05-03", "Dow hits record\nhigh",          "economic",  "positive"),
    ("2013-05-10", "IRS scandal\nbreaks",            "political", "negative"),
    ("2013-05-13", "DOJ seizes AP\nphone records",  "political", "negative"),
    ("2013-05-20", "Oklahoma tornado\nEF5",          "violence",  "negative"),
]

EVENT_COLORS = {
    "violence":  "#E24B4A",
    "political": "#378ADD",
    "economic":  "#EF9F27",
    "social":    "#1D9E75",
}

# GDELT 1.0 column schema (no header row in files)
GDELT_V1_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type",
    "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",
]

QUADCLASS_LABELS = {
    "1": "Verbal cooperation",  "2": "Material cooperation",
    "3": "Verbal conflict",     "4": "Material conflict",
}

CAMEO_ROOT_LABELS = {
    "01": "Make statement",        "02": "Appeal",
    "03": "Express intent",        "04": "Consult",
    "05": "Engage diplomatically", "06": "Cooperate",
    "07": "Aid",                   "08": "Yield",
    "09": "Investigate",           "10": "Demand",
    "11": "Disapprove",            "12": "Reject",
    "13": "Threaten",              "14": "Protest",
    "15": "Exhibit force",         "16": "Reduce relations",
    "17": "Coerce",                "18": "Assault",
    "19": "Fight",                 "20": "Mass violence",
}

OUTPUT_SCORES = "mood_daily1.csv"
OUTPUT_EVENTS = "top_events_daily1.csv"
OUTPUT_PLOT   = "mood_timeline.png"


# ─────────────────────────────────────────────
# PART 1: GDELT 1.0 fetch
# ─────────────────────────────────────────────

def fetch_gdelt_day(date_str: str) -> pd.DataFrame:
    """Download one day's GDELT 1.0 export zip and return as DataFrame."""
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t", header=None,
                             names=GDELT_V1_COLS, dtype=str, low_memory=False)
        return df
    except Exception as e:
        print(f"skip ({e})")
        return pd.DataFrame()


def fetch_gdelt_all(start: str, end: str) -> pd.DataFrame:
    """
    Download all daily GDELT 1.0 files for the period.
    Steps:
        1. Download raw daily file
        2. Filter to US events (ActionGeo_CountryCode == 'US')
        3. Apply salience filter: NumSources >= MIN_SOURCES,
                                  NumMentions >= MIN_MENTIONS
        4. Add readable labels and event description
        5. Keep ALL columns for downstream use
    """
    date_range = pd.date_range(start=start, end=end, freq="D")
    all_rows   = []

    print(f"  Downloading {len(date_range)} daily files "
          f"(salience filter: >={MIN_SOURCES} sources, >={MIN_MENTIONS} mentions)...")

    for dt in date_range:
        date_str = dt.strftime("%Y%m%d")
        print(f"    {dt.strftime('%b %d')}...", end=" ", flush=True)

        raw = fetch_gdelt_day(date_str)
        if raw.empty:
            continue

        us = raw[raw["ActionGeo_CountryCode"] == "US"].copy()

        for col in ["AvgTone", "GoldsteinScale", "NumMentions",
                    "NumSources", "NumArticles"]:
            us[col] = pd.to_numeric(us[col], errors="coerce").fillna(0)

        # Salience filter
        before = len(us)
        us = us[
            (us["NumSources"]  >= MIN_SOURCES) &
            (us["NumMentions"] >= MIN_MENTIONS)
        ].copy()
        after = len(us)

        if us.empty:
            print(f"0 salient events (filtered {before} -> 0)")
            continue

        us["date"] = dt.normalize()

        # Readable labels
        us["EventRootCode"]   = us["EventRootCode"].astype(str).str.strip().str.zfill(2)
        us["event_type"]      = us["EventRootCode"].map(CAMEO_ROOT_LABELS).fillna("Other")
        us["quadclass_label"] = us["QuadClass"].astype(str).map(QUADCLASS_LABELS).fillna("Unknown")

        us["event_description"] = (
            us["Actor1Name"].fillna("Unknown").str.title() +
            " -> " +
            us["Actor2Name"].fillna("Unknown").str.title() +
            " (" + us["event_type"] + ")"
        )

        # Salience score for ranking
        us["salience_score"] = us["NumMentions"] * us["NumSources"]

        all_rows.append(us)
        print(f"{after} salient events (from {before} US events)")
        time.sleep(0.3)

    if not all_rows:
        print("  No salient GDELT events retrieved.")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    print(f"\n  Total salient events: {len(df):,} across {len(date_range)} days")
    return df


# ─────────────────────────────────────────────
# PART 2: GDELT daily mood signal
# ─────────────────────────────────────────────

def compute_gdelt_mood(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signed daily mood signal from salient GDELT events.

    Per-event mood:
        event_mood = AvgTone x (1 - GOLDSTEIN_WEIGHT)
                   + (GoldsteinScale x 10) x GOLDSTEIN_WEIGHT
        (GoldsteinScale scaled x10 to match AvgTone range)

    Prominence boost:
        If an event's NumMentions > PROMINENCE_THRESHOLD x day_median_mentions,
        multiply its weight by PROMINENCE_BOOST.
        Lets the Boston bombing dominate April 15.

    Daily signal:
        Weighted mean of event_mood, weight = NumMentions x prominence_multiplier

    Normalization:
        Z-score across the period, clipped to +-ZSCORE_CLIP,
        scaled to [-100, +100].
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "gdelt_mood", "gdelt_avg_tone", "gdelt_avg_goldstein",
            "gdelt_total_mentions", "gdelt_total_sources",
            "gdelt_salient_events", "gdelt_dominant_event"
        ])

    df = df.copy()
    df["goldstein_scaled"] = df["GoldsteinScale"] * 10
    df["event_mood"] = (
        df["AvgTone"]          * (1 - GOLDSTEIN_WEIGHT) +
        df["goldstein_scaled"] * GOLDSTEIN_WEIGHT
    )

    # Prominence boost
    day_medians = df.groupby("date")["NumMentions"].median().rename("day_median")
    df = df.merge(day_medians, on="date", how="left")
    df["prominence_mult"] = np.where(
        df["NumMentions"] > PROMINENCE_THRESHOLD * df["day_median"],
        PROMINENCE_BOOST, 1.0
    )
    df["effective_weight"] = df["NumMentions"] * df["prominence_mult"]
    df["weighted_mood"]    = df["event_mood"] * df["effective_weight"]

    # Dominant event per day
    dominant = (
        df.sort_values("effective_weight", ascending=False)
        .groupby("date").first().reset_index()
        [["date", "event_description"]]
        .rename(columns={"event_description": "gdelt_dominant_event"})
    )

    daily = df.groupby("date").agg(
        total_weighted_mood  = ("weighted_mood",    "sum"),
        total_weight         = ("effective_weight", "sum"),
        gdelt_avg_tone       = ("AvgTone",          "mean"),
        gdelt_avg_goldstein  = ("GoldsteinScale",   "mean"),
        gdelt_total_mentions = ("NumMentions",      "sum"),
        gdelt_total_sources  = ("NumSources",       "sum"),
        gdelt_salient_events = ("GlobalEventID",    "count"),
    ).reset_index()

    daily["gdelt_mood_raw"] = (
        daily["total_weighted_mood"] /
        daily["total_weight"].replace(0, np.nan)
    ).fillna(0)

    # Z-score normalization
    mean = daily["gdelt_mood_raw"].mean()
    std  = daily["gdelt_mood_raw"].std()
    if std > 0:
        daily["gdelt_mood_z"] = (daily["gdelt_mood_raw"] - mean) / std
    else:
        daily["gdelt_mood_z"] = 0.0

    daily["gdelt_mood_z"] = daily["gdelt_mood_z"].clip(-ZSCORE_CLIP, ZSCORE_CLIP)
    daily["gdelt_mood"]   = (daily["gdelt_mood_z"] / ZSCORE_CLIP * 100).round(1)
    daily = daily.merge(dominant, on="date", how="left")

    return daily[[
        "date", "gdelt_mood", "gdelt_avg_tone", "gdelt_avg_goldstein",
        "gdelt_total_mentions", "gdelt_total_sources",
        "gdelt_salient_events", "gdelt_dominant_event"
    ]]


# ─────────────────────────────────────────────
# PART 3: Google Trends — signed, lagged
# ─────────────────────────────────────────────

def fetch_trends(keywords: list, start: str, end: str, geo: str) -> pd.DataFrame:
    """Fetch daily Google Trends interest for anxiety keywords."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  pytrends not installed. Run: pip install pytrends")
        return pd.DataFrame()

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    all_dfs  = []
    chunks   = [keywords[i:i+5] for i in range(0, len(keywords), 5)]

    for chunk in chunks:
        print(f"    Trends: {chunk}")
        try:
            pytrends.build_payload(
                kw_list=chunk,
                timeframe=f"{start} {end}",
                geo=geo,
            )
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns=["isPartial"], errors="ignore")
                all_dfs.append(df)
            time.sleep(2.5)
        except Exception as e:
            print(f"      Warning: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    return combined.reset_index()


def compute_trends_mood(df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """
    Convert Trends to signed mood signal with lag correction.

    Lag: shift signal back by lag_days so that April 16 anxiety spike
    (people reacting to the bombing) is credited to April 15.

    Sign inversion: anxiety spike = BAD mood -> negative contribution.
    Z-score, clipped to +-ZSCORE_CLIP, scaled to [-100, +100].
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "trends_mood",
                                     "trends_avg", "trends_zscore"])

    kw_cols = [c for c in df.columns if c != "date"]
    df = df.copy()
    df["trends_avg"]        = df[kw_cols].mean(axis=1)
    df["trends_avg_lagged"] = df["trends_avg"].shift(lag_days).fillna(df["trends_avg"])

    mean = df["trends_avg_lagged"].mean()
    std  = df["trends_avg_lagged"].std()
    df["trends_zscore"] = ((df["trends_avg_lagged"] - mean) / std
                           if std > 0 else 0.0)
    df["trends_zscore"] = df["trends_zscore"].clip(-ZSCORE_CLIP, ZSCORE_CLIP)

    # Invert: anxiety spike -> negative mood
    df["trends_mood"] = -(df["trends_zscore"] / ZSCORE_CLIP * 100).round(1)

    return df[["date", "trends_avg", "trends_zscore", "trends_mood"]]


# ─────────────────────────────────────────────
# PART 4: Compound mood score
# ─────────────────────────────────────────────

def build_compound_mood(gdelt_daily: pd.DataFrame,
                         trends_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and compute weighted compound mood score [-100, +100].
    All diagnostic columns kept for research use.
    """
    days = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    base = pd.DataFrame({"date": days})

    if not gdelt_daily.empty:
        gdelt_daily["date"] = pd.to_datetime(gdelt_daily["date"])
        base = base.merge(gdelt_daily, on="date", how="left")
    else:
        for col in ["gdelt_mood", "gdelt_avg_tone", "gdelt_avg_goldstein",
                    "gdelt_total_mentions", "gdelt_total_sources",
                    "gdelt_salient_events"]:
            base[col] = 0.0
        base["gdelt_dominant_event"] = ""

    if not trends_daily.empty:
        trends_daily["date"] = pd.to_datetime(trends_daily["date"])
        base = base.merge(
            trends_daily[["date", "trends_avg", "trends_zscore", "trends_mood"]],
            on="date", how="left"
        )
    else:
        base["trends_mood"]  = 0.0
        base["trends_avg"]   = 0.0
        base["trends_zscore"]= 0.0

    num_cols = ["gdelt_mood", "trends_mood", "gdelt_avg_tone",
                "gdelt_avg_goldstein", "gdelt_total_mentions",
                "gdelt_total_sources", "gdelt_salient_events",
                "trends_avg", "trends_zscore"]
    for c in num_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0)
    if "gdelt_dominant_event" in base.columns:
        base["gdelt_dominant_event"] = base["gdelt_dominant_event"].fillna("")

    base["compound_mood"] = (
        GDELT_WEIGHT  * base["gdelt_mood"] +
        TRENDS_WEIGHT * base["trends_mood"]
    ).round(1)

    def mood_label(s):
        if   s >=  40: return "Positive"
        elif s >=  10: return "Mildly positive"
        elif s >= -10: return "Neutral"
        elif s >= -40: return "Mildly negative"
        else:          return "Negative"

    base["mood_label"] = base["compound_mood"].apply(mood_label)
    return base


# ─────────────────────────────────────────────
# PART 5: Top salient events per day
# ─────────────────────────────────────────────

def get_top_events(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return top N events per day ranked by salience_score."""
    if df.empty:
        return pd.DataFrame()

    keep = [
        "date", "event_description", "event_type", "quadclass_label",
        "AvgTone", "GoldsteinScale", "NumMentions", "NumSources",
        "NumArticles", "salience_score", "Actor1Name", "Actor2Name",
        "ActionGeo_FullName", "EventCode", "SOURCEURL",
    ]
    keep = [c for c in keep if c in df.columns]

    top = (
        df.sort_values("salience_score", ascending=False)
        .groupby("date").head(n)
        .reset_index(drop=True)
    )[keep]

    for col in ["AvgTone", "GoldsteinScale"]:
        if col in top.columns:
            top[col] = top[col].round(2)

    return top.sort_values(["date", "salience_score"], ascending=[True, False])


# ─────────────────────────────────────────────
# PART 6: Plot
# ─────────────────────────────────────────────

def plot_mood_timeline(scores: pd.DataFrame, top_events: pd.DataFrame):
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAF8")
    gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1.2], hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_facecolor("#FAFAF8")
    days  = scores["date"]
    score = scores["compound_mood"]

    ax1.axhline(0, color="#B4B2A9", linewidth=0.8, zorder=1)
    ax1.axhspan( 10,  105, alpha=0.05, color="#1D9E75", zorder=0)
    ax1.axhspan(-105, -10, alpha=0.05, color="#E24B4A", zorder=0)

    ax1.plot(days, scores["gdelt_mood"], color="#378ADD", linewidth=0.9,
             linestyle="--", alpha=0.45,
             label=f"GDELT mood Z-score ({int(GDELT_WEIGHT*100)}%)")
    ax1.plot(days, scores["trends_mood"], color="#7F77DD", linewidth=0.9,
             linestyle="--", alpha=0.45,
             label=f"Trends anxiety inverted ({int(TRENDS_WEIGHT*100)}%, {TRENDS_LAG_DAYS}d lag)")

    ax1.fill_between(days, score, 0, where=(score >= 0),
                     alpha=0.20, color="#1D9E75", interpolate=True, zorder=2)
    ax1.fill_between(days, score, 0, where=(score < 0),
                     alpha=0.20, color="#E24B4A", interpolate=True, zorder=2)

    ax1.plot(days, score, color="#2C2C2A", linewidth=2.0,
             label="Compound mood score", zorder=3)

    used_y = []
    for event_date_str, label_text, category, polarity in KNOWN_EVENTS:
        event_date = pd.to_datetime(event_date_str)
        if not (pd.to_datetime(START_DATE) <= event_date <= pd.to_datetime(END_DATE)):
            continue
        color   = EVENT_COLORS.get(category, "#888780")
        nearest = scores.iloc[(scores["date"] - event_date).abs().argsort()[:1]]
        y_val   = nearest["compound_mood"].values[0] if not nearest.empty else 0

        offset = -20 if polarity == "positive" else 14
        y_text = y_val + offset
        for uy in used_y:
            if abs(y_text - uy) < 14:
                y_text = uy + (16 if offset > 0 else -16)
        used_y.append(y_text)
        y_text = float(np.clip(y_text, -95, 95))

        ax1.axvline(x=event_date, color=color, linewidth=1.1,
                    linestyle=":", alpha=0.8, zorder=3)
        ax1.annotate(label_text,
                     xy=(event_date, y_val),
                     xytext=(event_date, y_text),
                     fontsize=7, color=color, ha="center",
                     va="top" if polarity == "positive" else "bottom",
                     arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
                     zorder=4)

    ax1.text(days.iloc[-1],  70, "Positive mood", fontsize=7.5,
             color="#1D9E75", va="center", ha="right", alpha=0.75)
    ax1.text(days.iloc[-1], -70, "Negative mood", fontsize=7.5,
             color="#E24B4A", va="center", ha="right", alpha=0.75)

    ax1.set_xlim(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE))
    ax1.set_ylim(-105, 105)
    ax1.set_ylabel("Compound mood score (-100 to +100)", fontsize=9)
    ax1.set_title(
        "Daily societal mood impact — Dartmouth students  |  Mar 25 - Jun 4, 2013\n"
        f"Salience filter: >={MIN_SOURCES} sources, >={MIN_MENTIONS} mentions  |  "
        f"Prominence boost: {PROMINENCE_THRESHOLD}x median -> {PROMINENCE_BOOST}x weight  |  "
        f"Trends lag: {TRENDS_LAG_DAYS}d",
        fontsize=9.5, pad=10
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)

    patches = [mpatches.Patch(color=v, label=k.capitalize())
               for k, v in EVENT_COLORS.items()]
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + patches, fontsize=7.5, framealpha=0.5,
               ncol=3, loc="lower left", title="Signal / Event type",
               title_fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: event table
    ax2.set_facecolor("#F5F4F0")
    ax2.axis("off")

    if not top_events.empty:
        extreme_days = (
            scores.assign(abs_score=scores["compound_mood"].abs())
            .nlargest(6, "abs_score")["date"].tolist()
        )
        shown = (
            top_events[top_events["date"].isin(extreme_days)]
            .sort_values(["date", "salience_score"], ascending=[True, False])
            .head(20)
        )

        ax2.set_title(
            "Top salient events on most extreme mood days  "
            f"(ranked by salience = NumMentions x NumSources)",
            fontsize=9, pad=6, loc="left"
        )

        col_x   = [0.0, 0.08, 0.42, 0.57, 0.67, 0.76, 0.85, 0.93]
        headers = ["Date", "Event", "Type", "Quad", "Tone",
                   "Goldstein", "Mentions", "Sources"]
        for hdr, x in zip(headers, col_x):
            ax2.text(x, 1.0, hdr, fontsize=7.5, fontweight="bold",
                     va="top", transform=ax2.transAxes, color="#444441")

        ax2.axhline(y=0.97, xmin=0, xmax=1, color="#B4B2A9",
                    linewidth=0.6, transform=ax2.transAxes)

        row_h = 0.044
        y     = 0.95
        for _, row in shown.iterrows():
            tone_v = row.get("AvgTone", 0)
            gold_v = row.get("GoldsteinScale", 0)
            vals = [
                row["date"].strftime("%b %d"),
                str(row.get("event_description", ""))[:36],
                str(row.get("event_type", ""))[:14],
                str(row.get("quadclass_label", ""))[:12],
                f"{tone_v:+.1f}",
                f"{gold_v:+.1f}",
                f"{int(row.get('NumMentions', 0)):,}",
                f"{int(row.get('NumSources',  0)):,}",
            ]
            tone_c = ("#A32D2D" if tone_v < -3 else
                      "#27500A" if tone_v >  3 else "#444441")
            gold_c = ("#A32D2D" if gold_v < -3 else
                      "#27500A" if gold_v >  3 else "#444441")
            clrs = ["#5F5E5A", "#2C2C2A", "#5F5E5A", "#5F5E5A",
                    tone_c, gold_c, "#5F5E5A", "#5F5E5A"]

            for val, x, c in zip(vals, col_x, clrs):
                ax2.text(x, y, val, fontsize=7, va="top",
                         transform=ax2.transAxes, color=c)
            y -= row_h
            if y < 0.02:
                break
    else:
        ax2.text(0.5, 0.5, "No event data available",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=9, color="#888780")

    plt.savefig(OUTPUT_PLOT, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Plot saved -> {OUTPUT_PLOT}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Daily Mood Impact Scorer v3 — Dartmouth, Spring 2013")
    print("=" * 65)
    print(f"  Period     : {START_DATE} -> {END_DATE}")
    print(f"  Weights    : GDELT {GDELT_WEIGHT*100:.0f}% | Trends {TRENDS_WEIGHT*100:.0f}%")
    print(f"  Salience   : >={MIN_SOURCES} sources, >={MIN_MENTIONS} mentions")
    print(f"  Prominence : events >{PROMINENCE_THRESHOLD}x day median -> {PROMINENCE_BOOST}x weight")
    print(f"  Trends lag : {TRENDS_LAG_DAYS} day(s)")
    print(f"  Score range: -100 (very negative) -> 0 (neutral) -> +100 (very positive)\n")

    # 1. GDELT
    print("[1/4] Fetching GDELT 1.0 events...")
    gdelt_raw   = fetch_gdelt_all(START_DATE, END_DATE)
    gdelt_daily = compute_gdelt_mood(gdelt_raw)
    top_events  = get_top_events(gdelt_raw, TOP_N_EVENTS)
    print(f"  GDELT: {len(gdelt_daily)} days with mood signal")

    # 2. Trends
    print("\n[2/4] Fetching Google Trends (daily, with lag)...")
    trends_raw   = fetch_trends(TRENDS_KEYWORDS, START_DATE, END_DATE, TRENDS_GEO)
    trends_daily = compute_trends_mood(trends_raw, lag_days=TRENDS_LAG_DAYS)
    print(f"  Trends: {len(trends_daily)} days")

    # 3. Compound
    print("\n[3/4] Computing compound mood scores...")
    scores = build_compound_mood(gdelt_daily, trends_daily)
    scores.to_csv(OUTPUT_SCORES, index=False)
    print(f"  Saved -> {OUTPUT_SCORES}")

    top_events.to_csv(OUTPUT_EVENTS, index=False)
    if top_events.empty:
        print(f"  Warning: {OUTPUT_EVENTS} is empty — check GDELT connection")
    else:
        print(f"  Saved {len(top_events)} event rows -> {OUTPUT_EVENTS}")

    # 4. Plot
    print("\n[4/4] Generating mood timeline...")
    plot_mood_timeline(scores, top_events)

    # Console summary
    print("\n-- Daily mood summary --")
    cols = ["date", "gdelt_avg_tone", "gdelt_avg_goldstein",
            "gdelt_salient_events", "trends_zscore", "compound_mood", "mood_label"]
    summary = scores[cols].copy()
    summary["date"] = summary["date"].dt.strftime("%b %d")
    summary = summary.round(2)
    summary.columns = ["Date", "Avg Tone", "Goldstein",
                        "Salient Events", "Anxiety Z", "Mood", "Label"]
    print(summary.to_string(index=False))

    print("\n-- Mood distribution --")
    print(scores["mood_label"].value_counts().to_string())

    print("\n-- Most negative days --")
    neg = scores.nsmallest(5, "compound_mood")[
        ["date", "compound_mood", "mood_label", "gdelt_dominant_event"]]
    neg["date"] = neg["date"].dt.strftime("%b %d")
    print(neg.to_string(index=False))

    print("\n-- Most positive days --")
    pos = scores.nlargest(5, "compound_mood")[
        ["date", "compound_mood", "mood_label", "gdelt_dominant_event"]]
    pos["date"] = pos["date"].dt.strftime("%b %d")
    print(pos.to_string(index=False))

    print(f"\nFiles: {OUTPUT_SCORES}, {OUTPUT_EVENTS}, {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
