from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, abort, render_template, request

app = Flask(__name__)

CSV_PATH = Path("phase3_proposal1_AdditionalRules_final_riskprofile_16 Mar 2026.csv")
APP_TITLE = "SAF Driver Profiling - Strides Digital"

RISK_ORDER = ["High Risk", "Medium Risk", "Low Risk"]
RISK_RANK_MAP = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}

FINAL_RISK_COL = "final_model_grp"
ORIGINAL_RISK_COL = "model_grp_bef12"
WEEK_COL = "weekNo"
DRIVER_COL = "driverId"
SCORE_COL = "model_risk_score"
TASK_DAYS_COL = "task_days"
VIDEO_TOTAL_COL = "total_count"
ORIGINAL_RISK_FLAG_COL = "has_original_risk_group"

VIOLATION_COLS = [
    "count_Hard Braking",
    "count_Rapid Acc",
    "count_Speeding",
    "count_No Go Zone",
]

VIOLATION_LABELS = {
    "count_Hard Braking": "Hard Braking",
    "count_Rapid Acc": "Rapid Acceleration",
    "count_Speeding": "Speeding",
    "count_No Go Zone": "No Go Zone",
}

EMPHASIZED_COL = "emphasize_violation_count"
LESS_IMPORTANT_COL = "less_important_violation_count"

VIDEO_VIOLATIONS = [
    "Dozing Off",
    "Yawning",
    "Lane Changing",
    "Lane Discipline",
    "Safe Distancing (Obstacle Proximity)",
    "High-Speed Turning",
    "One-Hand Driving",
    "Safe Distancing (Following Distance)",
    "Moving-Off Drill",
    "Failure to Apply Handbrake",
    "Parking Drill",
    "Mobile Phone Usage",
]
EMPHASIZE_VIOLATIONS = ["Dozing Off", "Lane Changing", "Parking Drill"]
LESS_IMPORTANT_VIOLATIONS = [v for v in VIDEO_VIOLATIONS if v not in EMPHASIZE_VIOLATIONS]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path.resolve()}\n"
            f"Please place the file in the same folder as app.py, or update CSV_PATH."
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        WEEK_COL,
        DRIVER_COL,
        FINAL_RISK_COL,
        ORIGINAL_RISK_COL,
        SCORE_COL,
        TASK_DAYS_COL,
        VIDEO_TOTAL_COL,
        EMPHASIZED_COL,
        LESS_IMPORTANT_COL,
        ORIGINAL_RISK_FLAG_COL,
        *VIOLATION_COLS,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    for optional_col in ["unitId", "role"]:
        if optional_col not in df.columns:
            df[optional_col] = "-"

    for video_col in VIDEO_VIOLATIONS:
        if video_col not in df.columns:
            df[video_col] = 0

    numeric_cols = [
        *VIOLATION_COLS,
        TASK_DAYS_COL,
        VIDEO_TOTAL_COL,
        EMPHASIZED_COL,
        LESS_IMPORTANT_COL,
        SCORE_COL,
        *VIDEO_VIOLATIONS,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[WEEK_COL] = df[WEEK_COL].astype(str)
    df[FINAL_RISK_COL] = normalize_risk_group(df[FINAL_RISK_COL])
    df[ORIGINAL_RISK_COL] = normalize_risk_group(df[ORIGINAL_RISK_COL])
    df[ORIGINAL_RISK_FLAG_COL] = normalize_bool(df[ORIGINAL_RISK_FLAG_COL])

    return df


def normalize_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "y", "t"}
    return series.fillna(False).astype(str).str.strip().str.lower().isin(true_values)


def normalize_risk_group(series: pd.Series) -> pd.Series:
    s = series.fillna("Unknown").astype(str).str.strip()

    replace_map = {
        "High": "High Risk",
        "Medium": "Medium Risk",
        "Low": "Low Risk",
        "High Risk": "High Risk",
        "Medium Risk": "Medium Risk",
        "Low Risk": "Low Risk",
        "Med": "Medium Risk",
    }
    return s.replace(replace_map)


def get_week_options(df: pd.DataFrame) -> list[str]:
    return sorted(df[WEEK_COL].dropna().astype(str).unique().tolist())


def pct_str(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{n / total:.1%}"


def filter_week(df: pd.DataFrame, week_value: str) -> pd.DataFrame:
    return df.loc[df[WEEK_COL] == str(week_value)].copy()


def calc_risk_metrics(df_week: pd.DataFrame, risk_col: str) -> Dict[str, Any]:
    total = df_week[DRIVER_COL].nunique()
    counts = (
        df_week.groupby(risk_col)[DRIVER_COL]
        .nunique()
        .reindex(RISK_ORDER, fill_value=0)
    )
    return {
        "total": int(total),
        "high_count": int(counts.get("High Risk", 0)),
        "medium_count": int(counts.get("Medium Risk", 0)),
        "low_count": int(counts.get("Low Risk", 0)),
        "high_pct": pct_str(int(counts.get("High Risk", 0)), total),
        "medium_pct": pct_str(int(counts.get("Medium Risk", 0)), total),
        "low_pct": pct_str(int(counts.get("Low Risk", 0)), total),
    }


def avg_violation_df(df_week: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    data = []
    safe_task_days = df_week[TASK_DAYS_COL].replace(0, np.nan)

    for col in VIOLATION_COLS:
        if mode == "unit":
            avg_val = (df_week[col] / safe_task_days).replace([np.inf, -np.inf], np.nan).mean()
            y_name = "Average Count / Task Days"
        else:
            avg_val = df_week[col].mean()
            y_name = "Average Count"

        data.append(
            {
                "Violation Type": VIOLATION_LABELS[col],
                y_name: float(0 if pd.isna(avg_val) else avg_val),
            }
        )
    return pd.DataFrame(data), y_name


def make_pie_chart(df_week: pd.DataFrame) -> str:
    pie_df = (
        df_week.groupby(FINAL_RISK_COL)[DRIVER_COL]
        .nunique()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )
    pie_df.columns = ["Risk Group", "Drivers"]

    fig = px.pie(pie_df, names="Risk Group", values="Drivers", hole=0.4)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=360)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


def make_avg_violation_chart(df_week: pd.DataFrame, mode: str) -> str:
    chart_df, y_name = avg_violation_df(df_week, mode)
    fig = px.bar(chart_df, x="Violation Type", y=y_name, text_auto=".2f")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=360)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def make_importance_chart(df_week: pd.DataFrame) -> str:
    chart_df = pd.DataFrame(
        {
            "Type": ["Emphasized Violation", "Less Important Violation"],
            "Drivers": [
                int(df_week.loc[df_week[EMPHASIZED_COL] > 0, DRIVER_COL].nunique()),
                int(df_week.loc[df_week[LESS_IMPORTANT_COL] > 0, DRIVER_COL].nunique()),
            ],
        }
    )
    fig = px.bar(chart_df, x="Type", y="Drivers", text_auto=True)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=360)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def get_previous_week(week_options: list[str], selected_week: str) -> str | None:
    if selected_week not in week_options:
        return None
    idx = week_options.index(selected_week)
    if idx == 0:
        return None
    return week_options[idx - 1]


def calc_week_comparison(df_current: pd.DataFrame, df_prev: pd.DataFrame) -> list[Dict[str, Any]]:
    items = []
    for col in VIOLATION_COLS:
        curr = float(df_current[col].mean()) if len(df_current) else 0.0
        prev = float(df_prev[col].mean()) if len(df_prev) else 0.0
        delta = curr - prev

        if delta > 1e-9:
            arrow = "↑"
            css = "delta-up"
        elif delta < -1e-9:
            arrow = "↓"
            css = "delta-down"
        else:
            arrow = "→"
            css = "delta-flat"

        items.append(
            {
                "name": VIOLATION_LABELS[col],
                "delta": f"{delta:+.2f}",
                "current": f"{curr:.2f}",
                "previous": f"{prev:.2f}",
                "arrow": arrow,
                "css_class": css,
            }
        )
    return items


def make_score_histogram(df_week: pd.DataFrame) -> str:
    fig = px.histogram(df_week, x=SCORE_COL, nbins=20)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=360)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def make_tm_group_chart(df_week: pd.DataFrame, mode: str) -> str:
    temp = df_week.copy()
    temp[ORIGINAL_RISK_COL] = pd.Categorical(temp[ORIGINAL_RISK_COL], categories=RISK_ORDER, ordered=True)

    rows = []
    for risk in RISK_ORDER:
        grp = temp[temp[ORIGINAL_RISK_COL] == risk]
        grp_task_days = grp[TASK_DAYS_COL].replace(0, np.nan)

        for col in VIOLATION_COLS:
            if mode == "unit":
                value = (grp[col] / grp_task_days).replace([np.inf, -np.inf], np.nan).mean()
                measure = "Average Count / Task Days"
            else:
                value = grp[col].mean()
                measure = "Average Count"

            rows.append(
                {
                    "Risk Group": risk,
                    "Violation Type": VIOLATION_LABELS[col],
                    measure: float(0 if pd.isna(value) else value),
                }
            )

    chart_df = pd.DataFrame(rows)
    fig = px.bar(chart_df, x="Risk Group", y=measure, color="Violation Type", barmode="group")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=360)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def apply_risk_filters(
    df_week: pd.DataFrame,
    final_risk_filter: str,
    original_risk_filter: str,
    video_filter: str,
    search_driver: str,
) -> pd.DataFrame:
    filtered = df_week.copy()

    if final_risk_filter in set(RISK_ORDER):
        filtered = filtered[filtered[FINAL_RISK_COL] == final_risk_filter]

    if original_risk_filter in set(RISK_ORDER):
        filtered = filtered[filtered[ORIGINAL_RISK_COL] == original_risk_filter]

    if video_filter == "yes":
        filtered = filtered[filtered[VIDEO_TOTAL_COL] > 0]
    elif video_filter == "no":
        filtered = filtered[filtered[VIDEO_TOTAL_COL] <= 0]

    search_driver = search_driver.strip()
    if search_driver:
        filtered = filtered[
            filtered[DRIVER_COL].astype(str).str.contains(search_driver, case=False, na=False)
        ]

    return filtered


def format_telematics_profile(row: pd.Series) -> str:
    return (
        f"Hard Braking: {int(row.get('count_Hard Braking', 0))}<br>"
        f"Rapid Acceleration: {int(row.get('count_Rapid Acc', 0))}<br>"
        f"Speeding: {int(row.get('count_Speeding', 0))}<br>"
        f"No Go Zone: {int(row.get('count_No Go Zone', 0))}<br>"
        f"# task days: {float(row.get(TASK_DAYS_COL, 0)):.0f}<br>"
        f"Telematics risk score: {float(row.get(SCORE_COL, 0)):.2f}<br>"
        f"{row.get(ORIGINAL_RISK_COL, '-')}"
    )


def format_video_profile(row: pd.Series) -> str:
    total = int(row.get(VIDEO_TOTAL_COL, 0))
    emphasized = int(row.get(EMPHASIZED_COL, 0))
    less_important = int(row.get(LESS_IMPORTANT_COL, 0))
    return f"{total} ({emphasized} emphasized + {less_important} less important)"


def sort_risk_analysis(df_filtered: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df_filtered.copy()
    df_sorted["_risk_rank"] = df_sorted[FINAL_RISK_COL].map(RISK_RANK_MAP).fillna(99)
    df_sorted = df_sorted.sort_values(
        by=["_risk_rank", SCORE_COL, VIDEO_TOTAL_COL],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)
    return df_sorted


def build_detail_rows(df_filtered: pd.DataFrame, selected_week: str, peer_mode: str) -> list[Dict[str, Any]]:
    detail_df = sort_risk_analysis(df_filtered)

    rows = []
    for _, row in detail_df.iterrows():
        driver_id = row.get(DRIVER_COL, "-")
        rows.append(
            {
                "rank": int(row.get("rank", 0)),
                "driverId": driver_id,
                "unitId": row.get("unitId", "-"),
                "role": row.get("role", "-"),
                "telematics_profile": format_telematics_profile(row),
                "video_profile": format_video_profile(row),
                "final_risk_group": row.get(FINAL_RISK_COL, "-"),
                "action_link": f"/driver/{driver_id}?week={selected_week}&peer_mode={peer_mode}",
            }
        )
    return rows


def format_video_breakdown_html(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for col in cols:
        val = int(row.get(col, 0))
        if val > 0:
            parts.append(f"{col}: {val}")
    return "<br>".join(parts) if parts else "-"


def get_driver_row(df: pd.DataFrame, driver_id: str, week: str) -> pd.Series:
    df_week = filter_week(df, week)
    matched = df_week[df_week[DRIVER_COL].astype(str) == str(driver_id)]
    if matched.empty:
        abort(404, f"No record found for driverId={driver_id} in week={week}")
    return matched.iloc[0]


def get_driver_metric_value(row: pd.Series, col: str, mode: str) -> float:
    if mode == "unit":
        task_days = float(row.get(TASK_DAYS_COL, 0))
        if task_days <= 0:
            return 0.0
        return float(row.get(col, 0)) / task_days
    return float(row.get(col, 0))


def build_peer_charts(df_week: pd.DataFrame, driver_row: pd.Series, mode: str) -> List[str]:
    charts = []
    include_js = "cdn"

    for col in VIOLATION_COLS:
        driver_val = get_driver_metric_value(driver_row, col, mode)
        metric_series = df_week.apply(lambda r: get_driver_metric_value(r, col, mode), axis=1)
        avg_val = float(metric_series.mean()) if len(metric_series) else 0.0
        p25 = float(np.percentile(metric_series, 25)) if len(metric_series) else 0.0
        p50 = float(np.percentile(metric_series, 50)) if len(metric_series) else 0.0
        p75 = float(np.percentile(metric_series, 75)) if len(metric_series) else 0.0

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Individual", "Peer Average"],
                y=[driver_val, avg_val],
                name="Value",
            )
        )
        fig.add_hline(y=p25, line_dash="dot", annotation_text="25th", annotation_position="top left")
        fig.add_hline(y=p50, line_dash="dash", annotation_text="50th", annotation_position="top left")
        fig.add_hline(y=p75, line_dash="dashdot", annotation_text="75th", annotation_position="top left")

        ylabel = "Count / Task Days" if mode == "unit" else "Count"
        fig.update_layout(
            title=VIOLATION_LABELS[col],
            yaxis_title=ylabel,
            margin=dict(l=10, r=10, t=40, b=20),
            height=320,
            showlegend=False,
        )
        charts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,
                config={"displayModeBar": False},
            )
        )
        include_js = False

    return charts


def get_telematics_suggestion(value: float, mean_val: float, p50: float, mode: str) -> str:
    baseline = max(mean_val, p50)
    multiplier = 1.25 if mode == "unit" else 1.50
    mild_threshold = baseline * multiplier

    if value < baseline:
        return "Acceptable. You are doing well and should keep this performance."
    if value <= mild_threshold:
        return "Slight adjustment recommended next month."
    return "There is significant room for improvement."


def build_telematics_improvement_table(df_week: pd.DataFrame, driver_row: pd.Series, mode: str) -> List[Dict[str, Any]]:
    rows = []
    for col in VIOLATION_COLS:
        metric_series = df_week.apply(lambda r: get_driver_metric_value(r, col, mode), axis=1)
        individual_val = get_driver_metric_value(driver_row, col, mode)
        mean_val = float(metric_series.mean()) if len(metric_series) else 0.0
        p50 = float(np.percentile(metric_series, 50)) if len(metric_series) else 0.0

        rows.append(
            {
                "metric": VIOLATION_LABELS[col],
                "individual": round(individual_val, 2),
                "avg": round(mean_val, 2),
                "p50": round(p50, 2),
                "suggestion": get_telematics_suggestion(individual_val, mean_val, p50, mode),
            }
        )
    return rows


def build_video_improvement_table(driver_row: pd.Series) -> List[Dict[str, Any]]:
    rows = []
    for metric in VIDEO_VIOLATIONS:
        count = int(driver_row.get(metric, 0))
        if count > 0:
            importance = "Emphasized" if metric in EMPHASIZE_VIOLATIONS else "Less Important"
            rows.append(
                {
                    "metric": metric,
                    "importance": importance,
                    "individual": count,
                    "ideal": 0,
                    "suggestion": "There is room for improvement.",
                }
            )
    return rows


@app.route("/")
def home():
    df = load_data(CSV_PATH)
    week_options = get_week_options(df)
    if not week_options:
        return "No valid weekNo found in dataset."

    selected_week = request.args.get("week", week_options[-1])
    if selected_week not in week_options:
        selected_week = week_options[-1]

    show_tm_flag = request.args.get("show_tm", "0")
    if show_tm_flag not in {"0", "1"}:
        show_tm_flag = "0"

    avg_mode = request.args.get("avg_mode", "total")
    if avg_mode not in {"total", "unit"}:
        avg_mode = "total"

    tm_mode = request.args.get("tm_mode", "total")
    if tm_mode not in {"total", "unit"}:
        tm_mode = "total"

    final_risk_filter = request.args.get("final_risk_filter", "All")
    original_risk_filter = request.args.get("original_risk_filter", "All")
    video_filter = request.args.get("video_filter", "all")
    search_driver = request.args.get("search_driver", "")
    peer_mode = request.args.get("peer_mode", "total")
    if peer_mode not in {"total", "unit"}:
        peer_mode = "total"

    df_week = filter_week(df, selected_week)
    overview_metrics = calc_risk_metrics(df_week, FINAL_RISK_COL)

    tm_df_week = df_week[df_week[ORIGINAL_RISK_FLAG_COL] == True].copy()
    tm_metrics = calc_risk_metrics(tm_df_week, ORIGINAL_RISK_COL)

    prev_week = get_previous_week(week_options, selected_week)
    comparison_available = prev_week is not None
    comparison_items = []
    if comparison_available:
        df_prev = filter_week(df, prev_week)
        comparison_items = calc_week_comparison(df_week, df_prev)

    filtered_risk_df = apply_risk_filters(
        df_week=df_week,
        final_risk_filter=final_risk_filter,
        original_risk_filter=original_risk_filter,
        video_filter=video_filter,
        search_driver=search_driver,
    )
    detail_rows = build_detail_rows(filtered_risk_df, selected_week, peer_mode)

    return render_template(
        "index.html",
        app_title=APP_TITLE,
        week_options=week_options,
        selected_week=selected_week,
        show_tm_flag=show_tm_flag,
        avg_mode=avg_mode,
        tm_mode=tm_mode,
        overview_metrics=overview_metrics,
        tm_metrics=tm_metrics,
        pie_chart=make_pie_chart(df_week),
        avg_violation_chart=make_avg_violation_chart(df_week, avg_mode),
        importance_chart=make_importance_chart(df_week),
        comparison_available=comparison_available,
        comparison_items=comparison_items,
        score_histogram=make_score_histogram(tm_df_week),
        tm_group_chart=make_tm_group_chart(tm_df_week, tm_mode),
        final_risk_filter=final_risk_filter,
        original_risk_filter=original_risk_filter,
        video_filter=video_filter,
        search_driver=search_driver,
        detail_rows=detail_rows,
        detail_count=len(detail_rows),
        peer_mode=peer_mode,
    )


@app.route("/driver/<driver_id>")
def driver_report(driver_id: str):
    df = load_data(CSV_PATH)
    week_options = get_week_options(df)
    if not week_options:
        abort(404, "No valid weeks found.")

    selected_week = request.args.get("week", week_options[-1])
    if selected_week not in week_options:
        selected_week = week_options[-1]

    peer_mode = request.args.get("peer_mode", "total")
    if peer_mode not in {"total", "unit"}:
        peer_mode = "total"

    df_week = filter_week(df, selected_week)
    driver_row = get_driver_row(df, driver_id, selected_week)

    telematics_total = int(sum(driver_row.get(col, 0) for col in VIOLATION_COLS))
    video_total = int(driver_row.get(VIDEO_TOTAL_COL, 0))
    total_violation_count = telematics_total + video_total
    task_days = float(driver_row.get(TASK_DAYS_COL, 0))

    peer_charts = build_peer_charts(df_week, driver_row, peer_mode)
    telematics_improvement_rows = build_telematics_improvement_table(df_week, driver_row, peer_mode)
    video_improvement_rows = build_video_improvement_table(driver_row)

    weekly_info = {
        "driverId": driver_row.get(DRIVER_COL, "-"),
        "unitId": driver_row.get("unitId", "-"),
        "role": driver_row.get("role", "-"),
        "hard_braking": int(driver_row.get("count_Hard Braking", 0)),
        "rapid_acc": int(driver_row.get("count_Rapid Acc", 0)),
        "speeding": int(driver_row.get("count_Speeding", 0)),
        "no_go_zone": int(driver_row.get("count_No Go Zone", 0)),
        "task_days": round(task_days, 2),
        "telematics_score": round(float(driver_row.get(SCORE_COL, 0)), 2),
        "telematics_group": driver_row.get(ORIGINAL_RISK_COL, "-"),
        "emphasized_count": int(driver_row.get(EMPHASIZED_COL, 0)),
        "emphasized_breakdown": format_video_breakdown_html(driver_row, EMPHASIZE_VIOLATIONS),
        "less_important_count": int(driver_row.get(LESS_IMPORTANT_COL, 0)),
        "less_important_breakdown": format_video_breakdown_html(driver_row, LESS_IMPORTANT_VIOLATIONS),
        "final_risk_group": driver_row.get(FINAL_RISK_COL, "-"),
    }

    return render_template(
        "driver_report.html",
        app_title=APP_TITLE,
        selected_week=selected_week,
        peer_mode=peer_mode,
        weekly_info=weekly_info,
        total_violation_count=total_violation_count,
        telematics_total=telematics_total,
        video_total=video_total,
        task_days=round(task_days, 2),
        peer_charts=peer_charts,
        telematics_improvement_rows=telematics_improvement_rows,
        video_improvement_rows=video_improvement_rows,
        driver_id=driver_id,
    )


if __name__ == "__main__":
    app.run(debug=True)