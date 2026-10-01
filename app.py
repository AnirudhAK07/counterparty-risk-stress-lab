"""Interactive demonstration of an illustrative FX counterparty exposure workflow.

Run from the project root with ``streamlit run app.py`` after creating the model
artifacts. The dashboard reads model outputs and calls the C++ pricing engine
through ``src.engine``; it does not train a model on page load.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
FORECAST_PATH = PROJECT_DIR / "models" / "latest_forecast.json"
HISTORY_PATH = PROJECT_DIR / "models" / "forecast_history.csv"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
TRADES_PATH = PROJECT_DIR / "data" / "trades.csv"
NETTING_PATH = PROJECT_DIR / "data" / "netting_sets.csv"
DATABASE_PATH = PROJECT_DIR / "data" / "risk_lab.sqlite"
REQUIRED_ENGINE_COLUMNS = {
    "counterparty",
    "netting_set",
    "mtm_usd",
    "collateral_usd",
    "exposure_usd",
}


st.set_page_config(
    page_title="Counterparty Risk Stress Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1260px; padding-top: 1.7rem; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    .hero {
        background: linear-gradient(115deg, #102944 0%, #17486a 65%, #147f87 100%);
        border-radius: 16px; color: #fff; padding: 27px 32px; margin-bottom: 22px;
    }
    .hero .eyebrow { color: #9ce2dc; font-size: 0.76rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; }
    .hero h1 { color: #fff; margin: 7px 0 6px; font-size: 2.25rem; }
    .hero p { color: #e0edf5; margin: 0; max-width: 720px; }
    .note { border-left: 4px solid #16868b; background: #edf7f7;
        border-radius: 4px; padding: 12px 16px; margin: 10px 0 20px;
        color: #173c48; }
    .risk-label { font-size: .75rem; font-weight: 700; letter-spacing: .08em;
        color: #167c81; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _forecast_values(data: dict[str, Any]) -> dict[str, Any] | None:
    required = (
        "spot", "lower_spot", "upper_spot", "q05_log_return",
        "q95_log_return", "horizon_observations",
    )
    if any(key not in data for key in required):
        return None
    try:
        parsed = {key: float(data[key]) for key in required}
        parsed["horizon_observations"] = int(parsed["horizon_observations"])
        parsed["origin_date"] = str(data.get("origin_date", "latest observation"))
        if parsed["spot"] <= 0 or parsed["lower_spot"] <= 0 or parsed["upper_spot"] <= 0:
            return None
        return parsed
    except (TypeError, ValueError, OverflowError):
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _price(spot: float, collateral_multiplier: float, elapsed_days: int = 0) -> pd.DataFrame:
    # Import here so an absent or unbuilt native engine yields a useful UI error.
    from src.engine import run_engine

    result = run_engine(
        project_dir=PROJECT_DIR,
        spot=float(spot),
        elapsed_days=elapsed_days,
        collateral_multiplier=float(collateral_multiplier),
    )
    if not isinstance(result, pd.DataFrame):
        raise TypeError("The pricing engine must return a pandas DataFrame.")
    missing = REQUIRED_ENGINE_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"The pricing engine is missing columns: {', '.join(sorted(missing))}")
    return result


def _by_counterparty(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby("counterparty", as_index=False, dropna=False)[
        ["mtm_usd", "collateral_usd", "exposure_usd"]
    ].sum()
    return result.sort_values("exposure_usd", ascending=False).reset_index(drop=True)


def _exposure_total(frame: pd.DataFrame) -> float:
    return float(pd.to_numeric(frame["exposure_usd"], errors="coerce").fillna(0).sum())


def _hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Banking analytics · interactive model demonstration</div>
          <h1>Counterparty Risk Stress Lab</h1>
          <p>Explore how a forecast range for EUR/USD and collateral change the
          amount a bank could be owed on illustrative FX forwards.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status() -> None:
    st.markdown(
        """
        <div class="note"><strong>Illustrative research project.</strong> The trades and
        counterparties are fictional. The FX history is public market data. These
        figures are scenario exposures, not a default probability, regulatory
        potential future exposure, capital amount, or trading recommendation.</div>
        """,
        unsafe_allow_html=True,
    )


def _forecast_cards(forecast: dict[str, Any]) -> None:
    st.markdown('<span class="risk-label">XGBoost forecast</span>', unsafe_allow_html=True)
    st.subheader("Where might EUR/USD move?")
    st.caption(
        f"Forecast made on {forecast['origin_date']} for the next "
        f"{forecast['horizon_observations']} FX observations. Spot is USD per EUR."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed spot", f"{forecast['spot']:.4f}")
    c2.metric("Lower spot (5th percentile)", f"{forecast['lower_spot']:.4f}")
    c3.metric("Upper spot (95th percentile)", f"{forecast['upper_spot']:.4f}")
    st.caption(
        "The two XGBoost models estimate a 5th and 95th percentile of the future "
        "10-observation log return. This is an estimated 90% interval under "
        "conditions like the historical test data; it does not guarantee coverage."
    )


def _scenario_chart(
    current: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame
) -> go.Figure:
    names = sorted(set(current["counterparty"]) | set(lower["counterparty"]) | set(upper["counterparty"]))
    fig = go.Figure()
    specs = [
        ("Observed spot", current, "#297f90"),
        ("Forecast lower FX", lower, "#e4a440"),
        ("Forecast upper FX", upper, "#6e75b8"),
    ]
    for label, frame, color in specs:
        values = frame.set_index("counterparty")["exposure_usd"].reindex(names, fill_value=0)
        fig.add_bar(name=label, x=names, y=values, marker_color=color)
    fig.update_layout(
        barmode="group", height=390, margin=dict(l=12, r=12, t=12, b=15),
        yaxis_title="Uncollateralized exposure (USD)", xaxis_title="Counterparty",
        legend=dict(orientation="h", y=1.13, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, rangemode="tozero")
    return fig


def _exposure_scenarios(forecast: dict[str, Any], multiplier: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with st.spinner("Valuing illustrative FX forwards through the C++ engine..."):
        current = _by_counterparty(_price(forecast["spot"], multiplier, elapsed_days=0))
        lower = _by_counterparty(_price(forecast["lower_spot"], multiplier, elapsed_days=14))
        upper = _by_counterparty(_price(forecast["upper_spot"], multiplier, elapsed_days=14))
    return current, lower, upper


def _overview(forecast: dict[str, Any]) -> None:
    _forecast_cards(forecast)
    try:
        current, lower, upper = _exposure_scenarios(forecast)
    except Exception as exc:
        st.error(f"Exposure calculation is unavailable: {exc}")
        st.info("Build the C++ engine and load the sample trade data, then refresh this page.")
        return

    st.divider()
    st.markdown('<span class="risk-label">Trade valuation</span>', unsafe_allow_html=True)
    st.subheader("What would those FX levels mean for the bank?")
    a, b, c = st.columns(3)
    a.metric("Exposure at observed FX", _money(_exposure_total(current)))
    b.metric("Exposure at lower FX", _money(_exposure_total(lower)))
    c.metric("Exposure at upper FX", _money(_exposure_total(upper)))
    st.plotly_chart(_scenario_chart(current, lower, upper), width="stretch")
    st.caption(
        "Positive values show the amount the bank could be owed after netting and "
        "illustrative collateral. Each bar reprices the same trades at a different "
        "FX spot. Observed spot is valued now; forecast-boundary spots are valued "
        "14 calendar days later as an approximation for ten published FX observations. "
        "Rates and received collateral are held fixed."
    )

    st.subheader("How the parts fit together")
    x, y, z = st.columns(3)
    x.markdown("**1 · Forecast**\n\nXGBoost learns a low and high FX return from earlier public exchange-rate observations.")
    y.markdown("**2 · Reprice**\n\nA C++ engine values the sample FX forwards at the observed and forecast-boundary spots.")
    z.markdown("**3 · Review**\n\nThe dashboard nets trades, applies collateral, and compares the model with a historical baseline.")


def _scenario_explorer(forecast: dict[str, Any]) -> None:
    st.subheader("Scenario explorer")
    st.write(
        "Move the FX rate and change available collateral. The app reprices every "
        "sample trade and sums exposure after each netting set is processed."
    )
    left, right = st.columns([1, 1])
    stress_pct = left.slider("EUR/USD spot change", min_value=-20, max_value=20, value=0, step=1, format="%d%%")
    collateral_multiplier = right.slider(
        "Collateral multiplier", min_value=0.0, max_value=2.0,
        value=1.0, step=0.1,
        help="1.0 uses the sample collateral amounts; 0.0 removes them; 2.0 doubles them.",
    )
    shocked_spot = forecast["spot"] * (1 + stress_pct / 100)
    st.caption(f"Scenario FX spot: {shocked_spot:.4f} USD per EUR. The valuation date and interest rates stay fixed.")
    try:
        current = _by_counterparty(_price(forecast["spot"], 1.0))
        shocked_sets = _price(shocked_spot, collateral_multiplier)
        shocked = _by_counterparty(shocked_sets)
    except Exception as exc:
        st.error(f"Exposure calculation is unavailable: {exc}")
        return

    base_total, scenario_total = _exposure_total(current), _exposure_total(shocked)
    a, b, c = st.columns(3)
    a.metric("Observed spot · original collateral", _money(base_total))
    b.metric("Your scenario", _money(scenario_total))
    c.metric("Change in exposure", _money(scenario_total - base_total))

    names = sorted(set(current["counterparty"]) | set(shocked["counterparty"]))
    base_values = current.set_index("counterparty")["exposure_usd"].reindex(names, fill_value=0)
    scenario_values = shocked.set_index("counterparty")["exposure_usd"].reindex(names, fill_value=0)
    fig = go.Figure()
    fig.add_bar(name="Observed spot", x=names, y=base_values, marker_color="#297f90")
    fig.add_bar(name="Your scenario", x=names, y=scenario_values, marker_color="#e4a440")
    fig.update_layout(
        barmode="group", height=400, margin=dict(l=12, r=12, t=15, b=15),
        yaxis_title="Uncollateralized exposure (USD)",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, rangemode="tozero")
    st.plotly_chart(fig, width="stretch")

    # Show each netting set. Summed MTM/collateral across sets would not
    # reconcile to exposure because the zero floor applies per set.
    detail = shocked_sets.sort_values(["counterparty", "netting_set"]).copy()
    for col in ("mtm_usd", "collateral_usd", "exposure_usd"):
        detail[col] = detail[col].map(_money)
    detail = detail.rename(columns={
        "counterparty": "Counterparty", "netting_set": "Netting set",
        "mtm_usd": "Net market value",
        "collateral_usd": "Collateral", "exposure_usd": "Exposure",
    })
    st.dataframe(detail, width="stretch", hide_index=True)
    st.caption(
        "This is a spot and collateral sensitivity, not a probability of counterparty default. "
        "The model's lower and upper FX forecast scenarios appear on Overview."
    )


def _validation(history: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.subheader("Backtest and model monitoring")
    st.write(
        "A backtest compares each historical prediction with the FX return that "
        "actually followed. Later dates are the most meaningful check because the "
        "model did not train on those outcomes."
    )
    required = {"origin_date", "realized_return", "q05", "q95", "baseline_q05", "baseline_q95"}
    if history.empty or not required.issubset(history.columns):
        st.warning("Forecast history is not available yet. Run the training pipeline, then refresh.")
        return
    data = history.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"], errors="coerce")
    for col in required - {"origin_date"}:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(required)).sort_values("origin_date")
    if data.empty:
        st.warning("The saved backtest contains no usable dated rows.")
        return

    split_options = [str(x) for x in data.get("split", pd.Series(dtype="object")).dropna().unique()]
    if split_options:
        preferred = next((x for x in split_options if x.lower() in {"test", "holdout", "out_of_sample"}), split_options[-1])
        selected = st.selectbox("Backtest period", split_options, index=split_options.index(preferred))
        data = data[data["split"].astype(str) == selected].copy()
    else:
        st.caption("The history file has no named data split; metrics below use all saved rows.")

    model_inside = data["realized_return"].between(data["q05"], data["q95"])
    baseline_inside = data["realized_return"].between(data["baseline_q05"], data["baseline_q95"])
    model_width = (data["q95"] - data["q05"]).mean()
    baseline_width = (data["baseline_q95"] - data["baseline_q05"]).mean()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("XGBoost interval coverage", _percent(float(model_inside.mean())))
    m2.metric("Baseline interval coverage", _percent(float(baseline_inside.mean())))
    m3.metric("XGBoost average width", _percent(float(model_width)))
    m4.metric("Backtest forecasts", f"{len(data):,}")
    st.caption(
        f"The target interval is 90%. The baseline uses historical returns and has "
        f"average width {_percent(float(baseline_width))}. Coverage alone is not "
        "enough: an interval can cover more simply by being wider. Returns are log returns."
    )

    plot = data.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot["origin_date"], y=plot["q95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=plot["origin_date"], y=plot["q05"], mode="lines",
        fill="tonexty", fillcolor="rgba(41,127,144,0.19)",
        line=dict(color="#297f90", width=1), name="XGBoost 5th–95th interval",
    ))
    fig.add_trace(go.Scatter(
        x=plot["origin_date"], y=plot["realized_return"], mode="lines+markers",
        line=dict(color="#d87634", width=1.5), marker=dict(size=3), name="Realized return",
    ))
    fig.update_layout(
        height=420, margin=dict(l=12, r=12, t=15, b=15),
        yaxis_title="10-observation FX log return", xaxis_title="Forecast origin date",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.caption("Chart shows at most the last 120 forecasts in the selected period.")

    misses = data.loc[~model_inside, ["origin_date", "realized_return", "q05", "q95"]].copy()
    with st.expander(f"Inspect {len(misses):,} forecasts outside the XGBoost interval"):
        if misses.empty:
            st.write("No misses in the selected period. Check interval width before interpreting this as strong performance.")
        else:
            misses["origin_date"] = misses["origin_date"].dt.date
            st.dataframe(misses.tail(25), width="stretch", hide_index=True)
    if metrics:
        with st.expander("Saved training and validation metrics"):
            st.json(metrics)


def _database_summary() -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    try:
        connection = sqlite3.connect(f"file:{DATABASE_PATH.as_posix()}?mode=ro", uri=True)
        try:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (name,) in tables:
                # SQLite identifiers cannot be parameterized; names come from sqlite_master.
                quoted_name = '"' + name.replace('"', '""') + '"'
                count = connection.execute(f"SELECT COUNT(*) FROM {quoted_name}").fetchone()[0]
                rows.append({"Table": name, "Rows": count})
        finally:
            connection.close()
    except sqlite3.Error:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _data_and_assumptions(forecast: dict[str, Any], metrics: dict[str, Any]) -> None:
    st.subheader("Data, assumptions, and limits")
    st.write(
        "The market series is the public Federal Reserve DEXUSEU exchange rate "
        "(US dollars per euro). The trades, counterparties, collateral agreements, "
        "and portfolio are synthetic examples made for this project."
    )
    st.markdown("#### Sample trades")
    trades = _read_csv(TRADES_PATH)
    if trades.empty:
        st.info("Sample trades have not been generated yet.")
    else:
        st.dataframe(trades, width="stretch", hide_index=True)
    st.markdown("#### Netting sets and collateral")
    netting = _read_csv(NETTING_PATH)
    if netting.empty:
        st.info("Netting-set data have not been generated yet.")
    else:
        st.dataframe(netting, width="stretch", hide_index=True)

    db = _database_summary()
    st.markdown("#### SQLite run history")
    if db.empty:
        st.caption("The local SQLite file has not been populated yet.")
    else:
        st.dataframe(db, width="stretch", hide_index=True)
        st.caption("Row counts show saved project inputs and model runs; the database contains illustrative data.")

    st.markdown("#### Concepts to know")
    with st.expander("What is a counterparty and why does exposure matter?"):
        st.write(
            "A counterparty is the other party to a trade. If a trade has positive "
            "value to the bank, the counterparty may owe the bank money. Exposure "
            "is the part that remains after eligible offsets and collateral."
        )
    with st.expander("What is netting?"):
        st.write(
            "Trades in the same legal netting set can offset positive and negative "
            "market values. The project nets within each set before subtracting "
            "its illustrative collateral. It does not assume every trade across "
            "a company can be netted together."
        )
    with st.expander("What do XGBoost and the 5th/95th percentiles do?"):
        st.write(
            "XGBoost combines many small decision trees. Here two quantile models "
            "learn a lower and upper bound for future FX returns from past market "
            "features. A backtest checks how often later observed returns fell "
            "inside those bounds."
        )
    with st.expander("What is a log return?"):
        st.write(
            "A log return is ln(future FX spot / current FX spot). It converts "
            "a predicted return back to spot with current spot × exp(return). "
            "The model horizon is measured in FX observations, not exact calendar days."
        )
    st.markdown("#### Model limitations")
    st.markdown(
        "- Only one FX rate and a small synthetic forward portfolio are modeled.\n"
        "- Market regimes can change; historical coverage may not hold in a new crisis.\n"
        "- Forecast scenarios change FX spot and advance trade age by an illustrative 14 days. Manual scenarios change today's spot and received collateral. Rates stay fixed.\n"
        "- Collateral treatment is simplified; real agreements have thresholds, timing, and disputes.\n"
        "- This is not a default model, a regulatory PFE calculation, or a capital model."
    )
    if metrics:
        with st.expander("Full saved model metrics"):
            st.json(metrics)


def main() -> None:
    _hero()
    _status()
    forecast = _forecast_values(_read_json(FORECAST_PATH))
    if forecast is None:
        st.warning(
            "The trained model's latest forecast is not available. Run the project's "
            "setup and training steps, then refresh this page."
        )
        st.info("Use the setup and training commands in the project README, then run `streamlit run app.py`.")
        return
    history = _read_csv(HISTORY_PATH)
    metrics = _read_json(METRICS_PATH)

    st.sidebar.markdown("### Explore the lab")
    page = st.sidebar.radio(
        "Page", ["Overview", "Scenario explorer", "Backtest & monitoring", "Data & assumptions"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Forecast origin: {forecast['origin_date']}")
    st.sidebar.caption("Public FX history · fictional bank trades")
    st.sidebar.caption("Research demonstration only")

    if page == "Overview":
        _overview(forecast)
    elif page == "Scenario explorer":
        _scenario_explorer(forecast)
    elif page == "Backtest & monitoring":
        _validation(history, metrics)
    else:
        _data_and_assumptions(forecast, metrics)


if __name__ == "__main__":
    main()
