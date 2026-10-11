import plotly.graph_objects as go
import streamlit as st

from gateway_client import GatewayClient, GatewayError

st.set_page_config(page_title="EUR Forex Direction Signal", page_icon="📈")


@st.cache_resource
def get_client() -> GatewayClient:
    return GatewayClient()


@st.cache_data(ttl=3600, show_spinner=False)
def load_currencies() -> list:
    return get_client().currencies()["currencies"]


@st.cache_data(ttl=300, show_spinner=False)
def load_model_info() -> dict:
    return get_client().model_info()


def gauge(prob_up: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_up * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probability the rate is higher"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#333333", "thickness": 0.25},
            "steps": [{"range": [0, 50], "color": "#f4c7c3"}, {"range": [50, 100], "color": "#c8e6c9"}],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.9, "value": 50},
        },
    ))
    fig.update_layout(height=260, margin={"t": 60, "b": 10, "l": 30, "r": 30})
    return fig


st.title("EUR forex: next-day direction signal")
st.warning(
    "Educational project, not financial advice. Daily currency moves are close to random: "
    "the model's historical accuracy is only slightly above a coin flip."
)

try:
    currencies = load_currencies()
except GatewayError as e:
    st.error(str(e))
    st.stop()

currency = st.selectbox("Currency (units per 1 EUR)", currencies,
                        index=currencies.index("LKR") if "LKR" in currencies else 0)

if st.button("Get signal", type="primary"):
    try:
        with st.spinner("Calling the WSO2 gateway..."):
            st.session_state["result"] = get_client().predict(currency)
    except GatewayError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

res = st.session_state.get("result")
if res:
    arrow = "⬆️" if res["direction"] == "up" else "⬇️"
    c1, c2, c3 = st.columns(3)
    c1.metric("Direction", f"{arrow} {res['direction'].upper()}")
    c2.metric("Probability up", f"{res['prob_up'] * 100:.1f}%")
    c3.metric("Strength", res["strength"].title())
    st.plotly_chart(gauge(res["prob_up"]))
    st.caption(f"{res['currency']} signal for {res['for_trading_day']} (data through {res['as_of_date']}) "
               f"| model {res['model_version']}")
    st.info(
        f"On past out-of-sample days, {res['strength']}-strength calls were right "
        f"{res['historical_hit_rate'] * 100:.1f}% of the time ({res['historical_sample_size']:,} days). "
        "A coin flip is 50%."
    )
    st.caption(res["disclaimer"])

with st.expander("About the model"):
    try:
        info = load_model_info()
        st.write(f"Version **{info['model_version']}**, data through **{info['data_through']}**.")
        st.write(f"Out-of-sample test from {info['test_from']}: accuracy **{info['test_accuracy'] * 100:.1f}%**, "
                 f"ROC-AUC **{info['test_roc_auc']:.3f}** (0.5 means no skill).")
    except GatewayError as e:
        st.write(str(e))