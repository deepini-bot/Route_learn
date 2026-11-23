"""Streamlit chat interface for the RouteLearn API."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


class APIError(Exception):
    """An API request failed or returned an unsuccessful status."""


def api_request(base_url: str, path: str, *, method: str = "GET",
                payload: dict | None = None, api_key: str = "") -> dict:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=90) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise APIError(f"API returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise APIError(f"Could not reach the RouteLearn API: {error.reason}") from error
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise APIError("The RouteLearn API returned an invalid JSON response.") from error
    if not isinstance(result, dict):
        raise APIError("The RouteLearn API returned an unexpected response.")
    return result


st.set_page_config(page_title="RouteLearn", page_icon="🧭", layout="wide")

st.title("🧭 RouteLearn")
st.caption("An adaptive model router that learns from answer quality and feedback.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input(
        "API URL",
        value=os.getenv("ROUTELEARN_API_URL", "http://localhost:8000"),
        help="Base URL for the running RouteLearn FastAPI server.",
    ).strip()
    api_key = st.text_input(
        "API key",
        value=os.getenv("ROUTELEARN_API_KEY", ""),
        type="password",
        help="Only needed when the API is configured with ROUTELEARN_API_KEYS.",
    )
    explain_routes = st.toggle("Explain routing decisions", value=True)

    st.divider()
    st.header("Router settings")
    lam = st.slider(
        "Cost preference (lambda)",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.get("router_lam", 0.6)),
        step=0.05,
        help="Higher values favor lower-cost model tiers.",
    )
    judge_rate = st.slider(
        "Judge sampling rate",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("judge_rate", 0.1)),
        step=0.05,
        help="Fraction of answers scored by the quality judge.",
    )
    cascade_tau = st.slider(
        "Cascade quality threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("cascade_tau", 0.0)),
        step=0.05,
        help="A low-scoring answer is retried on the next tier. Set to 0 to disable.",
    )
    if st.button("Apply router settings", use_container_width=True):
        try:
            current = api_request(
                api_url,
                "/config",
                method="POST",
                payload={
                    "lam": lam,
                    "judge_rate": judge_rate,
                    "cascade_tau": None if cascade_tau == 0 else cascade_tau,
                },
                api_key=api_key,
            )
        except APIError as error:
            st.error(str(error))
        else:
            st.session_state.router_lam = current["lam"]
            st.session_state.judge_rate = current["judge_rate"]
            st.session_state.cascade_tau = current["cascade_tau"] or 0.0
            st.success("Router settings updated.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            details = message.get("details", {})
            if details:
                quality = details.get("judged_quality")
                quality_label = f"{quality:.2f}" if quality is not None else "not sampled"
                st.caption(
                    f"Tier: {details.get('tier', 'unknown')} · "
                    f"Model: {details.get('model', 'unknown')} · "
                    f"Cost: ${details.get('cost', 0):.6f} · "
                    f"Judge quality: {quality_label}"
                )
                if message.get("explanation"):
                    with st.expander("Routing explanation"):
                        st.write(f"Selected tier: **{message['explanation']['selected_tier']}**")
                        estimates = message["explanation"].get("quality_est", [])
                        uncertainty = message["explanation"].get("uncertainty", [])
                        tiers = message["explanation"].get("tiers", [])
                        if estimates and uncertainty:
                            for tier, estimate, spread in zip(tiers, estimates, uncertainty):
                                st.write(f"{tier}: estimated quality {estimate:.3f} ± {spread:.3f}")

                if details.get("request_id") and not message.get("feedback_sent"):
                    left, right, _ = st.columns([1, 1, 8])
                    with left:
                        if st.button("👍", key=f"good-{details['request_id']}"):
                            try:
                                api_request(
                                    api_url,
                                    "/feedback",
                                    method="POST",
                                    payload={"request_id": details["request_id"], "score": 1.0},
                                    api_key=api_key,
                                )
                            except APIError as error:
                                st.error(str(error))
                            else:
                                st.session_state.messages[index]["feedback_sent"] = True
                                st.toast("Thanks for the feedback!")
                                st.rerun()
                    with right:
                        if st.button("👎", key=f"bad-{details['request_id']}"):
                            try:
                                api_request(
                                    api_url,
                                    "/feedback",
                                    method="POST",
                                    payload={"request_id": details["request_id"], "score": 0.0},
                                    api_key=api_key,
                                )
                            except APIError as error:
                                st.error(str(error))
                            else:
                                st.session_state.messages[index]["feedback_sent"] = True
                                st.toast("Thanks for the feedback!")
                                st.rerun()
                elif message.get("feedback_sent"):
                    st.caption("Feedback submitted")

prompt = st.chat_input("Ask RouteLearn anything")
if prompt:
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
        if message["role"] in ("user", "assistant")
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        explanation = None
        if explain_routes:
            explanation = api_request(
                api_url,
                "/explain",
                method="POST",
                payload={"prompt": prompt, "history": history},
                api_key=api_key,
            )
        result = api_request(
            api_url,
            "/chat",
            method="POST",
            payload={"prompt": prompt, "history": history},
            api_key=api_key,
        )
    except APIError as error:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Request failed:** {error}",
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "details": result,
            "explanation": explanation,
        })
    st.rerun()

with st.expander("Router status"):
    if st.button("Refresh status"):
        try:
            st.session_state.router_stats = api_request(
                api_url, "/stats", api_key=api_key
            )
        except APIError as error:
            st.error(str(error))
    stats = st.session_state.get("router_stats")
    if stats:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Requests", stats.get("requests", 0))
        metric_columns[1].metric(
            "Average cost",
            f"${stats.get('avg_cost', 0):.6f}",
        )
        metric_columns[2].metric(
            "Escalation rate",
            f"{stats.get('escalation_rate', 0):.1%}",
        )
        metric_columns[3].metric(
            "Lambda",
            f"{stats.get('lam', 0):.2f}",
        )
        st.write("Observations by tier", stats.get("observations_per_tier", {}))
        st.write("Traffic share", stats.get("tier_share", {}))
        st.caption(
            "Provider mode: "
            + ("live" if stats.get("live_models") else "mock")
        )
    else:
        st.caption("Refresh status to load current router statistics.")
