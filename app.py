import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPUs — avoids ptxas/nvlink crash
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress C++ INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN numeric warnings

import streamlit as st

st.set_page_config(
    page_title="Emoji Expression Predictor",
    page_icon="😂",
    layout="centered",
)

# Custom CSS 
st.markdown(
    """
    <style>
        /* dark-ish neutral card feel */
        .main { background-color: #0f1117; }

        /* big emoji display */
        .emoji-display {
            text-align: center;
            font-size: 90px;
            line-height: 1.1;
            margin: 0.2rem 0;
        }

        /* confidence label under emoji */
        .confidence-label {
            text-align: center;
            font-size: 22px;
            color: #e0e0e0;
            margin-bottom: 0.2rem;
        }

        /* muted cleaned-text line */
        .clean-text {
            text-align: center;
            font-size: 14px;
            color: #888888;
            font-style: italic;
            margin-bottom: 1rem;
        }

        /* top-5 row */
        .bar-emoji {
            font-size: 26px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load inference module (cached — model loads only once per session)
@st.cache_resource(show_spinner="⏳ Loading model, please wait...")
def load_predictor():
    import inference  

    
    try:
        inference._load_resources()
    except FileNotFoundError:
        pass  
    return inference


_predictor = None
_load_error: str | None = None

try:
    _predictor = load_predictor()
except Exception as exc:
    _load_error = str(exc)


#Example sentences 
EXAMPLES = [
    "I love you so much ❤️",
    "This is absolutely hilarious 😂",
    "Happy birthday beautiful! 🎂",
    "I miss being home 💙",
    "Just won the championship! 🏆",
]

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "result" not in st.session_state:
    st.session_state["result"] = None


st.title("😂 Emoji Expression Predictor")
st.markdown(
    "Type a sentence and the Transformer model will predict "
    "which emoji best matches the sentiment."
)

if _load_error:
    st.error(f"⚠️ Model failed to load: {_load_error}")

st.divider()

#  Example sentence buttons 
st.markdown("**💡 Try an example:**")
btn_cols = st.columns(len(EXAMPLES))
for col, example in zip(btn_cols, EXAMPLES):
    if col.button(example, use_container_width=True, key=f"ex_{example[:8]}"):
        st.session_state["user_input"] = example
        st.session_state["result"] = None
        st.rerun()

#  Text input 
user_input: str | None = st.text_area(
    label="Your sentence",
    placeholder='e.g. "This sunset is absolutely gorgeous ✨"',
    height=130,
    label_visibility="collapsed",
    key="user_input",
)

predict_btn = st.button(
    "✨  Predict Emoji",
    type="primary",
    use_container_width=True,
    disabled=(_load_error is not None),
)

# ── Prediction logic 
if predict_btn:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    elif _predictor is None:
        st.error("Model is not available. Check the error above.")
    else:
        with st.spinner("Predicting..."):
            try:
                result = _predictor.predict(user_input)
                st.session_state["result"] = result
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

#  Results display 
result = st.session_state.get("result")

if result is not None:
    st.divider()

    #   predicted emoji 
    st.markdown(
        f"<div class='emoji-display'>{result['emoji']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='confidence-label'>"
        f"Confidence: <b>{result['confidence'] * 100:.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='clean-text'>"
        f"Cleaned input: &ldquo;{result['clean_text']}&rdquo;"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    #  Top-5 bar chart 
    st.markdown("#### 🔝 Top 5 Predictions")

    for i, item in enumerate(result["top5"]):
        emoji = item["emoji"]
        conf = item["confidence"]
        pct = conf * 100

        left, bar_col = st.columns([1, 9])

        left.markdown(
            f"<div class='bar-emoji'>{emoji}</div>",
            unsafe_allow_html=True,
        )

        bar_col.progress(
            value=float(conf),
            text=f"{'**' if i == 0 else ''}{pct:.1f}%{'**' if i == 0 else ''}",
        )

    st.divider()

    with st.expander("🔍 Raw prediction details"):
        st.json(result)
