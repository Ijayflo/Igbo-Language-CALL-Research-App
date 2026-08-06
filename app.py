from __future__ import annotations

import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Igbo Tone Perception Trainer",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "igbo_words.csv"
RESULTS_FILE = BASE_DIR / "results.csv"
TONE_IMAGE_DIR = BASE_DIR / "tone_images"
MAX_QUESTIONS = 10

LEVEL_MAP = {
    "1 Syl": "1 Syl",
    "2 Syl": "2 Syl",
    "3 Syl": "3 Syl",
}

TONE_IMAGES = {
    tone: TONE_IMAGE_DIR / f"{tone}.png"
    for tone in [
        "H", "L", "HL", "HM", "LH", "HH", "LL", "HLH", "LLL",
        "HHH", "HHL", "LLH", "LHL", "HLL", "LHH", "LHM", "HHM",
    ]
}

REQUIRED_COLUMNS = {
    "ID", "IGBO_WORD", "TONE_PATTERN", "AUDIO_FILE", "SYLLABLE_TYPE"
}


@st.cache_data
def load_word_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place igbo_words.csv inside the data folder."
        )

    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].astype(str).str.strip()

    if "GLOSS" not in data.columns:
        data["GLOSS"] = ""
    else:
        data["GLOSS"] = data["GLOSS"].fillna("").astype(str).str.strip()

    data = data[
        (data["ID"] != "")
        & (data["IGBO_WORD"] != "")
        & (data["TONE_PATTERN"] != "")
        & (data["AUDIO_FILE"] != "")
        & (data["SYLLABLE_TYPE"] != "")
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError("The dataset contains no usable rows.")

    return data


try:
    df = load_word_data(DATA_FILE)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()


SESSION_DEFAULTS: dict[str, Any] = {
    "participant_id": "",
    "test_type": "Practice",
    "selected_level": "1 Syl",
    "active_level": "1 Syl",
    "score": 0,
    "attempts": 0,
    "challenge": None,
    "choices": [],
    "answered": False,
    "selected_tone": None,
    "feedback": "",
    "used_word_ids": [],
    "question_started_at": None,
    "round_number": 1,
}


def initialize_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


def resolve_asset_path(raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    return path if path.is_absolute() else BASE_DIR / path


def calculate_accuracy() -> float:
    attempts = st.session_state.attempts
    return 0.0 if attempts == 0 else st.session_state.score / attempts * 100


def get_level_dataframe(level_label: str) -> pd.DataFrame:
    csv_value = LEVEL_MAP[level_label]
    return df[
        df["SYLLABLE_TYPE"].str.casefold() == csv_value.casefold()
    ].copy()


def build_choices(filtered_df: pd.DataFrame, correct_pattern: str) -> list[str]:
    if LEVEL_MAP[st.session_state.active_level] == "1 Syl":
        choices = ["H", "L"]
        random.shuffle(choices)
        return choices

    available_patterns = sorted(
        pattern
        for pattern in filtered_df["TONE_PATTERN"].dropna().unique().tolist()
        if pattern and pattern != correct_pattern
    )
    distractors = random.sample(available_patterns, min(3, len(available_patterns)))
    choices = distractors + [correct_pattern]
    random.shuffle(choices)
    return choices


def create_challenge() -> None:
    filtered_df = get_level_dataframe(st.session_state.active_level)

    if filtered_df.empty:
        st.session_state.challenge = None
        st.session_state.choices = []
        st.session_state.feedback = (
            f"No words were found for {LEVEL_MAP[st.session_state.active_level]}."
        )
        return

    used_ids = {str(value) for value in st.session_state.used_word_ids}
    remaining_df = filtered_df[~filtered_df["ID"].astype(str).isin(used_ids)]

    if remaining_df.empty:
        st.session_state.used_word_ids = []
        remaining_df = filtered_df

    challenge = remaining_df.sample(1).iloc[0]
    correct_pattern = str(challenge["TONE_PATTERN"]).strip()

    st.session_state.challenge = challenge.to_dict()
    st.session_state.choices = build_choices(filtered_df, correct_pattern)
    st.session_state.answered = False
    st.session_state.selected_tone = None
    st.session_state.feedback = ""
    st.session_state.question_started_at = time.perf_counter()


def write_result(
    *,
    challenge: dict[str, Any],
    selected_tone: str,
    correct_tone: str,
    is_correct: bool,
    response_time_seconds: float,
    question_number: int,
) -> None:
    file_exists = RESULTS_FILE.exists()

    headers = [
        "Timestamp", "Participant_ID", "Test_Type", "Round_Number",
        "Question_Number", "Practice_Level", "Word_ID", "Igbo_Word",
        "Gloss", "Correct_Tone", "Selected_Tone", "Correct",
        "Response_Time_Seconds",
    ]

    row = [
        datetime.now().isoformat(timespec="seconds"),
        st.session_state.participant_id.strip(),
        st.session_state.test_type,
        st.session_state.round_number,
        question_number,
        LEVEL_MAP[st.session_state.active_level],
        challenge.get("ID", ""),
        challenge.get("IGBO_WORD", ""),
        str(challenge.get("GLOSS", "")).strip(),
        correct_tone,
        selected_tone,
        is_correct,
        f"{response_time_seconds:.3f}",
    ]

    try:
        with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
    except PermissionError:
        st.session_state.feedback = (
            "⚠️ Your answer was processed, but results.csv could not be updated. "
            "Close it in Excel and continue."
        )


def submit_answer(selected_tone: str) -> None:
    if st.session_state.answered or st.session_state.challenge is None:
        return

    challenge = st.session_state.challenge
    correct_tone = str(challenge["TONE_PATTERN"]).strip()
    question_number = st.session_state.attempts + 1
    started_at = st.session_state.question_started_at
    response_time = (
        max(0.0, time.perf_counter() - started_at)
        if started_at is not None
        else 0.0
    )
    is_correct = selected_tone == correct_tone

    st.session_state.selected_tone = selected_tone
    st.session_state.answered = True
    st.session_state.attempts += 1

    if is_correct:
        st.session_state.score += 1
        st.session_state.feedback = "✅ Correct!"
    else:
        st.session_state.feedback = (
            f"❌ Try again! The correct tone pattern is {correct_tone}."
        )

    word_id = str(challenge.get("ID", ""))
    if word_id and word_id not in st.session_state.used_word_ids:
        st.session_state.used_word_ids.append(word_id)

    write_result(
        challenge=challenge,
        selected_tone=selected_tone,
        correct_tone=correct_tone,
        is_correct=is_correct,
        response_time_seconds=response_time,
        question_number=question_number,
    )


def start_next_challenge() -> None:
    if st.session_state.attempts < MAX_QUESTIONS:
        create_challenge()


def reset_round() -> None:
    st.session_state.score = 0
    st.session_state.attempts = 0
    st.session_state.challenge = None
    st.session_state.choices = []
    st.session_state.answered = False
    st.session_state.selected_tone = None
    st.session_state.feedback = ""
    st.session_state.used_word_ids = []
    st.session_state.question_started_at = None
    st.session_state.round_number += 1
    st.session_state.active_level = st.session_state.selected_level
    create_challenge()


def handle_level_change() -> None:
    st.session_state.active_level = st.session_state.selected_level
    st.session_state.score = 0
    st.session_state.attempts = 0
    st.session_state.challenge = None
    st.session_state.choices = []
    st.session_state.answered = False
    st.session_state.selected_tone = None
    st.session_state.feedback = ""
    st.session_state.used_word_ids = []
    st.session_state.question_started_at = None
    create_challenge()


with st.sidebar:
    #st.subheader("")

    st.text_input(
        "Participant ID",
        key="participant_id",
        placeholder="Enter participant code",
    )

    st.selectbox(
        "Session Type",
        options=["Practice", "Pre-test", "Post-test"],
        key="test_type",
    )

    st.radio(
        "Choose a Level",
        options=list(LEVEL_MAP.keys()),
        horizontal=True,
        key="selected_level",
        on_change=handle_level_change,
    )

    st.subheader("Progress")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", st.session_state.score)
    col2.metric("Attempts", st.session_state.attempts)
    col3.metric("Accuracy", f"{calculate_accuracy():.0f}%")

    if st.session_state.feedback:
        st.subheader("")
        st.write(st.session_state.feedback)

        if st.session_state.answered and st.session_state.challenge:
            challenge = st.session_state.challenge
            st.write(f"**Word:** {challenge.get('IGBO_WORD', '')}")
            gloss = str(challenge.get("GLOSS", "")).strip()
            if gloss:
                st.write(f"**Meaning:** {gloss}")
            st.write(f"**Tone:** {challenge.get('TONE_PATTERN', '')}")


st.header("🎧 Igbo Tone Perception Trainer")

if not st.session_state.participant_id.strip():
    st.info("Enter a Participant ID in the sidebar before answering questions.")

if st.session_state.challenge is None and st.session_state.attempts < MAX_QUESTIONS:
    st.session_state.active_level = st.session_state.selected_level
    create_challenge()

if st.session_state.attempts >= MAX_QUESTIONS:
    st.balloons()
    st.success(
        f"### Round Complete!\n"
        f"**Score:** {st.session_state.score}/{MAX_QUESTIONS}  \n"
        f"**Accuracy:** {calculate_accuracy():.1f}%"
    )
    st.button(
        "🔄 Start New Round",
        type="primary",
        on_click=reset_round,
        use_container_width=True,
    )
    st.stop()

question_number = st.session_state.attempts + 1
st.write(f"**Question {question_number} of {MAX_QUESTIONS}**")
st.progress(st.session_state.attempts / MAX_QUESTIONS)

challenge = st.session_state.challenge
if challenge is None:
    st.error(st.session_state.feedback or "A challenge could not be created.")
    st.stop()

audio_path = resolve_asset_path(str(challenge["AUDIO_FILE"]))
st.subheader("🔊 Listen Carefully")
if audio_path.exists():
    st.audio(str(audio_path))
else:
    st.error(f"Audio file not found: {audio_path}")

st.subheader("Choose the Correct Tone Pattern")
choices = st.session_state.choices
if not choices:
    st.error("No tone-pattern choices are available for this question.")
    st.stop()

columns = st.columns(len(choices))
for index, tone in enumerate(choices):
    with columns[index]:
        image_path = TONE_IMAGES.get(tone)
        if image_path and image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Image missing: {tone}")

        is_selected = st.session_state.selected_tone == tone
        st.button(
            "Selected" if is_selected else "Select",
            key=f"answer_{question_number}_{challenge['ID']}_{tone}",
            on_click=submit_answer,
            args=(tone,),
            disabled=(
                st.session_state.answered
                or not st.session_state.participant_id.strip()
            ),
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        )

if st.session_state.answered:
    st.markdown("---")
    if st.session_state.attempts < MAX_QUESTIONS:
        st.button(
            "▶ Next Challenge",
            type="primary",
            on_click=start_next_challenge,
            use_container_width=True,
        )
