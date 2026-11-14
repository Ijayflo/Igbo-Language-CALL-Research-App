from __future__ import annotations

import json
import hmac
import uuid
from zoneinfo import ZoneInfo
import requests
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
TONE_IMAGE_DIR = BASE_DIR / "tone_images"
PRACTICE_QUESTIONS = 10
ASSESSMENT_DISTRIBUTION = {"1 Syl": 3, "2 Syl": 6, "3 Syl": 6}


def is_assessment() -> bool:
    return st.session_state.test_type in ("Pre-test", "Post-test")


def max_questions() -> int:
    return sum(ASSESSMENT_DISTRIBUTION.values()) if is_assessment() else PRACTICE_QUESTIONS

LEVEL_MAP = {
    "1 Syl": "1 Syl",
    "2 Syl": "2 Syl",
    "3 Syl": "3 Syl",
}

TONE_IMAGES = {
    tone: TONE_IMAGE_DIR / f"{tone}.png"
    for tone in [
        "H", "L", "HL", "HM", "LH", "HH", "LL", "HLH", "LLL",
        "HHH", "HHL", "LLH", "LHL", "HLL", "LHH", "LHM", "HHM", "HMM"
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
    "session_id": "",
    "researcher_unlocked": False,
    "practice_context": "Home",
    "active_context": "Home",
    "history_key": None,
    "practice_total": 0,
    "practice_week": 1,
    "active_practice_week": 1,
    "pending_response": None,
    "save_error": "",
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
    "assessment_queue": [],
    "post_test_review": [],
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


def build_choices(filtered_df: pd.DataFrame, correct_pattern: str, level: str) -> list[str]:
    if level == "1 Syl":
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


def prepare_assessment_queue() -> None:
    """Sample distinct word IDs per syllable group, then shuffle question order."""
    queue = []
    for level, count in ASSESSMENT_DISTRIBUTION.items():
        pool = get_level_dataframe(level)
        if pool["ID"].nunique() < count:
            raise ValueError(f"At least {count} distinct words are needed for {level}.")
        selected = pool.drop_duplicates(subset="ID").sample(n=count)
        queue.extend(selected["ID"].astype(str).tolist())
    random.shuffle(queue)
    st.session_state.assessment_queue = queue


def create_challenge() -> None:
    if is_assessment():
        if not st.session_state.assessment_queue:
            try:
                prepare_assessment_queue()
            except ValueError as exc:
                st.session_state.challenge = None
                st.session_state.choices = []
                st.session_state.feedback = str(exc)
                return
        next_id = st.session_state.assessment_queue.pop(0)
        selected_row = df[df["ID"].astype(str) == next_id].iloc[0]
        level = str(selected_row["SYLLABLE_TYPE"]).strip()
        filtered_df = get_level_dataframe(level)
        challenge = selected_row
    else:
        level = st.session_state.active_level
        filtered_df = get_level_dataframe(level)
        if filtered_df.empty:
            st.session_state.challenge = None
            st.session_state.choices = []
            st.session_state.feedback = f"No words were found for {level}."
            return
        used_ids = {str(value) for value in st.session_state.used_word_ids}
        remaining_df = filtered_df[~filtered_df["ID"].astype(str).isin(used_ids)]
        if remaining_df.empty:
            st.session_state.used_word_ids = []
            remaining_df = filtered_df
        challenge = remaining_df.sample(1).iloc[0]

    correct_pattern = str(challenge["TONE_PATTERN"]).strip()
    st.session_state.challenge = challenge.to_dict()
    st.session_state.choices = build_choices(filtered_df, correct_pattern, level)
    st.session_state.answered = False
    st.session_state.selected_tone = None
    st.session_state.feedback = ""
    st.session_state.question_started_at = time.perf_counter()


class RedcapDiagnosticError(Exception):
    pass


def redcap_call(stage, **payload):
    try:
        url = st.secrets["REDCAP_API_URL"]
        token = st.secrets["REDCAP_API_TOKEN"]
    except Exception:
        raise RedcapDiagnosticError(f"{stage}: missing or unreadable Streamlit Secrets.") from None
    try:
        response = requests.post(
            url, data={"token": token, "format": "json", "returnFormat": "json", **payload},
            timeout=(5, 20), allow_redirects=False, verify=True,
        )
    except requests.RequestException as exc:
        raise RedcapDiagnosticError(f"{stage}: network error ({type(exc).__name__}).") from None
    # Never display raw server text, credentials, URLs or response records.
    try:
        result = response.json()
    except ValueError:
        result = None
    error = result.get("error") if isinstance(result, dict) else None
    if not response.ok or error is not None or 300 <= response.status_code < 400:
        message = str(error or "").lower()
        hints = []
        for terms, hint in [
            (("token", "authentication"), "Token/authentication issue"),
            (("permission", "privilege", "rights", "authorized"), "Permission issue"),
            (("repeat", "instance"), "Repeating instrument/instance issue"),
            (("event", "longitudinal"), "Event configuration issue"),
            (("record", "record_id"), "Record-related issue"),
            (("validation", "invalid", "field"), "Field or value validation issue"),
        ]:
            if any(term in message for term in terms):
                hints.append(hint)
        if 300 <= response.status_code < 400:
            hints.append("Unexpected redirect; verify API endpoint")
        detail = "; ".join(hints) or "No recognized error category"
        raise RedcapDiagnosticError(f"{stage}: HTTP {response.status_code}; {detail}.")
    if result is None:
        raise RedcapDiagnosticError(f"{stage}: HTTP {response.status_code}; response was not valid JSON.")
    return result


def current_event():
    if st.session_state.test_type == "Pre-test":
        return "pretest_arm_1"
    if st.session_state.test_type == "Post-test":
        return "posttest_arm_1"
    return f"practice_session_{st.session_state.active_practice_week}_arm_1"


def export_history(participant):
    fields = ["record_id", "response_id", "round_number", "session_id", "session_type",
              "question_number", "response_timestamp"]
    rows = redcap_call("Export existing responses", content="record", type="flat", **{
        "records[0]": participant, **{f"fields[{i}]": f for i, f in enumerate(fields)}
    })
    if not isinstance(rows, list):
        raise RedcapDiagnosticError("Export: unexpected JSON structure.")
    return [r for r in rows if r.get("record_id") == participant and
            r.get("redcap_repeat_instrument") == "igbo_tone_response"]


def assessment_rows(rows, mode):
    code = "1" if mode == "Pre-test" else "3"
    return [r for r in rows if str(r.get("session_type")) == code]


def completed(rows):
    groups = {}
    for r in rows:
        if r.get("session_id") and str(r.get("question_number", "")).isdigit():
            groups.setdefault(r["session_id"], set()).add(int(r["question_number"]))
    return any(set(range(1, 16)) <= questions for questions in groups.values())


def check_access(rows, mode, session_id):
    pre = assessment_rows(rows, "Pre-test")
    post = assessment_rows(rows, "Post-test")
    if mode != "Practice":
        if not st.session_state.researcher_unlocked:
            raise RedcapDiagnosticError("Researcher authorization required.")
        prior = assessment_rows(rows, mode)
        if any(r.get("session_id") != session_id for r in prior):
            raise RedcapDiagnosticError("Assessment already started or completed. Contact the researcher; do not retake it.")
        if mode == "Post-test" and not completed(pre):
            raise RedcapDiagnosticError("A completed pre-test is required.")
    else:
        if not completed(pre):
            raise RedcapDiagnosticError("A completed pre-test is required before recorded practice.")
        if post:
            raise RedcapDiagnosticError("Research practice has ended for this participant.")
    return pre


def practice_week(pre):
    dates = [datetime.fromisoformat(r["response_timestamp"]).date() for r in pre if r.get("response_timestamp")]
    if not dates:
        raise RedcapDiagnosticError("Pre-test date is unavailable. Contact the researcher.")
    elapsed = (datetime.now(ZoneInfo("America/Regina")).date() - min(dates)).days
    return max(1, min(4, elapsed // 7 + 1))


def write_result(record):
    existing = export_history(record["record_id"])
    instances = [r for r in existing if r.get("redcap_event_name") == record["redcap_event_name"]]
    if any(r.get("response_id") == record["response_id"] for r in instances):
        return
    check_access(existing, st.session_state.test_type, record["session_id"])
    # Allocate a round number only when its first answer is saved.
    same_round = [r for r in instances if r.get("session_id") == record["session_id"]]
    if same_round:
        record["round_number"] = int(same_round[0]["round_number"])
    else:
        record["round_number"] = 1 + max((int(r["round_number"]) for r in instances if str(r.get("round_number", "")).isdigit()), default=0)
    # Use REDCap's automatic instance allocation (requires supported server version).
    payload = dict(record, redcap_repeat_instrument="igbo_tone_response", redcap_repeat_instance="new")
    result = redcap_call("Import new response", content="record", type="flat", overwriteBehavior="normal",
                         data=json.dumps([payload], ensure_ascii=False), returnContent="count")
    if not isinstance(result, dict) or int(result.get("count", 0)) != 1:
        raise RedcapDiagnosticError("Import: server response did not confirm exactly one record.")


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

    if st.session_state.pending_response is None:
        if not st.session_state.session_id:
            st.session_state.session_id = uuid.uuid4().hex
        syllable = str(challenge["SYLLABLE_TYPE"]).strip()
        st.session_state.pending_response = {
            "record_id": st.session_state.participant_id.strip(),
            "redcap_event_name": current_event(),
            "igbo_tone_response_complete": 2,
            "session_type": {"Pre-test": 1, "Practice": 2, "Post-test": 3}[st.session_state.test_type],
            "round_number": st.session_state.round_number,
            "question_number": question_number,
            "word_id": str(challenge["ID"]),
            "igbo_word": str(challenge["IGBO_WORD"]),
            "gloss": str(challenge.get("GLOSS", "")),
            "correct_tone": correct_tone,
            "selected_tone": selected_tone,
            "is_correct": int(is_correct),
            "response_time": f"{response_time:.3f}",
            "response_timestamp": datetime.now(ZoneInfo("America/Regina")).strftime("%Y-%m-%d %H:%M:%S"),
            "practice_level": "Mixed" if is_assessment() else st.session_state.active_level,
            "syllable_type": {"1 Syl": 1, "2 Syl": 2, "3 Syl": 3}[syllable],
            "session_id": st.session_state.session_id,
            "response_id": uuid.uuid4().hex,
            **({"practice_context": 1 if st.session_state.active_context == "School" else 2} if not is_assessment() else {}),
        }
    record = st.session_state.pending_response
    try:
        write_result(record)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, RedcapDiagnosticError) else f"Unexpected application error ({type(exc).__name__})."
        st.session_state.save_error = "Answer could not be confirmed in REDCap. Retry saving; do not close or refresh this page."
        if str(record["record_id"]).startswith("TEST"):
            st.session_state.save_error += "\n\nDiagnostic: " + detail
        return
    selected_tone = record["selected_tone"]
    is_correct = bool(record["is_correct"])
    st.session_state.pending_response = None
    st.session_state.save_error = ""

    st.session_state.selected_tone = selected_tone
    st.session_state.answered = True
    st.session_state.attempts += 1
    if not is_assessment():
        st.session_state.practice_total += 1

    if is_correct:
        st.session_state.score += 1

    if is_assessment():
        st.session_state.feedback = "Answer recorded."
    elif is_correct:
        st.session_state.feedback = "✅ Correct!"
    else:
        st.session_state.feedback = (
            f"❌ Try again! The correct tone pattern is {correct_tone}."
        )

    if st.session_state.test_type == "Post-test":
        st.session_state.post_test_review.append({
            "Question": question_number,
            "Word": challenge.get("IGBO_WORD", ""),
            "Meaning": challenge.get("GLOSS", ""),
            "Your answer": selected_tone,
            "Correct tone": correct_tone,
            "Result": "Correct" if is_correct else "Incorrect",
        })

    word_id = str(challenge.get("ID", ""))
    if word_id and word_id not in st.session_state.used_word_ids:
        st.session_state.used_word_ids.append(word_id)



def start_next_challenge() -> None:
    if st.session_state.attempts < max_questions():
        create_challenge()


def reset_round() -> None:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.score = 0
    st.session_state.attempts = 0
    st.session_state.challenge = None
    st.session_state.choices = []
    st.session_state.answered = False
    st.session_state.selected_tone = None
    st.session_state.feedback = ""
    st.session_state.used_word_ids = []
    st.session_state.assessment_queue = []
    st.session_state.post_test_review = []
    st.session_state.question_started_at = None
    st.session_state.round_number = 1
    st.session_state.history_key = None
    st.session_state.active_context = st.session_state.get("practice_context", "Home")
    st.session_state.active_level = st.session_state.get("selected_level", "1 Syl")
    create_challenge()


def handle_level_change() -> None:
    reset_round()


def handle_session_change() -> None:
    reset_round()


with st.sidebar:
    with st.expander("Researcher access"):
        if not st.session_state.researcher_unlocked:
            password = st.text_input("Researcher password", type="password", key="research_password")
            if st.button("Unlock assessments"):
                expected = str(st.secrets.get("RESEARCHER_PASSWORD", ""))
                if expected and hmac.compare_digest(password, expected):
                    st.session_state.researcher_unlocked = True
                    st.rerun()
                else:
                    st.error("Access denied.")
        elif st.button("Lock assessments", disabled=st.session_state.pending_response is not None or is_assessment()):
            st.session_state.researcher_unlocked = False
            st.rerun()
    if not st.session_state.researcher_unlocked:
        st.session_state.test_type = "Practice"

    st.text_input(
        "Participant ID",
        key="participant_id",
        placeholder="Enter participant code",
        disabled=st.session_state.pending_response is not None,
        on_change=reset_round,
    )

    st.selectbox(
        "Session Type",
        options=["Practice", "Pre-test", "Post-test"] if st.session_state.researcher_unlocked else ["Practice"],
        key="test_type",
        on_change=handle_session_change,
        disabled=st.session_state.pending_response is not None,
    )

    if is_assessment():
        st.caption("")
    else:
        if "practice_context" not in st.session_state:
            st.session_state.practice_context = st.session_state.active_context
        st.radio("Practice location", ["Home", "School"], key="practice_context",
                 on_change=reset_round, disabled=st.session_state.pending_response is not None)
        if "selected_level" not in st.session_state:
            st.session_state.selected_level = "1 Syl"
        st.radio(
            "Choose a Level",
            options=list(LEVEL_MAP.keys()),
            horizontal=True,
            key="selected_level",
            on_change=handle_level_change,
            disabled=st.session_state.pending_response is not None,
        )

    st.subheader("Progress")
    col1, col2, col3 = st.columns(3)
    show_score = not is_assessment() or (st.session_state.test_type == "Post-test" and st.session_state.attempts >= max_questions())
    col1.metric("Score", st.session_state.score if show_score else "—")
    col2.metric("Attempts", st.session_state.attempts)
    col3.metric("Accuracy", f"{calculate_accuracy():.0f}%" if show_score else "—")

    if st.session_state.feedback:
        st.subheader("")
        st.write(st.session_state.feedback)

        if not is_assessment() and st.session_state.answered and st.session_state.challenge:
            challenge = st.session_state.challenge
            st.write(f"**Word:** {challenge.get('IGBO_WORD', '')}")
            gloss = str(challenge.get("GLOSS", "")).strip()
            if gloss:
                st.write(f"**Meaning:** {gloss}")
            st.write(f"**Tone:** {challenge.get('TONE_PATTERN', '')}")


st.header("🎧 Igbo Tone Perception Trainer")

if not st.session_state.participant_id.strip():
    st.info("Enter a Participant ID in the sidebar before answering questions.")
    st.stop()

if st.session_state.challenge is None and st.session_state.attempts < max_questions():
    st.session_state.active_level = st.session_state.get("selected_level", "1 Syl")
    create_challenge()

if st.session_state.pending_response is not None:
    st.error(st.session_state.save_error)
    st.button("Retry saving answer", on_click=submit_answer, args=(st.session_state.pending_response["selected_tone"],))
    st.stop()

if not st.session_state.session_id:
    st.session_state.session_id = uuid.uuid4().hex
key = (st.session_state.participant_id.strip(), st.session_state.test_type, st.session_state.session_id)
if st.session_state.history_key != key:
    try:
        history = export_history(key[0])
        pre = check_access(history, key[1], key[2])
        if not is_assessment():
            st.session_state.active_practice_week = practice_week(pre)
            st.session_state.active_context = st.session_state.get("practice_context", "Home")
        st.session_state.practice_total = sum(str(r.get("session_type")) == "2" for r in history)
        st.session_state.history_key = key
    except RedcapDiagnosticError as exc:
        st.error(str(exc) if key[0].startswith("TEST") or st.session_state.researcher_unlocked else "Unable to start recorded practice. Please contact the researcher.")
        st.stop()
if not is_assessment():
    st.sidebar.caption(f"Practice week {st.session_state.active_practice_week} · Total saved practice answers: {st.session_state.practice_total}")

if st.session_state.attempts >= max_questions():
    st.balloons()
    if st.session_state.test_type == "Pre-test":
        st.success("### Pre-test Complete!\nGreat job! You completed all 15 questions.")
    elif st.session_state.test_type == "Post-test":
        st.success("### Post-test Complete!\nGreat job! You completed your Igbo tone challenge!")
        c1, c2 = st.columns(2)
        c1.metric("Final Score", f"{st.session_state.score}/{max_questions()}")
        c2.metric("Final Accuracy", f"{calculate_accuracy():.1f}%")
        with st.expander("Review your answers", expanded=True):
            if st.session_state.post_test_review:
                st.dataframe(pd.DataFrame(st.session_state.post_test_review), hide_index=True, use_container_width=True)
            else:
                st.info("No answer review is available for this round. Start a new round to collect review details.")
    else:
        st.success(
            f"### Round Complete!\n"
            f"**Score:** {st.session_state.score}/{max_questions()}  \n"
            f"**Accuracy:** {calculate_accuracy():.1f}%"
        )
    if not is_assessment():
        st.button("🔄 Start New Round", type="primary", on_click=reset_round, use_container_width=True)
    else:
        st.info("Assessment finished. Ask the researcher to select the next activity.")
    st.stop()

question_number = st.session_state.attempts + 1
st.write(f"**Question {question_number} of {max_questions()}**")
st.progress(st.session_state.attempts / max_questions())

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
    if st.session_state.attempts < max_questions():
        st.button(
            "▶ Next Challenge",
            type="primary",
            on_click=start_next_challenge,
            use_container_width=True,
        )
