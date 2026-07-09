#!/usr/bin/env python3
"""Streamlit GUI for MCQ generation with live logs, prompt editing, and DOCX export."""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Allow running as `streamlit run streamlit_app.py` from mcq_agent/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cost_tracking import UsageCostCallback
from src.export import PandocNotFoundError, convert_md_to_docx
from src.graph import stream_workflow
from src.prompt_store import ALL_PROMPT_KEYS, PROMPT_GROUPS, get_defaults, prompt_overrides
from src.schemas import Difficulty, MCQBatchOutput
from src.utils import (
    ENV_EXAMPLE_PATH,
    PROJECT_ROOT,
    get_model_config,
    load_active_dotenv,
    resolve_output_paths,
    set_dotenv_path,
    setup_logging,
    validate_data_directories,
    write_outputs,
)

logger = logging.getLogger("mcq_agent.streamlit")


class ListLogHandler(logging.Handler):
    """Append formatted log records to a shared list for the UI."""

    def __init__(self, records: list[str]) -> None:
        super().__init__()
        self.records = records
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


def _apply_api_key(api_key: str) -> None:
    """Set OPENAI_API_KEY so chat/embedding clients pick it up for this process."""
    os.environ["OPENAI_API_KEY"] = api_key.strip()


def _configure_streamlit_env() -> None:
    """Load non-secret settings from ``.env.example``; API key comes from the UI only."""
    set_dotenv_path(ENV_EXAMPLE_PATH)
    # Drop any process-level key so Streamlit never silently uses a local .env secret.
    os.environ.pop("OPENAI_API_KEY", None)
    load_active_dotenv(override=True)
    # Example file keeps OPENAI_API_KEY empty; ensure it stays unset until the user enters one.
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        os.environ.pop("OPENAI_API_KEY", None)


def _init_session_state() -> None:
    defaults = get_defaults()
    if "prompt_values" not in st.session_state:
        st.session_state.prompt_values = dict(defaults)
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "run_result" not in st.session_state:
        st.session_state.run_result = None
    if "cost_snapshot" not in st.session_state:
        st.session_state.cost_snapshot = None
    if "output_files" not in st.session_state:
        st.session_state.output_files = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def _reset_prompts_to_defaults() -> None:
    st.session_state.prompt_values = get_defaults()
    for key in ALL_PROMPT_KEYS:
        widget_key = f"prompt_widget_{key}"
        if widget_key in st.session_state:
            st.session_state[widget_key] = st.session_state.prompt_values[key]


def _collect_prompt_overrides_from_widgets() -> dict[str, str]:
    """Read current text-area values into session state and return overrides."""
    defaults = get_defaults()
    overrides: dict[str, str] = {}
    for key in ALL_PROMPT_KEYS:
        widget_key = f"prompt_widget_{key}"
        value = st.session_state.get(widget_key, st.session_state.prompt_values.get(key))
        if value is None:
            value = defaults[key]
        st.session_state.prompt_values[key] = value
        if value != defaults[key]:
            overrides[key] = value
    return overrides


def _render_prompts_tab() -> None:
    st.markdown(
        "Edit **system** prompts and retrieval queries for this browser session only. "
        "User-prompt templates are not editable here. Changes apply to the next "
        "**Generate** run and reset when you refresh the page."
    )
    if st.button("Reset all prompts to defaults", key="reset_prompts"):
        _reset_prompts_to_defaults()
        st.success("Prompts restored to built-in defaults.")
        st.rerun()

    for group_name, keys in PROMPT_GROUPS.items():
        with st.expander(group_name, expanded=False):
            for key in keys:
                widget_key = f"prompt_widget_{key}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state.prompt_values.get(
                        key, get_defaults()[key]
                    )
                st.text_area(
                    key,
                    key=widget_key,
                    height=180,
                    help="Use the same `{placeholders}` as the original template.",
                )


def _format_usage(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "No usage recorded yet."
    lines = [
        f"Prompt tokens: {snapshot.get('prompt_tokens', 0):,}",
        f"Completion tokens: {snapshot.get('completion_tokens', 0):,}",
        f"Embedding tokens: {snapshot.get('embedding_tokens', 0):,}",
        f"Total tokens: {snapshot.get('total_tokens', 0):,}",
    ]
    models = snapshot.get("models_seen") or []
    if models:
        lines.append(f"Models: {', '.join(models)}")
    return "\n\n".join(lines)


def _session_prompts_for_output() -> dict[str, str]:
    """Effective editable prompts for this session (widget values or defaults)."""
    defaults = get_defaults()
    prompts: dict[str, str] = {}
    for key in ALL_PROMPT_KEYS:
        widget_key = f"prompt_widget_{key}"
        value = st.session_state.get(
            widget_key,
            st.session_state.prompt_values.get(key, defaults[key]),
        )
        prompts[key] = value
    return prompts


def _preview_questions(batch: MCQBatchOutput) -> None:
    if not batch.questions:
        st.warning("No approved questions.")
        return
    for index, record in enumerate(batch.questions, start=1):
        mcq = record.mcq
        with st.expander(f"Q{index}: {mcq.question[:80]}…", expanded=index == 1):
            st.markdown(f"**Stem:** {mcq.question}")
            for letter in ("A", "B", "C", "D"):
                text = getattr(mcq.options, letter)
                marker = "✓" if letter == mcq.correct_answer else " "
                st.markdown(f"- [{marker}] **{letter}.** {text}")
            st.markdown(f"**Explanation:** {mcq.explanation}")
            st.caption(
                f"Misconception: {record.question_blueprint.target_misconception} · "
                f"Revisions: {record.revision_rounds}"
            )


def _write_run_outputs(
    batch: MCQBatchOutput,
    output_name: str,
    session_prompts: dict[str, str] | None = None,
) -> dict[str, Any]:
    json_path, md_path, rejected_path = resolve_output_paths(output_name=output_name)
    written_json, written_md = write_outputs(
        batch,
        json_path,
        md_path,
        rejected_path,
        session_prompts=session_prompts,
    )
    result: dict[str, Any] = {
        "json_path": written_json,
        "md_path": written_md,
        "docx_path": None,
        "docx_error": None,
        "json_bytes": written_json.read_bytes(),
        "md_bytes": written_md.read_bytes(),
        "docx_bytes": None,
    }
    docx_path = written_md.with_suffix(".docx")
    try:
        convert_md_to_docx(written_md, docx_path)
        result["docx_path"] = docx_path
        result["docx_bytes"] = docx_path.read_bytes()
    except PandocNotFoundError as exc:
        result["docx_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        result["docx_error"] = f"DOCX export failed: {exc}"
    return result


def _run_generation(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    num_questions: int,
    batch_size: int,
    max_revision_rounds: int,
    rebuild_index: bool,
    output_name: str,
    prompt_override_map: dict[str, str],
    log_placeholder: Any,
    progress_placeholder: Any,
    usage_placeholder: Any,
) -> None:
    st.session_state.log_lines = []
    st.session_state.run_result = None
    st.session_state.cost_snapshot = None
    st.session_state.output_files = None
    st.session_state.last_error = None

    log_lines: list[str] = st.session_state.log_lines
    handler = ListLogHandler(log_lines)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    cost_cb = UsageCostCallback()
    session_prompts = _session_prompts_for_output()
    completed: list = []
    rejected: list = []
    workflow_error: str | None = None

    def _refresh_ui(node_name: str | None = None) -> None:
        if node_name:
            progress_placeholder.info(
                f"Node: `{node_name}` · approved {len(completed)}/{num_questions} · "
                f"rejected {len(rejected)}"
            )
        log_placeholder.code("\n".join(log_lines[-200:]) or "(waiting for logs…)", language=None)
        usage_placeholder.markdown(_format_usage(cost_cb.snapshot()))

    try:
        validate_data_directories(logger)
        with prompt_overrides(prompt_override_map):
            for node_name, update in stream_workflow(
                topic=topic,
                learning_objective=learning_objective,
                difficulty=difficulty,
                num_questions=num_questions,
                batch_size=batch_size,
                max_revision_rounds=max_revision_rounds,
                rebuild_indexes=rebuild_index,
                callbacks=[cost_cb],
            ):
                if "completed_questions" in update and update["completed_questions"]:
                    completed.extend(update["completed_questions"])
                if "rejected_questions" in update and update["rejected_questions"]:
                    rejected.extend(update["rejected_questions"])
                if update.get("error"):
                    workflow_error = update["error"]
                _refresh_ui(node_name)
                # Brief yield so Streamlit can paint updates.
                time.sleep(0.01)

        st.session_state.cost_snapshot = cost_cb.snapshot()
        _refresh_ui()

        if workflow_error:
            st.session_state.last_error = workflow_error
            return
        if not completed:
            st.session_state.last_error = (
                f"No approved questions were generated ({len(rejected)} rejected)."
            )
            return

        batch = MCQBatchOutput(
            topic=topic,
            learning_objective=learning_objective,
            difficulty=difficulty,
            num_questions=len(completed),
            questions=completed,
            rejected_questions=rejected,
        )
        st.session_state.run_result = batch
        st.session_state.output_files = _write_run_outputs(
            batch,
            output_name,
            session_prompts=session_prompts,
        )
        logger.info(
            "Wrote %d approved question(s) (%d rejected).",
            len(completed),
            len(rejected),
        )
        _refresh_ui()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow failed: %s", exc)
        st.session_state.last_error = str(exc)
        st.session_state.cost_snapshot = cost_cb.snapshot()
        _refresh_ui()
    finally:
        root_logger.removeHandler(handler)


def main() -> None:
    st.set_page_config(
        page_title="Chem MCQ Generator",
        page_icon="🧪",
        layout="wide",
    )
    _configure_streamlit_env()
    setup_logging()
    _init_session_state()

    model_name = str(get_model_config().get("model") or "unknown")
    st.title("Chemistry MCQ Generator")
    st.caption(
        f"LangGraph workflow · config from `{ENV_EXAMPLE_PATH.name}` · "
        f"model `{model_name}` · prompt edits are session-only"
    )

    with st.sidebar:
        st.header("API key")
        st.caption(
            f"Model and retrieval settings load from `{ENV_EXAMPLE_PATH.name}`. "
            "Enter your OpenAI API key here (required)."
        )
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Required. Used only for this Streamlit process; not saved to disk.",
            key="openai_api_key_input",
        )

        st.header("Run configuration")
        topic = st.text_input("Topic", placeholder="intermolecular forces")
        learning_objective = st.text_area(
            "Learning objective",
            placeholder="Explain how IMFs affect boiling points",
            height=100,
        )
        difficulty = st.selectbox(
            "Difficulty",
            options=[d.value for d in Difficulty],
            index=1,
        )
        num_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )
        max_revision_rounds = st.number_input(
            "Max revision rounds",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )
        rebuild_index = st.checkbox("Rebuild vector index before run", value=False)
        output_name = st.text_input(
            "Output name",
            value=f"generated_mcqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Base filename under outputs/ (without extension).",
        )
        generate = st.button("Generate", type="primary", use_container_width=True)

    tab_generate, tab_prompts = st.tabs(["Generate", "Prompts"])

    with tab_prompts:
        _render_prompts_tab()

    with tab_generate:
        progress_placeholder = st.empty()
        usage_placeholder = st.empty()
        log_placeholder = st.empty()

        if st.session_state.cost_snapshot and not generate:
            usage_placeholder.markdown(_format_usage(st.session_state.cost_snapshot))
        if st.session_state.log_lines and not generate:
            log_placeholder.code(
                "\n".join(st.session_state.log_lines[-200:]),
                language=None,
            )

        if generate:
            if not topic.strip() or not learning_objective.strip():
                st.error("Topic and learning objective are required.")
            elif not (api_key_input or "").strip():
                st.error("OpenAI API key is required. Enter it in the sidebar.")
            else:
                _apply_api_key(api_key_input)
                overrides = _collect_prompt_overrides_from_widgets()
                with st.status("Running MCQ workflow…", expanded=True) as status:
                    _run_generation(
                        topic=topic.strip(),
                        learning_objective=learning_objective.strip(),
                        difficulty=difficulty,
                        num_questions=int(num_questions),
                        batch_size=int(batch_size),
                        max_revision_rounds=int(max_revision_rounds),
                        rebuild_index=rebuild_index,
                        output_name=output_name.strip() or "generated_mcqs",
                        prompt_override_map=overrides,
                        log_placeholder=log_placeholder,
                        progress_placeholder=progress_placeholder,
                        usage_placeholder=usage_placeholder,
                    )
                    if st.session_state.last_error:
                        status.update(
                            label="Workflow finished with errors",
                            state="error",
                        )
                    else:
                        status.update(label="Workflow complete", state="complete")

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        if st.session_state.cost_snapshot:
            st.subheader("Token usage")
            st.markdown(_format_usage(st.session_state.cost_snapshot))

        result: MCQBatchOutput | None = st.session_state.run_result
        if result is not None:
            st.subheader("Results")
            st.success(
                f"Approved {len(result.questions)} / requested "
                f"(rejected {len(result.rejected_questions)})."
            )
            _preview_questions(result)

            files = st.session_state.output_files or {}
            st.subheader("Downloads")
            col_json, col_md, col_docx = st.columns(3)
            with col_json:
                if files.get("json_bytes"):
                    st.download_button(
                        "Download JSON",
                        data=files["json_bytes"],
                        file_name=Path(files["json_path"]).name,
                        mime="application/json",
                        use_container_width=True,
                    )
            with col_md:
                if files.get("md_bytes"):
                    st.download_button(
                        "Download Markdown",
                        data=files["md_bytes"],
                        file_name=Path(files["md_path"]).name,
                        mime="text/markdown",
                        use_container_width=True,
                    )
            with col_docx:
                if files.get("docx_bytes"):
                    st.download_button(
                        "Download DOCX",
                        data=files["docx_bytes"],
                        file_name=Path(files["docx_path"]).name,
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        use_container_width=True,
                    )
                elif files.get("docx_error"):
                    st.warning(files["docx_error"])

        st.subheader("Live logs")
        if st.session_state.log_lines:
            st.code("\n".join(st.session_state.log_lines), language=None)
        else:
            st.info("Logs from the next run will appear here.")


if __name__ == "__main__":
    main()
