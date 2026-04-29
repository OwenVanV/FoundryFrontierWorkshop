"""
Shared Streamlit helpers for the workshop demo apps.
Provides subprocess streaming, module metadata extraction, and UI components.
"""

import os
import sys
import ast
import subprocess
import threading
import queue
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent


def get_module_docstring(filepath: Path) -> str:
    """Extract the module-level docstring from a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        return docstring or ""
    except Exception:
        return ""


def get_function_docs(filepath: Path) -> list[dict]:
    """Extract function names, docstrings, source code, and preceding comments from a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        source_lines = source.split("\n")
        tree = ast.parse(source)
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)

                # Extract the full function source code
                start = node.lineno - 1  # 0-indexed
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
                func_source = "\n".join(source_lines[start:end])

                # Extract the comment block immediately above the function
                # (the # ==== / # ---- sections that describe each step)
                preceding_comments = []
                scan_line = start - 1
                while scan_line >= 0:
                    stripped = source_lines[scan_line].strip()
                    if stripped.startswith("#"):
                        preceding_comments.insert(0, source_lines[scan_line])
                        scan_line -= 1
                    elif stripped == "":
                        # Allow one blank line gap
                        if scan_line - 1 >= 0 and source_lines[scan_line - 1].strip().startswith("#"):
                            preceding_comments.insert(0, source_lines[scan_line])
                            scan_line -= 1
                        else:
                            break
                    else:
                        break

                # Clean up the comment block into readable text
                comment_text = ""
                if preceding_comments:
                    cleaned = []
                    for cl in preceding_comments:
                        c = cl.strip()
                        if c.startswith("# ====") or c.startswith("# ----"):
                            continue  # Skip separator lines
                        if c.startswith("# "):
                            cleaned.append(c[2:])
                        elif c == "#":
                            cleaned.append("")
                    comment_text = "\n".join(cleaned).strip()

                funcs.append({
                    "name": node.name,
                    "line": node.lineno,
                    "doc": doc or "",
                    "source": func_source,
                    "comments": comment_text,
                })
        return funcs
    except Exception:
        return []


def get_comment_blocks(filepath: Path) -> list[dict]:
    """Extract large comment blocks (# ===... sections) that describe steps."""
    blocks = []
    try:
        lines = filepath.read_text(encoding="utf-8").split("\n")
        current_block = []
        start_line = 0
        in_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# ==========") or stripped.startswith("# ----------"):
                if current_block and len(current_block) > 2:
                    text = "\n".join(current_block)
                    # Extract a title from the block
                    title = ""
                    for bl in current_block:
                        bl_stripped = bl.strip().lstrip("# ").strip()
                        if bl_stripped and not bl_stripped.startswith("===") and not bl_stripped.startswith("---"):
                            title = bl_stripped
                            break
                    blocks.append({"title": title, "text": text, "line": start_line})
                current_block = [line]
                start_line = i + 1
                in_block = True
            elif in_block and stripped.startswith("#"):
                current_block.append(line)
            elif in_block and not stripped.startswith("#"):
                if current_block and len(current_block) > 2:
                    text = "\n".join(current_block)
                    title = ""
                    for bl in current_block:
                        bl_stripped = bl.strip().lstrip("# ").strip()
                        if bl_stripped and not bl_stripped.startswith("===") and not bl_stripped.startswith("---"):
                            title = bl_stripped
                            break
                    blocks.append({"title": title, "text": text, "line": start_line})
                current_block = []
                in_block = False

        return blocks
    except Exception:
        return []


def stream_subprocess(cmd: list[str], cwd: str = None) -> None:
    """Run a subprocess and stream its output into a Streamlit container.

    Uses st.status for the running state and streams output line-by-line
    into a code block that auto-scrolls. Provides a Stop button to kill
    the process mid-run.
    """
    if cwd is None:
        cwd = str(PROJECT_ROOT)

    output_container = st.empty()
    status_container = st.empty()
    stop_container = st.empty()

    lines = []
    return_code = None

    # Show a stop button that persists during the run
    stop_key = f"stop_{id(cmd)}_{hash(tuple(cmd))}"

    with status_container.status("🔄 Running...", expanded=True) as status:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )

        # Place stop button
        stop_clicked = stop_container.button("⏹️ Stop Module", key=stop_key, type="secondary")

        import select
        import time

        while process.poll() is None:
            # Non-blocking read with timeout so we can check the stop state
            try:
                line = process.stdout.readline()
            except Exception:
                break

            if line:
                lines.append(line.rstrip())
                display_lines = lines[-200:]
                output_container.code("\n".join(display_lines), language="log", line_numbers=False)

            # Check if stop was requested (via session state since button rerenders)
            if stop_clicked:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                lines.append("\n⏹️ Module stopped by user.")
                output_container.code("\n".join(lines[-200:]), language="log", line_numbers=False)
                status.update(label="⏹️ Stopped by user", state="complete")
                stop_container.empty()
                return -15

        # Read any remaining output
        remaining = process.stdout.read()
        if remaining:
            for line in remaining.strip().split("\n"):
                lines.append(line.rstrip())

        return_code = process.returncode

        if return_code == 0:
            status.update(label="✅ Completed successfully", state="complete")
        else:
            status.update(label=f"❌ Failed (exit code {return_code})", state="error")

    stop_container.empty()

    # Final full output
    output_container.code("\n".join(lines[-300:]), language="log", line_numbers=False)
    return return_code


def render_module_page(
    module_title: str,
    module_file: Path,
    run_cmd: list[str],
    extra_description: str = "",
):
    """Render a standard module page with description, function docs, and run button."""

    st.header(module_title)

    # Module docstring as description
    docstring = get_module_docstring(module_file)
    if docstring:
        # Parse out key sections from the docstring
        sections = {}
        current_section = "overview"
        current_lines = []
        for line in docstring.split("\n"):
            stripped = line.strip()
            if stripped.endswith(":") and stripped.isupper():
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = stripped.rstrip(":")
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        # Display narrative
        narrative = sections.get("WORKSHOP NARRATIVE", sections.get("overview", ""))
        if narrative:
            st.markdown("### 📖 Narrative")
            st.info(narrative.strip())

        # Learning objectives
        objectives = sections.get("LEARNING OBJECTIVES", "")
        if objectives:
            with st.expander("🎯 Learning Objectives", expanded=False):
                st.markdown(objectives.strip())

        # Azure services
        services = sections.get("AZURE SERVICES USED", sections.get("AZURE SERVICES", ""))
        if services:
            with st.expander("☁️ Azure Services", expanded=False):
                st.markdown(services.strip())

        # Estimated time
        time_est = sections.get("ESTIMATED TIME", "")
        if time_est:
            st.caption(f"⏱️ {time_est.strip()}")

    if extra_description:
        st.markdown(extra_description)

    # Function documentation with source code
    funcs = get_function_docs(module_file)
    if funcs:
        public_funcs = [f for f in funcs if not f["name"].startswith("_")]
        with st.expander(f"🔍 Functions in this module ({len(public_funcs)})", expanded=False):
            for func in public_funcs:
                st.markdown(f"---")
                st.markdown(f"#### `{func['name']}()` — line {func['line']}")

                # Show the preceding comment block (the workshop narrative for this step)
                if func["comments"]:
                    st.info(func["comments"])

                # Show the docstring
                if func["doc"]:
                    st.caption(func["doc"])

                # Show the source code
                st.code(func["source"], language="python", line_numbers=True)

    st.divider()

    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("▶️ Run Module", type="primary", use_container_width=True)
    with col2:
        st.code(" ".join(run_cmd), language="bash")

    if run_clicked:
        st.divider()
        st.subheader("📺 Live Output")
        stream_subprocess(run_cmd)


def setup_page(title: str, icon: str):
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
