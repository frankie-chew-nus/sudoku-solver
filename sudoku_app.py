import json
import time
import streamlit as st
from logic_ import *
from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
    pl_bc_entails,
)

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="centered")
st.title("🧩 Sudoku Solver")
st.caption("Sudoku knowledge representation and inference using propositional logic")


@st.cache_data
def load_puzzles(filename="puzzles.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


pool = load_puzzles()
n = pool["n"]
box_h = pool["box_h"]
box_w = pool["box_w"]
puzzles = pool["puzzles"]


def convert_givens(raw_givens):
    return {
        tuple(map(int, key.split("_"))): int(value)
        for key, value in raw_givens.items()
    }


def board_as_html(board, givens=None):
    """Render a board using theme-aware text/background colours."""
    givens = givens or {}
    html = ["<table style='border-collapse:collapse; margin:auto; color:inherit;'>"]

    for r in range(1, n + 1):
        html.append("<tr>")
        for c in range(1, n + 1):
            value = board.get((r, c), "")
            top = "3px" if (r - 1) % box_h == 0 else "1px"
            left = "3px" if (c - 1) % box_w == 0 else "1px"
            bottom = "3px" if r % box_h == 0 else "1px"
            right = "3px" if c % box_w == 0 else "1px"

            if (r, c) in givens:
                content = f"<strong>{value}</strong>"
                background = "rgba(128,128,128,0.18)"
            else:
                content = str(value)
                background = "transparent"

            html.append(
                "<td style='"
                "width:42px;height:42px;text-align:center;"
                f"font-size:20px;color:inherit;background:{background};"
                f"border-top:{top} solid currentColor;"
                f"border-left:{left} solid currentColor;"
                f"border-bottom:{bottom} solid currentColor;"
                f"border-right:{right} solid currentColor;"
                "'>"
                f"{content}</td>"
            )
        html.append("</tr>")

    html.append("</table>")
    return "".join(html)


def show_board(board, givens=None):
    st.markdown(board_as_html(board, givens), unsafe_allow_html=True)


def decode_atom(proposition):
    name = proposition.op
    if name.startswith("Is"):
        prefix, coordinates = "Is", name[2:]
    elif name.startswith("Not"):
        prefix, coordinates = "Not", name[3:]
    else:
        return None

    try:
        r, c, v = map(int, coordinates.split("_"))
        return prefix, r, c, v
    except ValueError:
        return None


def proposition_text(proposition):
    decoded = decode_atom(proposition)
    if decoded is None:
        return str(proposition)

    prefix, r, c, v = decoded
    if prefix == "Is":
        return f"cell ({r}, {c}) has value {v}"
    return f"value {v} is eliminated from cell ({r}, {c})"


def explain_rule(premises, conclusion):
    conclusion_info = decode_atom(conclusion)
    if conclusion_info is None:
        return f"Inferred {conclusion}."

    prefix, r, c, v = conclusion_info

    if prefix == "Not" and len(premises) == 1:
        premise_info = decode_atom(premises[0])
        if premise_info is not None and premise_info[0] == "Is":
            _, pr, pc, pv = premise_info

            if (pr, pc) == (r, c):
                return (
                    f"Cell ({r}, {c}) is already known to be {pv}, "
                    f"so it cannot also contain {v}."
                )

            if pr == r:
                relation = "same row"
            elif pc == c:
                relation = "same column"
            else:
                relation = "same box"

            return (
                f"Cell ({pr}, {pc}) contains {pv}. Because cell ({r}, {c}) "
                f"is in the {relation}, value {v} is eliminated from cell ({r}, {c})."
            )

    if prefix == "Is":
        eliminated = []
        for premise in premises:
            info = decode_atom(premise)
            if (
                info is not None
                and info[0] == "Not"
                and info[1] == r
                and info[2] == c
            ):
                eliminated.append(info[3])

        if eliminated and len(eliminated) == len(premises):
            eliminated.sort()
            values = ", ".join(map(str, eliminated))
            return (
                f"For cell ({r}, {c}), values {values} have all been eliminated. "
                f"Therefore {v} is the last remaining candidate."
            )

    premise_text = "; ".join(proposition_text(p) for p in premises)
    return f"Because {premise_text}, infer that {proposition_text(conclusion)}."


def backward_chain_with_trace(kb, query):
    """Multi-pass BC trace matching the core solver's retry behaviour.

    A witness is recorded whenever a proposition is proved. Once the query is
    proved, those witnesses are followed backwards to build a real proof path.
    """
    facts = set()
    rules_by_conclusion = {}

    for clause in kb.clauses:
        premises, conclusion = parse_definite_clause(clause)
        if not premises:
            facts.add(conclusion)
        else:
            rules_by_conclusion.setdefault(conclusion, []).append(premises)

    proved = set(facts)
    witness = {fact: ("fact", []) for fact in facts}

    while True:
        before = len(proved)
        failed_this_pass = set()
        active = set()

        def prove(goal):
            if goal in proved:
                return True
            if goal in failed_this_pass or goal in active:
                return False

            active.add(goal)
            try:
                for premises in rules_by_conclusion.get(goal, []):
                    if all(prove(p) for p in premises):
                        proved.add(goal)
                        witness[goal] = ("rule", list(premises))
                        return True
            finally:
                active.remove(goal)

            failed_this_pass.add(goal)
            return False

        if prove(query):
            break

        if len(proved) == before:
            return False, []

    trace = []
    emitted = set()

    def build_trace(goal):
        if goal in emitted:
            return
        step_type, premises = witness[goal]
        for premise in premises:
            build_trace(premise)

        emitted.add(goal)
        trace.append({
            "type": step_type,
            "conclusion": goal,
            "premises": premises,
        })

    build_trace(query)
    return True, trace


def display_reasoning_trace(trace, query):
    if not trace:
        st.info(
            "No successful proof path was found for this query. "
            "A False entailment result means the current Horn knowledge base "
            "cannot derive the requested proposition."
        )
        return

    st.subheader("Reasoning trace")
    st.write(
        "The steps below show one successful backward-chaining proof path "
        "for the requested proposition."
    )

    for step_number, step in enumerate(trace, start=1):
        conclusion = step["conclusion"]
        premises = step["premises"]

        if step["type"] == "fact":
            explanation = (
                f"{proposition_text(conclusion).capitalize()} is an initial "
                "given fact in the puzzle."
            )
            title = f"Step {step_number}: Use a given fact"
        else:
            explanation = explain_rule(premises, conclusion)
            title = f"Step {step_number}: Infer {proposition_text(conclusion)}"

        with st.expander(title):
            st.write(explanation)
            if premises:
                st.markdown("**Premises used:**")
                for premise in premises:
                    st.write(f"• {proposition_text(premise)}")
            st.markdown(f"**Conclusion:** {proposition_text(conclusion)}")

    st.success(
        f"Therefore, {proposition_text(query)} is entailed by the "
        "definite-clause knowledge base."
    )


# 1. Puzzle selection
st.header("1. Select a puzzle")

puzzle_index = st.selectbox(
    "Puzzle",
    options=range(len(puzzles)),
    format_func=lambda i: f"Puzzle {i + 1} — {puzzles[i]['given_count']} givens",
)

selected_puzzle = puzzles[puzzle_index]
givens = convert_givens(selected_puzzle["givens"])

st.write(
    f"Selected puzzle: **Puzzle {puzzle_index + 1}** "
    f"with **{selected_puzzle['given_count']} givens**."
)
show_board(givens, givens)
st.caption("Bold shaded values are the original givens. Blank cells are unknown.")


# 2. Full-grid solver
st.header("2. Solve the full grid")

algorithm = st.radio(
    "Inference algorithm",
    options=["Forward chaining", "Backward chaining"],
    horizontal=True,
    help=(
        "Forward chaining is data-driven. Backward chaining is goal-driven "
        "and proves each requested cell value from the definite-clause KB."
    ),
)

if st.button("Solve puzzle", type="primary"):
    start_time = time.perf_counter()

    if algorithm == "Forward chaining":
        solved = solve_full_grid_fc(n, box_h, box_w, givens)
    else:
        solved = solve_full_grid_bc(n, box_h, box_w, givens)

    elapsed = time.perf_counter() - start_time

    if len(solved) == n * n:
        st.success(f"Solved using {algorithm} in {elapsed:.4f} seconds.")
        show_board(solved, givens)
    else:
        st.warning(
            f"The selected Horn inference method derived {len(solved)} of "
            f"{n * n} cells in {elapsed:.4f} seconds."
        )
        show_board(solved, givens)


# 3. Targeted entailment query
st.header("3. Ask a cell-value query")
st.write(
    "Choose a row, column, and value. The application will test whether "
    "`Is(r,c,v)` is entailed by the definite-clause knowledge base using "
    "your backward-chaining implementation."
)

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    query_r = st.number_input("Row", min_value=1, max_value=n, value=1, step=1)
with input_col2:
    query_c = st.number_input("Column", min_value=1, max_value=n, value=1, step=1)
with input_col3:
    query_v = st.number_input("Value", min_value=1, max_value=n, value=1, step=1)

show_trace = st.checkbox(
    "Show tutor-mode reasoning trace",
    value=True,
    help="Displays a human-readable successful proof path when the proposition is entailed.",
)

if st.button("Check entailment"):
    kb = build_definite_kb(n, box_h, box_w, givens)
    query = atom("Is", int(query_r), int(query_c), int(query_v))

    start_time = time.perf_counter()
    entailed = pl_bc_entails(kb, query)
    elapsed = time.perf_counter() - start_time

    if entailed:
        st.success(f"True — cell ({query_r}, {query_c}) = {query_v} is entailed.")
    else:
        st.error(
            f"False — cell ({query_r}, {query_c}) = {query_v} is not "
            "entailed by this knowledge base."
        )

    st.caption(f"Backward-chaining query time: {elapsed:.6f} seconds")

    if show_trace:
        if entailed:
            trace_proved, trace = backward_chain_with_trace(kb, query)
            if trace_proved:
                display_reasoning_trace(trace, query)
            else:
                st.warning(
                    "The query was proved, but the reasoning trace could not "
                    "be reconstructed."
                )
        else:
            display_reasoning_trace([], query)
