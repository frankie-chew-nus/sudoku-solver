import json
import time
import streamlit as st
from utils import *
from logic_ import *
from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
    pl_bc_entails,
)

st.title('Sudoku Solver')

with open('puzzles.json') as f:
    pool = json.load(f)

# --- 1. Puzzle selection & visual board display ---
# TODO: a dropdown/selectbox to pick a puzzle by index from pool['puzzles'].
# TODO: render the grid (e.g. a table or grid of st.columns), showing given
# cells and empty cells differently (e.g. bold givens, blank otherwise).
st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="centered",
)

st.title("🧩 Sudoku Solver")
st.caption(
    "Sudoku knowledge representation and inference using propositional logic"
)



# =============================================================================
# 2. LOAD THE PUZZLES
# =============================================================================

@st.cache_data
def load_puzzles(filename="puzzles.json"):
    """Read puzzles.json and cache the result.

    Streamlit reruns the Python script whenever a widget changes. Caching the
    JSON data avoids reopening and reparsing the same file on every rerun.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


pool = load_puzzles()

# These values are stored once at the top level of puzzles.json.
n = pool["n"]
box_h = pool["box_h"]
box_w = pool["box_w"]
puzzles = pool["puzzles"]


def convert_givens(raw_givens):
    """Convert JSON keys such as '2_7' into tuple keys such as (2, 7).

    sudoku_solver.py expects:
        {(row, column): value}

    while JSON stores object keys as strings:
        {"2_7": 8}
    """
    givens = {}

    for key, value in raw_givens.items():
        row, col = map(int, key.split("_"))
        givens[(row, col)] = int(value)

    return givens


# =============================================================================
# 3. BOARD DISPLAY HELPERS
# =============================================================================

def board_as_html(board, givens=None):
    """Return a Sudoku board as an HTML table.

    Parameters
    ----------
    board : dict[(int, int), int]
        Values currently displayed on the board. Missing cells are blank.

    givens : dict[(int, int), int] or None
        Original puzzle givens. If supplied, given values are shown in bold
        while inferred/solved values are shown normally.

    The thicker borders visually separate Sudoku boxes.
    """
    givens = givens or {}

    html = [
        "<table style='border-collapse:collapse; margin:auto;'>"
    ]

    for r in range(1, n + 1):
        html.append("<tr>")

        for c in range(1, n + 1):
            value = board.get((r, c), "")

            # A thicker line is drawn at each box boundary.
            top = "3px" if (r - 1) % box_h == 0 else "1px"
            left = "3px" if (c - 1) % box_w == 0 else "1px"
            bottom = "3px" if r % box_h == 0 else "1px"
            right = "3px" if c % box_w == 0 else "1px"

            # Initial clues are bold so that the user can distinguish them
            # from values inferred by the solver.
            if (r, c) in givens:
                content = f"<strong>{value}</strong>"
                background = "#eeeeee"
            else:
                content = str(value)
                background = "#ffffff"

            html.append(
                "<td style='"
                f"width:42px;height:42px;text-align:center;"
                f"font-size:20px;background:{background};"
                f"border-top:{top} solid #444;"
                f"border-left:{left} solid #444;"
                f"border-bottom:{bottom} solid #444;"
                f"border-right:{right} solid #444;"
                "'>"
                f"{content}"
                "</td>"
            )

        html.append("</tr>")

    html.append("</table>")
    return "".join(html)


def show_board(board, givens=None):
    """Render a Sudoku board in Streamlit."""
    st.markdown(board_as_html(board, givens), unsafe_allow_html=True)


# =============================================================================
# 4. REASONING-TRACE HELPERS
# =============================================================================

def decode_atom(proposition):
    """Convert an Is/Not proposition into a structured tuple.

    Examples
    --------
    Is3_2_4  -> ("Is", 3, 2, 4)
    Not3_2_4 -> ("Not", 3, 2, 4)

    Returns None for any proposition that does not use the assignment's
    Is/Not naming convention.
    """
    name = proposition.op

    if name.startswith("Is"):
        prefix = "Is"
        coordinates = name[2:]
    elif name.startswith("Not"):
        prefix = "Not"
        coordinates = name[3:]
    else:
        return None

    try:
        r, c, v = map(int, coordinates.split("_"))
        return prefix, r, c, v
    except ValueError:
        return None


def proposition_text(proposition):
    """Translate an internal proposition into plain English."""
    decoded = decode_atom(proposition)

    if decoded is None:
        return str(proposition)

    prefix, r, c, v = decoded

    if prefix == "Is":
        return f"cell ({r}, {c}) has value {v}"

    return f"value {v} is eliminated from cell ({r}, {c})"


def explain_rule(premises, conclusion):
    """Create a human-readable explanation for one Horn rule.

    This helper is for PRESENTATION only. It does not perform Sudoku
    inference and therefore does not duplicate the solver.

    The explanation recognises the two main rule families used by
    build_definite_kb():

    1. Elimination:
       Is(r,c,v) ==> Not(rr,cc,v)
       or
       Is(r,c,v) ==> Not(r,c,other_v)

    2. Last candidate:
       Not(...) & Not(...) & ... ==> Is(r,c,v)
    """
    conclusion_info = decode_atom(conclusion)

    if conclusion_info is None:
        return f"Inferred {conclusion}."

    prefix, r, c, v = conclusion_info

    # If the conclusion is Not(...), the rule is an elimination rule.
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
                f"is in the {relation}, value {v} is eliminated from "
                f"cell ({r}, {c})."
            )

    # If the conclusion is Is(...), and all premises are Not propositions for
    # the same cell, this is the last-candidate rule.
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
                f"For cell ({r}, {c}), values {values} have all been "
                f"eliminated. Therefore {v} is the last remaining candidate."
            )

    # Fallback explanation for an unexpected definite-clause shape.
    premise_text = "; ".join(proposition_text(p) for p in premises)
    return (
        f"Because {premise_text}, infer that "
        f"{proposition_text(conclusion)}."
    )


def backward_chain_with_trace(kb, query):
    """Prove a query while recording the successful reasoning path.

    The official boolean answer is still obtained with pl_bc_entails() in the
    UI below. This helper exists only to satisfy the assignment requirement
    for a user-friendly reasoning trace.

    The algorithm is a small, app-specific trace instrument:
    - facts are collected from the definite KB;
    - rules are indexed by their conclusion;
    - the requested goal is explored recursively;
    - an 'active' set prevents cycles;
    - only successful proof steps are retained in the displayed trace.

    Returns
    -------
    (bool, list[dict])
        bool: whether this trace search proved the query
        list: successful proof steps in dependency order
    """

    # Separate the KB into known facts and rules indexed by conclusion.
    facts = set()
    rules_by_conclusion = {}

    for clause in kb.clauses:
        premises, conclusion = parse_definite_clause(clause)

        if not premises:
            facts.add(conclusion)
        else:
            rules_by_conclusion.setdefault(conclusion, []).append(premises)

    # memo stores completed results so repeated subgoals do not need to be
    # proved from scratch during this trace.
    memo = {}

    # active stores goals on the current recursion path. Encountering an
    # already-active goal means following that rule would create a cycle.
    active = set()

    def prove(goal):
        """Return (proved?, successful_steps_for_goal)."""

        # Base case: a given/fact is already known.
        if goal in facts:
            return True, [{
                "type": "fact",
                "conclusion": goal,
                "premises": [],
            }]

        # Reuse a result already computed during this trace.
        if goal in memo:
            return memo[goal]

        # Cycle protection.
        if goal in active:
            return False, []

        active.add(goal)

        try:
            # OR across candidate rules:
            # proving any one rule whose conclusion is the goal is enough.
            for premises in rules_by_conclusion.get(goal, []):
                rule_steps = []
                all_premises_proved = True

                # AND across a rule's premises:
                # every premise must be proved before the conclusion follows.
                for premise in premises:
                    proved, premise_steps = prove(premise)

                    if not proved:
                        all_premises_proved = False
                        break

                    rule_steps.extend(premise_steps)

                if all_premises_proved:
                    rule_steps.append({
                        "type": "rule",
                        "conclusion": goal,
                        "premises": premises,
                    })

                    result = (True, rule_steps)
                    memo[goal] = result
                    return result

            result = (False, [])
            memo[goal] = result
            return result

        finally:
            active.remove(goal)

    proved, trace = prove(query)

    # A proof can contain the same fact/rule more than once because different
    # branches may share dependencies. Remove duplicate display entries while
    # preserving their first occurrence and reasoning order.
    unique_trace = []
    seen = set()

    for step in trace:
        key = (
            step["type"],
            step["conclusion"],
            tuple(step["premises"]),
        )

        if key not in seen:
            seen.add(key)
            unique_trace.append(step)

    return proved, unique_trace


def display_reasoning_trace(trace, query):
    """Render successful reasoning steps as expandable Streamlit cards."""
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

    step_number = 1

    for step in trace:
        conclusion = step["conclusion"]
        premises = step["premises"]

        if step["type"] == "fact":
            explanation = (
                f"{proposition_text(conclusion).capitalize()} is an initial "
                f"given fact in the puzzle."
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

            st.markdown(
                f"**Conclusion:** {proposition_text(conclusion)}"
            )

        step_number += 1

    st.success(
        f"Therefore, {proposition_text(query)} is entailed by the "
        f"definite-clause knowledge base."
    )


# =============================================================================
# 5. PUZZLE SELECTION
# =============================================================================

st.header("1. Select a puzzle")

puzzle_index = st.selectbox(
    "Puzzle",
    options=range(len(puzzles)),
    format_func=lambda i: (
        f"Puzzle {i + 1} — {puzzles[i]['given_count']} givens"
    ),
)

selected_puzzle = puzzles[puzzle_index]
givens = convert_givens(selected_puzzle["givens"])

st.write(
    f"Selected puzzle: **Puzzle {puzzle_index + 1}** "
    f"with **{selected_puzzle['given_count']} givens**."
)

show_board(givens, givens)

st.caption(
    "Bold shaded values are the original givens. Blank cells are unknown."
)



# =============================================================================
# 6. FULL-GRID AUTO SOLVER
# =============================================================================

# --- 2. Full-grid auto-solver, with algorithm selection ---
# TODO: a radio/selectbox letting the user choose forward chaining
# (solve_full_grid_fc) or backward chaining (solve_full_grid_bc).
# TODO: a button that times and calls the chosen solver on
# (n, box_h, box_w, givens), then displays the solved grid and the elapsed
# time.

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

    # Call the appropriate CORE function imported from sudoku_solver.py.
    if algorithm == "Forward chaining":
        solved = solve_full_grid_fc(n, box_h, box_w, givens)
    else:
        solved = solve_full_grid_bc(n, box_h, box_w, givens)

    elapsed = time.perf_counter() - start_time

    # A complete 9x9 solution should contain n*n cell assignments.
    if len(solved) == n * n:
        st.success(
            f"Solved using {algorithm} in {elapsed:.4f} seconds."
        )
        show_board(solved, givens)
    else:
        st.warning(
            f"The selected Horn inference method derived {len(solved)} of "
            f"{n * n} cells in {elapsed:.4f} seconds."
        )
        show_board(solved, givens)


# =============================================================================
# 7. TARGETED CELL ENTAILMENT QUERY
# =============================================================================

st.header("3. Ask a cell-value query")
# --- 3. Targeted cell entailment query ---
# TODO: number inputs for row (r), column (c), value (v).
# TODO: a button that builds the definite KB, calls
# pl_bc_entails(kb, atom('Is', r, c, v)), and displays True/False.

st.write(
    "Choose a row, column, and value. The application will test whether "
    "`Is(r,c,v)` is entailed by the definite-clause knowledge base using "
    "your backward-chaining implementation."
)

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    query_r = st.number_input(
        "Row",
        min_value=1,
        max_value=n,
        value=1,
        step=1,
    )

with input_col2:
    query_c = st.number_input(
        "Column",
        min_value=1,
        max_value=n,
        value=1,
        step=1,
    )

with input_col3:
    query_v = st.number_input(
        "Value",
        min_value=1,
        max_value=n,
        value=1,
        step=1,
    )

show_trace = st.checkbox(
    "Show tutor-mode reasoning trace",
    value=True,
    help=(
        "Displays a human-readable successful backward-chaining proof path "
        "when the proposition is entailed."
    ),
)

if st.button("Check entailment"):
    # Build the same definite KB used by the core Horn solver.
    kb = build_definite_kb(n, box_h, box_w, givens)

    query = atom(
        "Is",
        int(query_r),
        int(query_c),
        int(query_v),
    )

    start_time = time.perf_counter()

    # This is the required targeted query using the student's own BC function.
    entailed = pl_bc_entails(kb, query)

    elapsed = time.perf_counter() - start_time

    if entailed:
        st.success(
            f"True — cell ({query_r}, {query_c}) = {query_v} is entailed."
        )
    else:
        st.error(
            f"False — cell ({query_r}, {query_c}) = {query_v} is not "
            f"entailed by this knowledge base."
        )

    st.caption(f"Backward-chaining query time: {elapsed:.6f} seconds")


    # --- 4. Reasoning trace ("tutor mode") ---
    # TODO: instrument your forward- or backward-chaining approach to record each
    # reasoning step (which rule fired, on what premises, producing what
    # conclusion) as it answers the query above.
    # TODO: render that trace as human-readable output -- e.g. a sequence of
    # st.expander(...) blocks, one per step, each with a plain-English sentence
    # -- not a raw list/dict dump.
    # The trace helper is deliberately separate from the official boolean
    # query. It provides an explanation without changing the core solver.
    if show_trace:
        trace_proved, trace = backward_chain_with_trace(kb, query)

        if entailed and trace_proved:
            display_reasoning_trace(trace, query)
        elif entailed:
            st.info(
                "The core backward chainer proved the query, but the optional "
                "display helper did not reconstruct a proof path."
            )
        else:
            display_reasoning_trace([], query)




