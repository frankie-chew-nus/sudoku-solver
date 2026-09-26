"""IT5005 Assignment 1: Sudoku propositional-logic solver.

Core solver implementation.  This file intentionally imports only from
``utils.py`` and ``logic_.py``, as required by the assignment specification.
"""

from utils import *
from logic_ import *


# Do not change this function; it is used to create atomic propositions.
def atom(prefix, r, c, v):
    """prefix is 'Is' or 'Not'. Returns the Expr for e.g. Is3_2_4."""
    return expr(f'{prefix}{r}_{c}_{v}')


def _peers(n, box_h, box_w, r, c):
    """Return the cells sharing a row, column, or box with (r, c)."""
    peers = set()

    for k in range(1, n + 1):
        if k != c:
            peers.add((r, k))
        if k != r:
            peers.add((k, c))

    r0 = ((r - 1) // box_h) * box_h + 1
    c0 = ((c - 1) // box_w) * box_w + 1
    for rr in range(r0, r0 + box_h):
        for cc in range(c0, c0 + box_w):
            if (rr, cc) != (r, c):
                peers.add((rr, cc))

    return peers


def build_general_kb(n, box_h, box_w, givens):
    """Return a PropKB encoding this n x n Sudoku's constraints plus the given
    cells, as general clauses.

    Parameters
    ----------
    n, box_h, box_w : int
    givens : dict[(int, int), int]

    Returns
    -------
    PropKB
    """

    kb = PropKB() 

    #1. Every cell has at least one value from {1, . . . , n}.
    for r in range (1, n + 1):
      for c in range(1, n + 1):
        kb.tell(Expr('|', *[atom('Is', r, c, v) for v in range (1, n + 1)]))
  
    #2. Every cell has at most one value from {1, . . . , n}. (no cell holds two digits at once
    for r in range(1, n+1):
      for c in range(1, n+1):
        for v in range(1, n+1):
          for v1 in range(v+1, n+1):
            kb.tell(~atom('Is', r, c, v) | ~atom('Is', r, c, v1))
    
    #3. No two cells in the same row hold the same value
    for r in range(1, n+1):
      for c in range(1, n+1):
        for c1 in range(c+1, n+1):
          for v in range(1, n+1):
            kb.tell(~atom('Is', r, c, v) | ~atom('Is', r, c1, v))

    #4 No two cells in the same column hold the same value.
    for r in range(1, n+1):
      for r1 in range(r+1, n+1):
        for c in range(1, n+1):
          for v in range(1, n+1):
            kb.tell(~atom('Is', r, c, v) | ~atom('Is', r1, c, v))

    #5 No two cells in the same box hold the same value.
    #Trace from first cell of every box
    for r in range(1, n-1, box_h):
      for c in range(1, n-1, box_w):
        #trace each cell in the box and stack
        #Stack each cell into a linear list
        linearCells = []
        for rBox in range(r,r + box_h):
          for cBox in range (c,c + box_w):
            linearCells.append((rBox, cBox))
        #print(f"liner box: {linearCells}")
        for c in range (0, n):
          for c1 in range (c+1, n):
            #for v in range(1, n+1):
            (rAnchor, cAchor) = linearCells[c]
            (rCompare, cCompare) = linearCells[c1]
            #print(f"lhs{rAnchor, cAchor}, rhs:{rCompare,cCompare}, {c,c1}")
            for v in range(1, n+1):
              kb.tell(~atom('Is', rAnchor, cAchor, v) | ~atom('Is', rCompare, cCompare, v))
            


    for (r, c), v in givens.items():
      kb.tell(atom('Is', r, c, v))

    return kb
    



def build_definite_kb(n, box_h, box_w, givens):
    """Return a PropDefiniteKB using elimination and last-candidate rules.

    ``Not_r_c_v`` is a positive proposition meaning that value ``v`` has
    been eliminated from cell ``(r,c)``.  This lets all rules remain definite
    Horn clauses even though ordinary Sudoku constraints contain negation.
    """
    kb = PropDefiniteKB()

    # Givens are the initial positive facts.
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                is_rcv = atom('Is', r, c, v)

                # At most one value per cell:
                # Is(r,c,v) ==> Not(r,c,other_v).
                for other_v in range(1, n + 1):
                    if other_v != v:
                        kb.tell(Expr('==>', is_rcv, atom('Not', r, c, other_v)))

                # Row/column/box uniqueness:
                # Is(r,c,v) ==> peers cannot also contain v.
                for rr, cc in _peers(n, box_h, box_w, r, c):
                    kb.tell(Expr('==>', is_rcv, atom('Not', rr, cc, v)))

            # At least one value, expressed in Horn form by last candidate:
            # if every other value has been eliminated, the remaining one is Is.
            for v in range(1, n + 1):
                premises = [atom('Not', r, c, other_v)
                            for other_v in range(1, n + 1)
                            if other_v != v]
                kb.tell(Expr('==>', associate('&', premises), atom('Is', r, c, v)))

    # Speed up the library's forward chainer without changing logic_.py:
    # its default clauses_with_premise() linearly scans the whole KB each time.
    premise_index = {}
    for clause in kb.clauses:
        if clause.op == '==>':
            for premise in conjuncts(clause.args[0]):
                premise_index.setdefault(premise, []).append(clause)
    kb.clauses_with_premise = lambda p: premise_index.get(p, [])

    return kb


def solve_full_grid_fc(n, box_h, box_w, givens):
    """Solve every cell using the definite KB and library forward chaining."""
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                if pl_fc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break

    return solved


def pl_bc_entails(kb, query):
    """Return whether ``query`` follows from a propositional definite KB.

    This is goal-driven backward chaining with tabling.  Candidate rules are
    indexed by conclusion; each rule is proved by recursively proving all of
    its premises.  Because the Sudoku Horn encoding contains harmless cycles
    (Is -> Not -> Is), failed subgoals are cached only for one proof pass.
    Passes repeat while new goals are proved, which reaches the least Horn
    fixed point while retaining backward, query-directed rule exploration.
    """
    compiled = getattr(kb, '_bc_compiled', None)
    if compiled is None:
        facts = set()
        rules_by_conclusion = {}
        for clause in kb.clauses:
            premises, conclusion = parse_definite_clause(clause)
            if not premises:
                facts.add(conclusion)
            else:
                rules_by_conclusion.setdefault(conclusion, []).append(premises)
        compiled = (facts, rules_by_conclusion)
        kb._bc_compiled = compiled
    facts, rules_by_conclusion = compiled

    proved = getattr(kb, '_bc_proved', None)
    if proved is None:
        proved = set(facts)
        kb._bc_proved = proved

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
                        return True
            finally:
                active.remove(goal)

            failed_this_pass.add(goal)
            return False

        if prove(query):
            return True
        if len(proved) == before:
            return False


def solve_full_grid_bc(n, box_h, box_w, givens):
    """Solve every cell using the definite KB and backward chaining."""
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                if pl_bc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break

    return solved
