#!/usr/bin/env python3
# code.py
# Global alignment (Needleman-Wunsch) CLI version converted from the notebook.
# Underlying alignment functions are unchanged; this file just adds argument parsing.

"""
Global Sequence Alignment using the Needleman-Wunsch Algorithm

Implement a global alignment program for two biological sequences using dynamic programming. The program allows the user to choose between:
1. A simple scoring scheme (match, mismatch, gap), or
2. The PAM250 substitution matrix for protein sequence alignment.

Inputs:
* Sequence A (string)
* Sequence B (string)
* Scoring choice:
    * Simple Scoring:
        * Match score (integer)
        * Mismatch score (integer)
        * Gap penalty (integer)
    * Scoring with PAM250 matrix. Gap penalty still provided by user

Outputs:
1. Optimal alignment score
2. One optimal global alignment

Notes
* Global alignment using Needleman-Wunsch
* Single gap penalty only
"""

# Load dependencies
import numpy as np
from typing import Dict, Tuple

# PAM250 Definition
# Amino acid order used in PAM250
PAM250_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

# PAM250 substitution matrix
# Rows and columns follow PAM250_AA_ORDER
PAM250 = np.array([
    [ 2,-2, 0, 0,-2, 0, 0, 1,-1,-1,-2,-1,-1,-4, 1, 1, 1,-6,-3, 0],
    [-2, 6, 0,-1,-4, 1,-1,-3, 2,-2,-3, 3, 0,-4, 0, 0,-1, 2,-4,-2],
    [ 0, 0, 2, 2,-4, 1, 1, 0, 2,-2,-3, 1,-2,-4, 0, 1, 0,-4,-2,-2],
    [ 0,-1, 2, 4,-5, 2, 3, 1, 1,-2,-4, 0,-3,-6,-1, 0, 0,-7,-4,-2],
    [-2,-4,-4,-5,12,-5,-5,-3,-3,-2,-6,-5,-5,-4,-3, 0,-2,-8, 0,-2],
    [ 0, 1, 1, 2,-5, 4, 2,-1, 3,-2,-2, 1,-1,-5, 0,-1,-1,-5,-4,-2],
    [ 0,-1, 1, 3,-5, 2, 4, 0, 1,-2,-3, 0,-2,-5,-1, 0, 0,-7,-4,-2],
    [ 1,-3, 0, 1,-3,-1, 0, 5,-2,-3,-4,-2,-3,-5, 0, 1, 0,-7,-5,-1],
    [-1, 2, 2, 1,-3, 3, 1,-2, 6,-2,-2, 0,-2,-2, 0,-1,-1,-3, 0,-2],
    [-1,-2,-2,-2,-2,-2,-2,-3,-2, 5, 2,-2, 2, 1,-2,-1, 0,-5,-1, 4],
    [-2,-3,-3,-4,-6,-2,-3,-4,-2, 2, 6,-3, 4, 2,-3,-3,-2,-2,-1, 2],
    [-1, 3, 1, 0,-5, 1, 0,-2, 0,-2,-3, 5, 0,-5,-1, 0, 0,-3,-4,-2],
    [-1, 0,-2,-3,-5,-1,-2,-3,-2, 2, 4, 0, 6, 0,-2,-2,-1,-4,-2, 2],
    [-4,-4,-4,-6,-4,-5,-5,-5,-2, 1, 2,-5, 0, 9,-5,-3,-3, 0, 7,-1],
    [ 1, 0, 0,-1,-3, 0,-1, 0, 0,-2,-3,-1,-2,-5, 6, 1, 0,-6,-5,-1],
    [ 1, 0, 1, 0, 0,-1, 0, 1,-1,-1,-3, 0,-2,-3, 1, 2, 1,-2,-3,-1],
    [ 1,-1, 0, 0,-2,-1, 0, 0,-1, 0,-2, 0,-1,-3, 0, 1, 3,-5,-3, 0],
    [-6, 2,-4,-7,-8,-5,-7,-7,-3,-5,-2,-3,-4, 0,-6,-2,-5,17, 0,-6],
    [-3,-4,-2,-4, 0,-4,-4,-5, 0,-1,-1,-4,-2, 7,-5,-3,-3, 0,10,-2],
    [ 0,-2,-2,-2,-2,-2,-2,-1,-2, 4, 2,-2, 2,-1,-1,-1, 0,-6,-2, 4]
], dtype=int)

# Map amino acid -> index
PAM250_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(PAM250_AA_ORDER)}

def pam250_score(a: str, b: str) -> int:
    """Return PAM250 substitution score for amino acids a and b."""
    return PAM250[PAM250_INDEX[a], PAM250_INDEX[b]]

def make_scoring_function(
    use_pam250: bool,
    match_score: int = 1,
    mismatch_score: int = -1
):
    """
    Returns a function score(a, b) that computes the substitution score
    for characters a and b.

    If use_pam250 is True, PAM250 is used (protein sequences).
    Otherwise, simple match/mismatch scoring is used.
    """
    if use_pam250:
        def score(a: str, b: str) -> int:
            return int(pam250_score(a, b))
    else:
        def score(a: str, b: str) -> int:
            return match_score if a == b else mismatch_score

    return score

def init_global_alignment_matrices(seq1: str, seq2: str, gap_penalty: int):
    """
    Initialize DP and traceback matrices for Needleman-Wunsch global alignment.

    Let:
      n = len(seq1)
      m = len(seq2)

    The DP score matrix F and traceback matrix TB both have shape (n+1) x (m+1).

    Interpretation:
      F[i, j] stores the optimal alignment score for the prefixes:
        seq1[0:i] and seq2[0:j]

    Example (n = 3, m = 3):

           j=0    j=1     j=2    j=3
        -----------------------------
    i=0 | F[0,0] F[0,1] F[0,2] F[0,3]
    i=1 | F[1,0] F[1,1] F[1,2] F[1,3]
    i=2 | F[2,0] F[2,1] F[2,2] F[2,3]
    i=3 | F[3,0] F[3,1] F[3,2] F[3,3]

    Initialization:
      - F[0,0] = 0
      - First column (j=0): cumulative gap penalties (seq1 aligned to gaps)
      - First row (i=0): cumulative gap penalties (seq2 aligned to gaps)

    Traceback directions stored in TB:
      'D' = diagonal (match/mismatch)
      'U' = up (gap in seq2)
      'L' = left (gap in seq1)
    """
    n = len(seq1)
    m = len(seq2)

    # Score matrix
    F = np.zeros((n + 1, m + 1), dtype=int)

    # Traceback matrix
    TB = np.empty((n + 1, m + 1), dtype=object)
    TB[0, 0] = None

    # First column
    for i in range(1, n + 1):
        F[i, 0] = F[i - 1, 0] + gap_penalty
        TB[i, 0] = "U"

    # First row
    for j in range(1, m + 1):
        F[0, j] = F[0, j - 1] + gap_penalty
        TB[0, j] = "L"

    return F, TB

def fill_global_alignment_matrices(
    seq1: str,
    seq2: str,
    F: np.ndarray,
    TB: np.ndarray,
    score_fn,
    gap_penalty: int
):
    """
    Fill DP score matrix F and traceback matrix TB using the Needleman-Wunsch recurrence.
    Tie-breaking is deterministic: D > U > L.
    """
    n = len(seq1)
    m = len(seq2)

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            diag = F[i - 1, j - 1] + score_fn(seq1[i - 1], seq2[j - 1])
            up   = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty

            best_score = diag
            best_move = "D"

            if up > best_score:
                best_score = up
                best_move = "U"

            if left > best_score:
                best_score = left
                best_move = "L"

            F[i, j] = best_score
            TB[i, j] = best_move

def traceback_global_alignment(seq1: str, seq2: str, TB: np.ndarray):
    """
    Reconstruct one optimal global alignment using the traceback matrix TB.
    Returns (aligned_seq1, aligned_seq2).
    """
    i = len(seq1)
    j = len(seq2)

    aligned_seq1 = []
    aligned_seq2 = []

    while i > 0 or j > 0:
        move = TB[i, j]

        if move == "D":
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1

        elif move == "U":
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1

        elif move == "L":
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

        else:
            raise ValueError(f"Invalid traceback direction at ({i}, {j})")

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return "".join(aligned_seq1), "".join(aligned_seq2)

import argparse

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Global alignment (Needleman-Wunsch) with either simple scoring or PAM250."
    )

    p.add_argument("--seq1", required=True, help="First sequence (DNA/RNA/protein).")
    p.add_argument("--seq2", required=True, help="Second sequence (DNA/RNA/protein).")
    p.add_argument("--gap", type=int, required=True, help="Gap penalty (single constant, e.g., -2).")

    scoring = p.add_mutually_exclusive_group(required=True)
    scoring.add_argument("--simple", action="store_true",
                         help="Use simple match/mismatch scoring (requires --match and --mismatch).")
    scoring.add_argument("--pam250", action="store_true",
                         help="Use PAM250 substitution matrix (protein sequences).")

    p.add_argument("--match", type=int, default=None, help="Match score (only with --simple).")
    p.add_argument("--mismatch", type=int, default=None, help="Mismatch score (only with --simple).")

    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)

    seq1 = args.seq1
    seq2 = args.seq2
    gap_penalty = args.gap

    if args.simple:
        if args.match is None or args.mismatch is None:
            raise SystemExit("Error: --simple requires both --match and --mismatch.")
        score_fn = make_scoring_function(
            use_pam250=False,
            match_score=args.match,
            mismatch_score=args.mismatch
        )
    else:
        # PAM250
        bad1 = sorted({c for c in seq1 if c not in PAM250_INDEX})
        bad2 = sorted({c for c in seq2 if c not in PAM250_INDEX})
        if bad1 or bad2:
            raise SystemExit(
                "Error: --pam250 requires protein sequences using these amino acids: "
                f"{''.join(PAM250_AA_ORDER)}. "
                f"Invalid in seq1: {bad1 if bad1 else 'none'}, invalid in seq2: {bad2 if bad2 else 'none'}"
            )
        score_fn = make_scoring_function(use_pam250=True)

    F, TB = init_global_alignment_matrices(seq1, seq2, gap_penalty)
    fill_global_alignment_matrices(seq1, seq2, F, TB, score_fn, gap_penalty)
    aligned1, aligned2 = traceback_global_alignment(seq1, seq2, TB)

    print(f"Optimal alignment score: {F[len(seq1), len(seq2)]}")
    print(aligned1)
    print(aligned2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
