import numpy as np
import re
import math

# -------------------------------------------------
# Characters indicating mathematical or logical
# expression density within the text
# -------------------------------------------------
MATH_SYMBOLS = "+-*/%=<>!&|^[]'"

# -------------------------------------------------
# Keywords typically used in decision-based outputs
# (e.g., boolean or validation problems)
# -------------------------------------------------
DECISION_KEYWORDS = [
    "yes", "no",
    "yes or no",
    "true", "false",
    "possible", "impossible",
    "valid", "invalid"
]

# -------------------------------------------------
# Neutral algorithmic and data-structure keywords
# -------------------------------------------------
KEYWORDS = [
    # Basic comparisons and operations
    "min", "minimum", "max", "maximum",
    "increase", "decrease", "increment", "decrement",
    "greater", "smaller", "largest", "smallest",
    "sum", "difference", "product",

    # Arrays and sequences
    "array", "subarray", "prefix", "suffix",
    "sequence", "subsequence",

    # Grids and matrices
    "grid", "matrix",
    "row", "rows",
    "column", "columns",
    "dimension", "dimensions",
    "2d", "two dimensional",

    # Searching and sorting
    "sort", "sorted",
    "binary search", "two pointers",

    # Queries and ranges
    "query", "queries",
    "range", "update",

    # Geometry and coordinates
    "coordinate", "coordinates",
    "point", "distance",

    # Common data structures
    "stack", "queue", "deque",
    "heap", "priority queue",
    "set", "map", "hash", "dictionary",

    # Trees and graphs
    "tree", "binary tree", "bst",
    "graph", "directed", "undirected",
    "vertex", "vertices",
    "edge", "edges",
    "dfs", "bfs",
    "shortest", "path",

    # Bit manipulation concepts
    "bit", "bits",
    "binary", "xor", "and", "or", "shift",
    "bitmask",

    # Mathematical and number theory terms
    "gcd", "lcm",
    "modulo", "prime"
]


def extract_features(text: str, is_output: bool = False):
    

    text = text.lower()

    # -------------------------------------------------
    # Detection of decision-style output problems
    # -------------------------------------------------
    decision_flag = 0
    if is_output:
        for k in DECISION_KEYWORDS:
            if k in text:
                decision_flag = 1
                break

    # -------------------------------------------------
    # Basic textual statistics
    # -------------------------------------------------
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = text.count(".") + text.count("?")

    math_count = sum(c in MATH_SYMBOLS for c in text)
    digit_count = sum(c.isdigit() for c in text)

    avg_word_len = char_count / max(word_count, 1)

    # -------------------------------------------------
    # Constraint-related numerical features
    # Extracted conservatively from explicit numbers
    # -------------------------------------------------
    numbers = [int(x) for x in re.findall(r"\d+", text)]

    if numbers:
        max_val = max(numbers)
        max_log_constraint = math.log10(max_val + 1)
        large_constraint_flag = 1 if max_val >= 100_000 else 0
        constraint_count = len(numbers)
    else:
        max_log_constraint = 0.0
        large_constraint_flag = 0
        constraint_count = 0

    # -------------------------------------------------
    # Keyword frequency-based features
    # -------------------------------------------------
    keyword_counts = [text.count(k) for k in KEYWORDS]

    # -------------------------------------------------
    # Final feature vector construction
    # -------------------------------------------------
    return np.array(
        [
            word_count,
            char_count,
            sentence_count,
            math_count,
            digit_count,
            avg_word_len,
            decision_flag,
            max_log_constraint,
            large_constraint_flag,
            constraint_count
        ] + keyword_counts,
        dtype=np.float32
    ).reshape(1, -1)
