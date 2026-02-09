"""Ultilities for Number processing."""

# =============================================================================


def fequal(a: float, b: float, eps: float = 1e-6) -> bool:
    """Check if 2 floating-points numbers are equal or not."""
    return abs(a - b) < eps

# =============================================================================