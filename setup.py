"""Setup brv-bench CLI application."""

__author__ = ["Danh Doan"]
__email__ = ["danhdoancv@gmail.com"]
__date__ = "2025/05/21"
__status__ = "development"


# =============================================================================


from setuptools import find_packages, setup

# =============================================================================


def parse_requirements(filename: str) -> list[str]:
    """Parse packages from `requirements.txt` file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="brv_bench",
    version="0.1.0",
    description="Benchmark suite for evaluating retrieval quality, latency, and diversity of AI agent context systems",
    author="Danh Doan",
    author_email="danhdoancv@gmail.com",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.12",
)


# =============================================================================
