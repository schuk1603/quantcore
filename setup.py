from setuptools import setup, find_packages

setup(
    name="quantcore",
    version="1.0.0",
    author="Samuel Chukwuma",
    description="High-performance quantitative research and portfolio management platform",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "yfinance>=0.2.28",
        "plotly>=5.17",
        "dash>=2.14",
        "dash-bootstrap-components>=1.5",
        "cvxpy>=1.4",
        "pyportfolioopt>=1.5",
        "filterpy>=1.4",
        "numba>=0.58",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
