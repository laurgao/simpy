from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="simpy",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(where="src"),  # Tells setuptools to look for packages in the 'src' directory
    package_dir={"": "src"},  # Sets the root of packages to 'src'
    description="Computer algebra system & symbolic calculus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest>=5.2",  # Dependencies for testing
        ],
    },
    python_requires=">=3.8",
)
