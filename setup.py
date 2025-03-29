from setuptools import setup, find_packages

# Read requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vireya",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)
