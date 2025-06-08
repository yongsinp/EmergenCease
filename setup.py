from setuptools import setup, find_packages

with open("requirements.txt", "r") as r:
    requirements = [line for line in r if (line := line.strip()) and not line.startswith("#")]

setup(
    name="cap_translator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.11",
    author="Yongsin Park",
    author_email="yongsin.nlp@gmail.com",
    description="This project builds a pipeline to generate multilingual emergency alerts using FCC Alert Templates in 13 languages. It classifies the emergency type, extracts key information, and fills out multilingual templates in the Common Alerting Protocol format. The goal is to improve alert accessibility for non-native speakers.",
)
