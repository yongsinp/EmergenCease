from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as r:
    requirements = [stripped for line in r if (stripped := line.strip()) and not stripped.startswith("#")]

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
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yongsinp/EmergenCease",  # Replace with your repo
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="emergency, alerts, translation, multilingual, CAP, Common Alerting Protocol, crisis, disaster, nlp, natural language processing",
)
