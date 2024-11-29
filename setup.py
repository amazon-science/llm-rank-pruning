from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="llmrank",
    version="0.0.1",
    description=(
        "A large language model pruning package based on the"
        " PageRank centrality measure"
    ),
    package_dir={"llmrank": "llmrank"},
    packages=find_packages(where="."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aws.dev/adavidho/llm-rank-pruning",
    author="David Hoffmann",
    author_email="david.hoffmann[at]mail.de",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "accelerate>=0.31.0,<0.32.0",
        "datasets>=2.20.0,<2.21.0",
        "datetime>=5.5,<5.6",
        "lm_eval>=0.4.3,<0.5.0",
        "numpy>=1.24.3,<=1.26.4",
        "sentencepiece>=0.2.0,<0.3.0",
        "torch>=2.0.0,<2.4.0", 
        "transformers>=4.41.0,<4.42.0",
        "wandb>=0.17.3,<0.18.0",
    ],
    python_requires=">=3.10,<3.13",
)
