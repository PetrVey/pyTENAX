from setuptools import find_packages, setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="pyTENAX",
    version="0.0.1",
    description="Non-Asymptotic statistical model for eXtreme return levels (TENAX)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PetrVey/pyTENAX",
    author="Petr Vohnicky",
    author_email="petr.vohnicky@unipd.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11.5",
        "Operating System :: OS Independent",
    ],
    # install_requires=["bson >= 0.5.10"],
    # extras_require={
    #     "dev": ["pytest>=7.0", "twine>=4.0.2"],
    # },
    python_requires=">=3.11",
)
