import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detach_rocket",
    version="0.0.1",
    author="Gonzalo Uribarri & Federico Barone",
    description=(
        "DETACH-ROCKET reproduction and extended FordB evaluation "
        "with leakage-free preprocessing, additional baselines, "
        "and matched-budget multi-seed analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParastooAfshar/detach-rocket-fordb",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)