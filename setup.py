from setuptools import find_packages, setup

setup(
    name="centrodip",
    version="1.0.0",
    description="Find hypomethylated regions in centromeres",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jmenendez98/centrodip",
    author="Julian Menendez",
    author_email="jmmenend@ucsc.edu",
    license="MIT",
    packages=find_packages(),
    project_urls={
        "GitHub": "https://github.com/jmenendez98/centrodip",
        "Lab Website": "https://migalab.com/",
    },
    install_requires=[
        "numpy>=1.21.5",
        "scipy>=1.7.3",
        "matplotlib>=3.5.1"
    ],
    entry_points={
        "console_scripts": [
            "centrodip=centrodip.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest",
        ]
    },
)
