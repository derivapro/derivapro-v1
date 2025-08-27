from setuptools import setup, find_packages

setup(
    name="derivapro",
    version="0.1.0",
    packages=find_packages(),  # will find the derivapro/ package
    install_requires=[
        "flask",  # plus any other dependencies from requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "derivapro-launch=derivapro:launch",  # optional CLI command
        ],
    },
)
