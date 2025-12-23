from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="derivapro",
    version="0.1.2",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "derivapro": [
            "*.md",
            "templates/*.html",  # HTML templates
            "static/**/*",  # all files in static and subfolders
        ],
    },
)
