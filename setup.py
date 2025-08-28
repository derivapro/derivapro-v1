from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="derivapro",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,  # ensures package_data and MANIFEST.in are used
    package_data={
        "derivapro": ["*.md", "templates/*.html"],  # include Markdown and HTML templates
    },
)

