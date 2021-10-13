import setuptools
import sys


def remove_requirements(requirements, name, replace=None):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements


sys_platform = sys.platform

about = {}
with open("mindsdb_forecast_visualizer/__about__.py") as fp:
    exec(fp.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

dependency_links = []

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    url=about['__github__'],
    download_url=about['__pypi__'],
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={'project': ['requirements.txt']},
    install_requires=requirements,
    dependency_links=dependency_links,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.7"
)
