import setuptools
import sys


sys_platform = sys.platform

about = {}
with open("mindsdb_forecast_visualizer/__about__.py") as fp:
    exec(fp.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "plotly>=4.12",
    "lightwood>=1.0.0",
    "mindsdb>=2.51.1"
    ]

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
    install_requires=install_requires,
    dependency_links=[],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.7"
)
