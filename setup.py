from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))
from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()


setup(
    name="bob.measure",
    version=open("version.txt").read().rstrip(),
    description="Evaluation metrics for Bob",
    url="http://gitlab.idiap.ch/bob/bob.measure",
    license="BSD",
    author="Andre Anjos",
    author_email="andre.anjos@idiap.ch",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        # main entry for bob measure cli
        "bob.cli": [
            "measure  = bob.measure.script.measure:measure",
        ],
        # bob measure scripts
        "bob.measure.cli": [
            "evaluate = bob.measure.script.commands:evaluate",
            "metrics = bob.measure.script.commands:metrics",
            "multi-metrics = bob.measure.script.commands:multi_metrics",
            "roc = bob.measure.script.commands:roc",
            "det = bob.measure.script.commands:det",
            "epc = bob.measure.script.commands:epc",
            "hist = bob.measure.script.commands:hist",
            "gen = bob.measure.script.gen:gen",
        ],
    },
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
