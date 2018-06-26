#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core', 'bob.io.base', 'bob.math']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['blitz >= 0.10', 'boost']
boost_modules = ['system']

setup(

    name='bob.measure',
    version=version,
    description='Evaluation metrics for Bob',
    url='http://gitlab.idiap.ch/bob/bob.measure',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,

    ext_modules = [
      Extension("bob.measure.version",
        [
          "bob/measure/version.cpp",
        ],
        packages = packages,
        version = version,
        bob_packages = bob_packages,
        boost_modules = boost_modules,
      ),

      Extension("bob.measure._library",
        [
          "bob/measure/cpp/error.cpp",
          "bob/measure/main.cpp",
        ],
        packages = packages,
        version = version,
        bob_packages = bob_packages,
        boost_modules = boost_modules,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    entry_points={
      # main entry for bob measure cli
      'bob.cli': [
          'measure  = bob.measure.script.measure:measure',
      ],

      # bob measure scripts
      'bob.measure.cli': [
          'evaluate = bob.measure.script.commands:evaluate',
          'metrics = bob.measure.script.commands:metrics',
          'multi-metrics = bob.measure.script.commands:multi_metrics',
          'roc = bob.measure.script.commands:roc',
          'det = bob.measure.script.commands:det',
          'epc = bob.measure.script.commands:epc',
          'hist = bob.measure.script.commands:hist',
          'gen = bob.measure.script.gen:gen',
      ],
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
