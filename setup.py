#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz']))
from bob.blitz.extension import Extension

packages = ['bob-measure >= 1.2.2']
version = '2.0.0a0'

setup(

    name='bob.measure',
    version=version,
    description='Bindings for bob.measure',
    url='http://github.com/bioidiap/bob.measure',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.math',
      'bob.io.base',
      'matplotlib',
    ],

    namespace_packages=[
      "bob",
      ],

    ext_modules = [
      Extension("bob.measure.version",
        [
          "bob/measure/version.cpp",
          ],
        packages = packages,
        version = version,
        ),
      Extension("bob.measure._library",
        [
          "bob/measure/main.cpp",
          ],
        packages = packages,
        version = version,
        ),
      ],

    entry_points={
      'console_scripts': [
        'bob_compute_perf.py = bob.measure.script.compute_perf:main',
        'bob_eval_threshold.py = bob.measure.script.eval_threshold:main',
        'bob_apply_threshold.py = bob.measure.script.apply_threshold:main',
        'bob_plot_cmc.py = bob.measure.script.plot_cmc:main',
        ],
      },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
