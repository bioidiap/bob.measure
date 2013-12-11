#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

packages = ['bob-measure >= 1.3']
version = '2.0.0a0'

setup(

    name='xbob.measure',
    version=version,
    description='Bindings for bob.measure',
    url='http://github.com/anjos/xbob.measure',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
      'xbob.math',
      'xbob.io',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.measure._library",
        [
          "xbob/measure/main.cpp",
          ],
        packages = packages,
        version = version,
        ),
      ],

    entry_points={
      'console_scripts': [
        'xbob_compute_perf.py = xbob.measure.script.compute_perf:main',
        'xbob_eval_threshold.py = xbob.measure.script.eval_threshold:main',
        'xbob_apply_threshold.py = xbob.measure.script.apply_threshold:main',
        'xbob_plot_cmc.py = xbob.measure.script.plot_cmc:main',
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
