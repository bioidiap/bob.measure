#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'io', 'include')
include_dirs = [package_dir]

packages = ['bob-io >= 1.3']
version = '2.0.0a0'
define_macros = [("XBOB_IO_VERSION", '"%s"' % version)]

setup(

    name='xbob.io',
    version=version,
    description='Bindings for bob.io',
    url='http://github.com/anjos/xbob.io',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.io._externals",
        [
          "xbob/io/externals.cpp",
          ],
        packages = packages,
        define_macros = define_macros,
        include_dirs = include_dirs,
        ),
      Extension("xbob.io._library",
        [
          "xbob/io/bobskin.cpp",
          "xbob/io/file.cpp",
          "xbob/io/videoreader.cpp",
          "xbob/io/videowriter.cpp",
          "xbob/io/hdf5.cpp",
          "xbob/io/main.cpp",
          ],
        packages = packages,
        define_macros = define_macros,
        include_dirs = include_dirs,
        ),
      ],

    entry_points={
      'console_scripts': [
        'xbob_video_test.py = xbob.io.script.video_test:main',
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
