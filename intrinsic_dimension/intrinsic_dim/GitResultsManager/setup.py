#! /usr/bin/env python

from distutils.core import setup

with open('README.md') as ff:
    long_description = ff.read()
url='https://github.com/yosinski/GitResultsManager'

setup(name='GitResultsManager',
      description='The GitResultsManager Python module and scripts (resman) for keeping track of research results using Git.',
      long_description=long_description,
      version='0.3.3',
      url=url,
      author='Jason Yosinski',
      author_email='git_results_manager.jyo@0sg.net',
      packages=['GitResultsManager'],
      license='LICENSE.txt',
      scripts=['bin/git-recreate', 'bin/rmtd', 'bin/resman', 'bin/resman-td'],
)
