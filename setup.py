from setuptools import setup
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r', encoding='utf8').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='sdeint2',
    version=find_version('sdeint2', '__init__.py'),
    url='https://github.com/philipschw/sdeint2',
    bugtrack_url='https://github.com/philipschw/sdeint2/issues',
    license='GPLv3+',
    author='Philip Schwedler',
    install_requires=['numpy>=1.6', 'sdeint'],
    author_email='phil.schwedler@web.de',
    description='Numerical integration of stochastic differential equations (SDE) - Extension of sdeint',
    long_description='tbd',
    packages=['sdeint2'],
    platforms='any',
    zip_safe=False,
    keywords = ['stochastic', 'differential equations', 'SDE', 'SODE'],
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        ]
)