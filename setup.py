from setuptools import setup, find_packages

import deer

NAME = 'deer'
VERSION = '0.4.2'
AUTHOR = "Vincent Francois-Lavet"
AUTHOR_EMAIL = "vincent.francois@gmail.com"
URL = 'https://github.com/VinF/deer'
DESCRIPTION = 'Framework for deep reinforcement learning'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Utilities',
    'Topic :: Software Development :: Libraries',
]

if __name__ == '__main__':
    setup(name=NAME,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license='BSD',
          classifiers=CLASSIFIERS,
          platforms='any',
          packages=find_packages())