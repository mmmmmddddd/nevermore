# -*- coding: utf-8 -*-

import io
import re

from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE', 'r') as f:
    license = f.read()

with open('requirements.txt', 'r') as f:
    requires = []
    for line in f:
        line = line.strip()
        if not line.startswith('#'):
            requires.append(line)

with io.open("src/nevermore/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
    name='nevermore',
    version=version,
    description='TODO',
    long_description=readme,
    author='user',
    author_email='duino472365351@gmail.com',
    url='https://github.com/user/nevermore',
    license=license,
    platform='linux',
    zip_safe=False,
    packages=find_packages("src"),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=requires,
    entry_points={"console_scripts": ["nevermore = nevermore.cli:main"]},
)
