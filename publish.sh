#/bin/bash
git checkout master
python setup.py sdist bdist_wheel upload
