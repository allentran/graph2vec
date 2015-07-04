__author__ = 'porky-chu'

from setuptools import find_packages, setup


if __name__ == '__main__':
    name = 'graph2vec'
    setup(
        name=name,
        version="0.0.2",
        author='Allen Tran',
        author_email='realallentran@gmail.com',
        description='meaningful vector representations of nodes',
        url='https://github.com/allentran/graph2vec',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Programming Language :: Python',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            ],
        setup_requires=[
            'setuptools>=3.4.4',
            ],
    )
