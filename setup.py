from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''This function will return list of requirements'''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [require.replace("\n", "") for require in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ML Project',
    version='0.0.1',
    author='Om Dalvi',
    author_email='omdalvi1905@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)