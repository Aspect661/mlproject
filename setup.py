from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(filepath: str) -> List[str]:
    
    # Function will return list of requirements from requirements.txt
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.read().splitlines()

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements


setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Aspect",
    author_email = "tuhindas01.official@gmail.com",
    pacakages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)