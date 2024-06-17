from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []

    HYPHEN_E_DOT = "-e ."

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Loan Default Prediction Model',
    version='0.0.1',
    author='Tavishi Jaglan',
    author_email='tavishi.1402@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)