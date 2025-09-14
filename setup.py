#responsible for creating my machine learning application as a package, can install and use it to other projects
from setuptools import find_packages, setup #to find all packages in my project
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements] #to remove new line character
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="mlproject", #name of my project
    version="0.0.1", #version of my project
    author="Chushmitha", #author of my project  
    author_email="chushmithabattula99@gmail.com", #author email
    packages=find_packages(), #find all packages in my project
    install_requires=get_requirements('requirements.txt')
        #list of dependencies required for my project
        # we cant list all the packages required this way, it can be difficult to manage
        )