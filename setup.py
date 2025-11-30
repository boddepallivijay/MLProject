from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->list[str]:
    '''
    This function will return the list of requirements
    '''
    requiremnts=[]
    with open(file_path) as file_obj:
        requiremnts=file_obj.readlines()
        requiremnts=[req.replace("\n","") for req in requiremnts]
        
        if HYPHEN_E_DOT in requiremnts:
            requiremnts.remove(HYPHEN_E_DOT)
            
    return requiremnts  



setup(
    name='MLProject',
    version='0.0.1',
    author='vijay',
    author_email='vkb9177@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')   
)