from setuptools import setup, find_packages

setup(
    name='contact_graspnet',
    version='1.0.0',
    description='6-DoF Grasp Generation for Robotic Manipulation',
    packages=find_packages(include=['contact_graspnet', 'contact_graspnet.*']),  # Include only the contact_graspnet module
    include_package_data=True,
    license='MIT License',
    long_description=open('README.md').read(),
)