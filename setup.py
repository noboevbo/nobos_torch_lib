from setuptools import setup, find_packages

setup(name='nobos_torch_lib',
      version='0.1',
      description='Nobos PyTorch Library. Various datasets, models and utils for PyTorch models.',
      author='Dennis Ludl',
      author_email='dennis@noboevbo.de',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)