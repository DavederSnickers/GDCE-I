from setuptools import find_packages, setup

# Installed with `pip install -e .` only so that `import src...` resolves the
# same way inside and outside a container.  The reproduction scripts set
# PYTHONPATH instead and do not need this.
setup(
    name='gdce-i',
    version='1.0.0',
    description='Graph Diffusion Counterfactual Explanation via Inversion',
    url='https://github.com/DavederSnickers/GDCE-I',
    license='MIT',
    packages=find_packages(include=['src', 'src.*', 'configs']),
)
