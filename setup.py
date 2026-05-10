from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="para_eff_pt",
    version="0.1.0",
    author="Quan Wei",
    author_email="weiqua0128@gmail.com",
    description="Parameter Efficient Pretraining",
    url="https://github.com/quanwei0/parameter_efficient_pretraining",
    packages=["para_eff_pt"],
    install_requires=required,
)
