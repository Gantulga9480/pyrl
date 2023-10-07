import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyRL",
    version="1.0",
    author="Gantulga G",
    author_email="limited.tulgaa@gmail.com",
    description="Pytorch implementation of RL algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gantulga9480/PyRL",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=['numpy'],
)
