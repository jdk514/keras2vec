import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Keras2Vec",
    version="0.0.1",
    author="Joel Klein",
    author_email="jdk51405@gmail.com",
    description="Keras implementation of Doc2Vec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdk514/keras2vec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow >= 1.14.0',
        'keras >= 2.2.4',
        'numpy >= 1.16.4',
    ],
    python_requires='>=3.6',
)