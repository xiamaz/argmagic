import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="argmagic",
    version="0.0.1",
    author="Max Zhao",
    author_email="alcasa.mz@gmail.com",
    description="Parse environment variables and CLI arguments for a given function signature.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiamaz/argmagic",
    py_modules=["argmagic"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
