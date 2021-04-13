import setuptools

# using requirements.in when installing as a library,
# because requirements.txt is the fixed version which
# is unsuitable for library use
with open('requirements.in') as f:
    install_requires = f.readlines()

setuptools.setup(
    name="blindml",  # Replace with your own username
    version="0.0.1",
    author="The Data Station",
    author_email="author@example.com",
    description="blindml (short) description",
    install_requires=install_requires,
    long_description="see README",
    long_description_content_type="text/markdown",
    url="https://github.com/TheDataStation/blindml",
    project_urls={
        "Bug Tracker": "https://github.com/TheDataStation/blindml",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
)
