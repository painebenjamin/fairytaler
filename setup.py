from setuptools import find_packages, setup

setup(
    name="fairytaler",
    version="0.1.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="An unofficial reimplementation of F5TTS",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"fairytaler": ["py.typed"]},
    license="cc-by-nc-4.0",
    url="https://github.com/painebenjamin/fairytaler",
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=open("requirements.txt", "r").read().strip().split("\n"),
    entry_points={
        "console_scripts": [
            "fairytaler = fairytaler.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)
