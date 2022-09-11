import setuptools


setuptools.setup(
    name="char-diffusion",
    version="0.0.1",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    author="Jon Tow",
    author_email="jonathantow1@gmail.com",
    url="http://github.com/jon-tow/char-diffusion",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        "jax",
        "jaxtyping",
        "einops>=0.4",
        "equinox",
        "numpy",
        "optax",
        "wandb",
        "ml_collections",
    ],
    extras_require={
        "test": ["pytest"],
    },
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp machinelearning",
)
