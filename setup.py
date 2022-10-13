from setuptools import setup

setup(
    name="mniscnn",
    version="0.1",
    description="Microstructural neuroimaging using spherical CNNs",
    url="https://github.com/kerkelae/mniscnn",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["mniscnn"],
    install_requires=[
        "dipy",
        "healpy",
        "matplotlib",
        "nibabel",
        "numpy",
        "scipy",
        "seaborn",
        "torch",
    ],
)
