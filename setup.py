from setuptools import setup, find_packages

setup(
    name="comfyui-hollow-preserve",
    version="0.1.0",
    description="ComfyUI node to remove enclosed areas from masks for better inpainting",
    author="MAHIL K R",
    author_email="mahilkr246810@gmail.com",
    url="https://github.com/krmahil/comfyui-hollow-preserve",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
