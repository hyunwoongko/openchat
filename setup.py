from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='openchat',
    version='1.1.12',
    description='Opensource chatting framework for generative models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hyunwoong Ko',
    author_email='gusdnd852@naver.com',
    url='https://github.com/hyunwoongko/dialogpt-chat',
    install_requires=[
        'transformers>=4<4.3.0',
        'torch',
        "parlai",
    ],
    packages=find_packages(),
    python_requires='>=3',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
)
