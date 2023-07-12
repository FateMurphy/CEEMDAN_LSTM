#setup.py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='CEEMDAN_LSTM',
    version='1.2b3',
    packages=setuptools.find_packages(),
    install_requires=['numpy >= 1.17.3',
                      'pandas >= 1.2.0',
                      'EMD-signal >= 1.2.3',
                      'optuna >= 3.0.0',
                      'vmdpy',
                      'sampen',
                      'matplotlib',
                      'sklearn',
                      'tensorflow >= 2.5.0',
                      ],
    package_data={'CEEMDAN_LSTM': ['datasets/*']},
    description='CEEMDAN_LSTM is a Python project for decomposition-integration forecasting models based on EMD methods and LSTM.',
    url='http://github.com/FateMurphy/CEEMDAN_LSTM',

    author='Feite Zhou',
    author_email='jupiterzhou@foxmail.com',
    keywords = ['CEEMDAN', 'VMD', 'LSTM', 'decomposition', 'forecasting'],
    long_description=long_description,
    long_description_content_type="text/markdown",  
    classifiers=[
        'Development Status :: 4 - Beta',     
        'Intended Audience :: Science/Research',     
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',]
)