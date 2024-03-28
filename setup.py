from setuptools import setup, find_packages

setup(
    name='FAIM',
    version='0.1',
    description='Fairness-Aware Integral Regression',
    author='Mingxuan Liu',
    author_email='mingxuan.liu@u.duke.nus.edu',
    url='https://github.com/Michellemingxuan/FAIReg',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.0',
        'pandas>=2.0.1',
        'matplotlib>=3.7.2',
        'plotnine==0.12.1',
        'seaborn>=0.12.2',
        'aif360>=0.5.0',
        'fairlearn>=0.8.0',
        'scikit-learn>=1.2.2',
        'statsmodels>=0.14.0',
        'tqdm>=4.66.1',
        'plotly>=5.14.1',
        'nbformat>=4.2.0',
        'mizani==0.9.0',
        "patchworklib>=0.6.2",
        "shap>=0.43.0",
        "kaleido>=0.2.1",
        # "ShapleyVIC>=1.0.0",
        "ShapleyVIC@git+https://github.com/nliulab/ShapleyVIC#egg=ShapleyVIC&subdirectory=python"
    ],
    python_requires='>=3.6',
)
