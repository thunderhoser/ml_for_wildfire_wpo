"""Setup file for ml_for_wildfire_wpo."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml_for_wildfire_wpo', 'ml_for_wildfire_wpo.io',
    'ml_for_wildfire_wpo.utils', 'ml_for_wildfire_wpo.scripts',
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data mining', 'weather', 'meteorology', 'wildfire', 'fire weather',
    'fire behaviour', 'forest fire'
]
SHORT_DESCRIPTION = (
    'Machine learning for predicting extreme fire weather and behaviour.'
)
LONG_DESCRIPTION = (
    'ml_for_wildfire_wpo is an end-to-end machine-learning library for '
    'predicting extreme fire weather and behaviour over the United States at '
    'multi-day lead times.'
)
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml_for_wildfire_wpo',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/ml_for_wildfire_wpo',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
