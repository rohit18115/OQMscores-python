from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='oqmscore',
      version='0.1',
      description='Objective quality measure score for speech',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/rohit18115/OQMscores-python",
      packages=['oqmscore'],
      install_requires=[
      'pesq',
      'numpy',
      'librosa',
      'scipy'
      ],
      author='Rohit Arora',
      author_email='rohit18115@iiitd.ac.in',
      zip_safe=False)
