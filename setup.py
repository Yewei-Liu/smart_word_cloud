from setuptools import setup

setup(
    name="smartwordcloudgenerator",
    version="0.6",
    description="Generate smart word cloud.",
    author="刘晔玮  Yewei Liu (Lewis)",
    author_email="liuyeweilewis@gmail.com",
    url="https://github.com/Yewei-Liu/smart_word_cloud.git",
    license="MIT",
    install_requires=[
        'nltk',
        'Pillow',
    ]
)