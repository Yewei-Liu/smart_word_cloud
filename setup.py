from setuptools import setup, find_packages

setup(
    name="smartwordcloudgenerator",
    version="1.4",
    packages=find_packages(),
    description="Generate smart word cloud.",
    author="刘晔玮  Yewei Liu (Lewis)",
    author_email="liuyeweilewis@gmail.com",
    url="https://github.com/Yewei-Liu/smart_word_cloud.git",
    license="MIT",
    include_package_data=True,
    install_requires=[
        'nltk',
        'Pillow',
    ],
    package_data={
        'smartwordcloudgenerator': ['data/*', 'data/**/*', 'DroidSansMono.ttf'],  
    },
)