from setuptools import setup, find_packages

setup(
    name='soccer_player_reidentification',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.1',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.11.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.2',
        'scikit-learn>=1.3.0',
        'Pillow>=10.0.0',
        'pytest>=7.4.0',
        'tqdm>=4.66.3',
        'colorama>=0.4.6',
    ],
    extras_require={
        'dev': [
            'pytest-cov>=4.1.0',
            'flake8>=6.1.0',
            'black>=23.7.0',
            'isort>=5.12.0',
            'mypy>=1.5.1',
        ],
    },
    python_requires='>=3.8',
    author='Miloni Panchal',
    author_email='miloni.panchal@example.com',
    description='Real-time player tracking and re-identification system for sports analytics using advanced computer vision techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='computer-vision, sports-analytics, player-tracking, re-identification',
    url='https://github.com/miloni0731/player-reidentification',
    project_urls={
        'Documentation': 'https://github.com/miloni0731/player-reidentification/docs',
        'Source': 'https://github.com/miloni0731/player-reidentification',
        'Issues': 'https://github.com/miloni0731/player-reidentification/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    entry_points={
        'console_scripts': [
            'soccer-reid=src.main:main',
        ],
    },
)