from setuptools import setup, find_packages


# version 파일에서 버전 가져옴

def get_version(path='version'):
    version = ""
    with open(path, 'r') as f:
        version = f.read().strip()

    return version


# requirements 파일에서 requirements 정보 가져옴
def get_requirements(path='requirements.txt'):
    reqs = []

    try:
        with open(path, 'r') as f:
            reqs = f.read().split()

        return reqs
    except Exception:
        return []


name = 'ml-model-realization'
option = 'source'

if option == 'all':
    setup(
        name=name,
        version=get_version(path='version'),
        description='ml-model-realization',
        author='dhk',
        author_email='ehdgns322@gmail.com',
        urllib='',
        install_requires=get_requirements(path='requirements.txt'),
        packages=find_packages(exclude=['venv']),
        keywords=['bert', 'deep-learning'],
        python_requires='>=3.6'
    )
elif option == 'source':
    setup(
        name=name,
        version=get_version(path='version'),
        description='ml-model-realization',
        author='dhk',
        author_email='ehdgns322@gmail.com',
        urllib='',
        # install_requires=get_requirements(path='requirements.txt'),
        packages=find_packages(exclude=['venv']),
        keywords=['bert', 'deep-learning'],
        python_requires='>=3.6'
    )
