from setuptools import setup, find_packages, Command  # Import Command from setuptools
import pathlib
import subprocess

# Current directory
here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


def format_git_describe_version(version):
    if '-' in version:
        splitted = version.split('-')
        tag = splitted[0]
        index = f"dev{splitted[1]}"
        return f"{tag}.{index}"
    else:
        return version


def get_version_from_git():
    try:
        process = subprocess.run(
            ["git", "describe"],
            cwd=str(here),
            check=True,
            capture_output=True
        )
        version = process.stdout.decode('utf-8').strip()
        version = format_git_describe_version(version)
        with version_file.open('w') as f:
            f.write(version)
        return version
    except subprocess.CalledProcessError:
        if version_file.exists():
            return version_file.read_text().strip()
        else:
            return '0.1.0'


version = get_version_from_git()

print(f"Detected version {version} from git describe")


class GetVersionCommand(Command):
    """A custom command to get the current project version inferred from git describe."""

    description = 'gets the project version from git describe'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(version)


setup(
    name='AutoML Fairness Experiments',  # Required
    version=version,
    description='Experiments for the paper XXX',  # Optional
    license='Apache 2.0 License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MatteoMagnini/autoML-FAI-experiments',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Patients and Health Professionals',
        'Topic :: AutoML :: Fairness',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.12'
    ],
    keywords='AutoML, Fairness, Artificial Intelligence, Machine Learning',  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.12.0',
    zip_safe=False,
    platforms="Independent",
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/MatteoMagnini/autoML-FAI-experiments/issues',
        'Source': 'https://github.com/MatteoMagnini/autoML-FAI-experiments',
    },
    cmdclass={
        'get_project_version': GetVersionCommand
    },
)
