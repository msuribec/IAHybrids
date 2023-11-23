import os
import subprocess
import venv


def create_folders_if_not_exist(folder_paths):
    """Function that creates folders if they don't exist
    Parameters
    ----------
    folder_paths : list
        List with the paths of the folders to create
    """
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class CreateVenv:
    """Class that represents a virtual environment it can create and activate the environment
    as well as install required packages and run scripts inside the environment
    Parameters
    ----------
    venv_folder : str, optional
        Name of the folder where the virtual environment will be created, by default 'venv'
    
    """
    def __init__(self, venv_folder = 'venv'):
        self.venv_folder = venv_folder
        self.venv_dir = os.path.join(os.getcwd(), self.venv_folder)
        print(self.venv_dir)
        self.activate_dir = os.path.join(self.venv_dir, "Scripts", "activate.bat")
        self.pip_exe = os.path.join(self.venv_dir, "Scripts", "pip3.exe")
        self.python_exe = os.path.join(self.venv_dir, "Scripts", "python.exe")
        print(self.python_exe)
        self.create_venv()
        self.activate_venv()

    def create_venv(self):
        """Creates the virtual environment"""
        venv.create(self.venv_dir, with_pip=True)
    
    def activate_venv(self):
        """Activates the virtual environment"""
        subprocess.run([self.activate_dir])

    def install_packages(self,package_list):
        """Installs the given packages in the virtual environment
        Parameters
        ----------
        package_list : list
            List of packages to install
            """
        for package in package_list:
            subprocess.run([self.pip_exe, "install", package])
    
    def install_requirements(self, requirements_file):
        """Installs the packages in the given requirements file
        Parameters
        ----------
        requirements_file : str
            Path of the requirements file
        """
        subprocess.run([self.pip_exe, "install", "-r", requirements_file])
    
    def run_file(self, file_path, args = None):
        """Runs the given file inside the virtual environment
        Parameters
        ----------
        file_path : str
            Path of the file to run
        args : list, optional
            string of arguments to pass to the file, by default None
        """
        if args is None:
            subprocess.run([self.python_exe,  file_path])
        else:
            print(args)
            subprocess.run([self.python_exe,  file_path, *args])
