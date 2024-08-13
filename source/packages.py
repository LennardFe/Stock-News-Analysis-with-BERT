import subprocess

def install_packages(file, already_checked):
    if already_checked: # If already checked in a prior run, skip the check
        return
    
    try:
        subprocess.check_call(["pip", "install", "-r", file])
        print("Packages installed successfully!")
    except subprocess.CalledProcessError:
        print("An error occurred while installing packages.")


