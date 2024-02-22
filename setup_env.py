import os
import subprocess
import sys

# Define the directory for the virtual environment
venv_dir = ".venv"

# Function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

# Check if the virtual environment directory exists
if not os.path.isdir(venv_dir):
    print("Virtual environment not found. Creating one...")
    run_command(f"python -m venv {venv_dir}")

# Activate the virtual environment
# Note: Direct activation in script is complex; instead, use subprocess to ensure commands use the virtual environment
if sys.platform == "win32":
    activate_script = os.path.join(venv_dir, "Scripts", "activate")
else:
    activate_script = f"source {venv_dir}/bin/activate"

# Check if requirements.txt exists and install requirements
requirements_path = "requirements.txt"
if os.path.isfile(requirements_path):
    print("Installing requirements from requirements.txt...")
    run_command(f"{sys.executable} -m pip install -r {requirements_path}")
else:
    print("requirements.txt not found. Skipping requirements installation.")

print("Setup completed.")
