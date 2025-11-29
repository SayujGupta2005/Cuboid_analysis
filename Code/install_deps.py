import subprocess
import sys

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return f"Successfully installed {package}"
    except subprocess.CalledProcessError as e:
        return f"Failed to install {package}: {e}"

with open('install_result.txt', 'w') as f:
    f.write(install('rosbags') + '\n')
    f.write(install('numpy') + '\n')
    f.write(install('scipy') + '\n')
