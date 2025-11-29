try:
    import rosbags
    import numpy
    import scipy
    with open('test_output.txt', 'w') as f:
        f.write('Imports successful')
except Exception as e:
    with open('test_output.txt', 'w') as f:
        f.write(f'Import failed: {e}')
