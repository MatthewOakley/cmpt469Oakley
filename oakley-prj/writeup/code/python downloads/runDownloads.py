# runs the downloads for python
import pip

def install(package):
    pip.main(['install', package])

install('numpy-1.13.3+mkl-cp35-cp35m-win_amd64.whl')
install('scipy-1.0.0-cp35-cp35m-win_amd64.whl')
	
install('pandas')
install('pillow')
install('sklearn')
install('matplotlib')

print('all good')