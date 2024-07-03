import json
import os
import sys


CONFIG=sys.argv[1]

def is_BLAS(file):
    directory = '../../tlapack/include/tlapack/blas'
    filename = os.path.basename(file)
    if filename in os.listdir(directory):
        return True
    else:
        return False

def is_LAPACK(file):
    directory = '../../tlapack/include/tlapack/lapack'
    filename = os.path.basename(file)
    if filename in os.listdir(directory):
        return True
    else:
        return False

with open('include.json') as json_file:
    include = json.load(json_file)



PYTHON_LIBS = "-lpython3.8"
PYTHON_INCLUDE_DIRS = "-I/usr/include/python3.8"
PY_CFLAGS = "`python3.8-config --cflags`"
PY_LDFLAGS = "`python3.8-config --ldflags`"
INCLUDE_PATH = "-I/root/home/Precimonious/tlapack/include/"
OTHER_INCLUDE_PATH = "-iquote/root/home/Precimonious/tlapack/include/"
for file in include.keys():
    if is_BLAS(file) :
        file = '../../tlapack/include/tlapack/blas/' + file
        os.system(f"clang++-18 -std=c++20 {INCLUDE_PATH} {OTHER_INCLUDE_PATH} -shared  -lpython3.8 -w {PYTHON_LIBS} {PYTHON_INCLUDE_DIRS} {PY_CFLAGS} {PY_LDFLAGS} -Xclang -load -Xclang ../../plugin/TransformType.so -Xclang -plugin -Xclang trans-type -Xclang -plugin-arg-trans-type -Xclang -output-path -Xclang -plugin-arg-trans-type -Xclang ../tempscripts/ -Xclang -plugin-arg-trans-type -Xclang -input-config -Xclang -plugin-arg-trans-type -Xclang {CONFIG} {file}")
    elif is_LAPACK(file):
        file = '../../tlapack/include/tlapack/lapack/' + file
        os.system(f"clang++-18 -std=c++20 {INCLUDE_PATH} {OTHER_INCLUDE_PATH} -shared  -lpython3.8 -w {PYTHON_LIBS} {PYTHON_INCLUDE_DIRS} {PY_CFLAGS} {PY_LDFLAGS} -Xclang -load -Xclang ../../plugin/TransformType.so -Xclang -plugin -Xclang trans-type -Xclang -plugin-arg-trans-type -Xclang -output-path -Xclang -plugin-arg-trans-type -Xclang ../tempscripts/ -Xclang -plugin-arg-trans-type -Xclang -input-config -Xclang -plugin-arg-trans-type -Xclang {CONFIG} {file}")
    else :
        os.system(f"clang++-18 -std=c++20 {INCLUDE_PATH} {OTHER_INCLUDE_PATH} -shared  -lpython3.8 -w {PYTHON_LIBS} {PYTHON_INCLUDE_DIRS} {PY_CFLAGS} {PY_LDFLAGS} -Xclang -load -Xclang ../../plugin/TransformType.so -Xclang -plugin -Xclang trans-type -Xclang -plugin-arg-trans-type -Xclang -output-path -Xclang -plugin-arg-trans-type -Xclang ../tempscripts/ -Xclang -plugin-arg-trans-type -Xclang -input-config -Xclang -plugin-arg-trans-type -Xclang {CONFIG} ../scripts/{file}")
