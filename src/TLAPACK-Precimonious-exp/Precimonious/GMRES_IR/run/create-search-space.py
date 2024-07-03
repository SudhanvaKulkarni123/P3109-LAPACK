import json
import os
import json
import os


with open('include.json') as json_file:
    include = json.load(json_file)

file = 'GMRES_IR.cpp'
PYTHON_LIBS = "-lpython3.8"
PYTHON_INCLUDE_DIRS = "-I/usr/include/python3.8"
PY_CFLAGS = "`python3.8-config --cflags`"
PY_LDFLAGS = "`python3.8-config --ldflags`"
INCLUDE_PATH = "-I/root/home/Precimonious/tlapack/include/"
OTHER_INCLUDE_PATH = "-iquote/root/home/Precimonious/tlapack/include/"
os.system(f"clang++-18 -std=c++17 {INCLUDE_PATH} {OTHER_INCLUDE_PATH} -shared -lpython3.8 -w {PYTHON_LIBS} {PYTHON_INCLUDE_DIRS} {PY_CFLAGS} {PY_LDFLAGS} -Xclang -load -Xclang ../../plugin/CreateSearchSpace.so -Xclang -plugin -Xclang create-space -Xclang -plugin-arg-create-space -Xclang -output-path -Xclang -plugin-arg-create-space -Xclang ./ -Xclang -plugin-arg-create-space -Xclang -output-name -Xclang -plugin-arg-create-space -Xclang config.json -Xclang -plugin-arg-create-space -Xclang -input-file -Xclang -plugin-arg-create-space -Xclang {file} ../scripts/{file}")
    #     (.text+0x24): undefined reference to `main'
    #     clang-12: error: linker command failed with exit code 1 (use -v to see invocation)
    #A : Add -c flag to the command to avoid linking    

