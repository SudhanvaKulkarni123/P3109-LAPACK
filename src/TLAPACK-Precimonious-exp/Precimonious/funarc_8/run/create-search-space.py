import json
import os
import json
import os


with open('include.json') as json_file:
    include = json.load(json_file)

for file in include.keys():
    os.system(f"clang++-17 -Xclang -load -Xclang ../../plugin/NewCreateSearchSpace.so -Xclang -plugin -Xclang create-space -Xclang -plugin-arg-create-space -Xclang -output-path -Xclang -plugin-arg-create-space -Xclang ./ -Xclang -plugin-arg-create-space -Xclang -output-name -Xclang -plugin-arg-create-space -Xclang config.json -Xclang -plugin-arg-create-space -Xclang -input-file -Xclang -plugin-arg-create-space -Xclang {file} ../scripts/{file} -std=c++17 -shared")
    #     (.text+0x24): undefined reference to `main'
    #     clang-12: error: linker command failed with exit code 1 (use -v to see invocation)
    #A : Add -c flag to the command to avoid linking    

