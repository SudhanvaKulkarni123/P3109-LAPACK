import json
import os


include = {}
include["GMRES_IR.cpp"] = {}
include["GMRES_IR.cpp"]["function"] = {}
# add variables to exclude for each function to include
include["GMRES_IR.cpp"]["function"]["main"] = ["er3", "cpu_time_used", "start_time", "end_time"]
include["GMRES_IR.cpp"]["function"]["run"] = [""]


# Parse GMRES_IR.cpp and add #include files
GMRES_IR_path = "../scripts/"
GMRES_IR_file = "GMRES_IR.cpp"

with open(os.path.join(GMRES_IR_path, GMRES_IR_file), 'r') as f:
    lines = f.readlines()

BLAS_path = "../../tlapack/include/tlapack/blas/"
LAPACK_path = "../../tlapack/include/tlapack/lapack/"
for line in lines:
    if line.startswith("#include"):
        parsed = line.split("#include")[1].strip().strip('<>').split('/')
        include_file = parsed[-1]
        funcname = include_file[:-4]
        if len(parsed) <= 1 or funcname == "lacpy":
            continue
        if parsed[-2] == "blas" or parsed[-2] == "lapack":
            include[include_file] = {}
            include[include_file]["function"] = {}
            include[include_file]["function"][funcname] = [""]

include["getrf_recursive.hpp"] = {}
include["getrf_recursive.hpp"]["function"] = {}
include["getrf_recursive.hpp"]["function"]["getrf_recursive"] = [""]
# Save the updated include dictionary to include.json
with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)

    print("include.json is generated.")
