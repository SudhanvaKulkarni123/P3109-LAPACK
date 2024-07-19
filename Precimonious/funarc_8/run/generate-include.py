import json


include = {}
include["funarc_8.cpp"] = {}
include["funarc_8.cpp"]["function"] = {}
# add variables to exclude for each function to include
include["funarc_8.cpp"]["function"]["main"] = ["epsilon", "zeta_verify_value", "err", "cpu_time_used", "start_time", "end_time"]
include["funarc_8.cpp"]["function"]["fun"] = [""]



with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")

