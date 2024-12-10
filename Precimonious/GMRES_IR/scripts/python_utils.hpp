#ifdef PY_SSIZE_T_CLEAN
template<typename T>
std::vector<T> convertPythonListToVector(std::vector<T>& vec,PyObject* pyList) {

    if (!PyList_Check(pyList)) return vec;

    Py_ssize_t size = PyList_Size(pyList);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(pyList, i);
        vec.push_back(static_cast<T>(PyFloat_AsDouble(item)));
    }

    return vec;
}


template<typename T, typename matrix_t>
int construct_no_py(int n, float cond, matrix_t& A) {

    

}

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p, bool is_symmetric, bool is_diag_dom, float& true_cond) {
    //this is an ambitious function that uses a Python embedding to call the functions found in generate\ copy.py to fill in the entries of A
    
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;
    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
    pName = PyUnicode_DecodeFSDefault((char*)"gen");
    pModule = PyImport_Import(pName); 
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, (char *)"LU_gen");

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(7);
            for (i = 0; i < 7; ++i) {
                switch(i) {
                    case 0:
                        pValue = PyLong_FromLong(n);
                        break;
                    case 1:
                        pValue = PyFloat_FromDouble(cond);
                        break;
                    case 2:
                        pValue = PyLong_FromLong(space);
                        break;
                    case 3:
                        pValue = geom ? Py_True : Py_False;
                        break;
                    case 4:
                        pValue = PyLong_FromLong(p);
                        break;
                    case 6:
                        pValue = is_diag_dom ? Py_True : Py_False;
                        break;
                    default :
                        pValue = is_symmetric ? Py_True : Py_False;
                        break;
                }
                
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                std::vector<T> b(n*n);
                std::vector<T> c(n*n + 1);
                for(int i = 0 ; i < n; i++) {
                    for(int j = 0; j < n; j++){
                        A(i,j) = static_cast<float>(static_cast<T>(PyFloat_AsDouble(PyList_GetItem(pValue, n*i + j))));
                    }
                }

                
                // tlapack::LegacyMatrix<T, int> LU(n, n, b.data(), n);
                // printMatrix(LU);
                true_cond = PyFloat_AsDouble(PyList_GetItem(pValue, n*n));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function for gen\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load program\n");
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
    } 

#else
 std::vector<float> getFileNames(const std::string& directory, float& true_cond) {
        std::vector<float> fileNames;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string fileName = entry.path().filename().string();
                fileName = fileName.substr(7, fileName.size() - 11); 
                true_cond = std::stof(fileName);
                fileNames.push_back(true_cond);
            }
        }
        return fileNames;
    }

float find_closest(const std::vector<float>& fileNames, float true_cond) {
    float min_diff = std::abs(fileNames[0] - true_cond);
    float closest = fileNames[0];
    for (int i = 1; i < fileNames.size(); i++) {
        float diff = std::abs(fileNames[i] - true_cond);
        if (diff < min_diff) {
            min_diff = diff;
            closest = fileNames[i];
        }
    }
    return closest;
}

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p, float& true_cond) {
   

    std::vector<float> fileNames = getFileNames("/root/home/Precimonious/GMRES_IR/tempscripts/matrix_collection_" + std::to_string(p), true_cond);
    true_cond = find_closest(fileNames, cond);
    std::string fileName = "/root/home/Precimonious/GMRES_IR/tempscripts/matrix_collection_" + std::to_string(p) + "/" + "matrix_" + std::to_string(static_cast<int>(true_cond)) + ".csv";
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "File not found" << std::endl;
        return 1;
    }
    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int j = 0;
        while (std::getline(ss, token, ',')) {
            if(j == n && i == n) break;
            else {
            A(i, j) = std::stod(token);
            j++;
            if(j == n) {i++; j = 0;}
            }
        }
    }
    std::cout << "true_cond is : " <<  true_cond << std::endl;
    return 0;

}

#endif
