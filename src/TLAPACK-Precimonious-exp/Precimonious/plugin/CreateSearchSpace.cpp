// create-space Clang plugin
//
#include <iostream>
#include <fstream>
#include <filesystem>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "../utilities/json.hpp"


using namespace clang;
using namespace std;
using namespace llvm;


#define endline "\n"
#define VERBOSE 1
#define PRINT_DEBUG_MESSAGE(s) if (VERBOSE > 0) {errs() << s << endline; }



#define FP32 "float"
#define FP64 "double"
#define FP80 "long double"
#define FPB4 "ml_dtypes::float8_ieee_p<4>"
#define FPB3 "ml_dtypes::float8_ieee_p<3>"
#define FP16 "Eigen::half"
#define BF16 "Eigen::bfloat16"



string funcName;
//std::string basefilename;
nlohmann::json event; 
nlohmann::json searchspace;   
nlohmann::json includespace;
int varCount = 0;
string GLOBAL = "globalVar";
string LOCAL = "localVar";
string TYPEDEC = "typeDec";
string PARM = "parmVar";
string CALL = "call";
string OUTPUT_PATH = "";
string OUTPUT_NAME = "config.json";
string INPUT_FILE = "";
string INCLUDE = "include.json";
std::vector<std::string> TYPES = {FPB4, BF16, FP16, FP32, FP64};

// for string delimiter
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
} 

// Used by std::find_if
struct MatchPathSeparator
{
    bool operator()(char ch) const {
        return ch == '/';
    }
};

string basename(std::string path) {
    return std::string( std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(), path.end());
}

string filter(std::string& TypNam) {
    if(TypNam.find("struct") != std::string::npos) {
        return TypNam.substr(7);
    } else if(TypNam.find("const") != std::string::npos) {
        return TypNam.substr(6);
    } else if(TypNam.find("class") != std::string::npos) {
        return TypNam.substr(6);
    } else {
        return TypNam;
    }
}

bool isBLAS(std::string& funcname) {
    std::string currentDir = std::filesystem::current_path().string();
   std::string path = "../../tlapack/include/tlapack/blas/";
    for (const auto & entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().filename() == (funcname + ".hpp")) {
            return true;
        }
    }
    return false;
    
}

bool isLAPACK(std::string& funcname) {
    std::string currentDir = std::filesystem::current_path().string();
    std::string path = "../../tlapack/include/tlapack/lapack/";
    for (const auto & entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().filename() == (funcname + ".hpp")) {
            return true;
        }
    }
    return false;
    

}

struct FloatingPointTypeInfo {
    bool isFloatingPoint : 1;
    unsigned int isVector : 3;
    bool isPointer : 1;
    bool isArray : 1;
    const clang::Type* typeObj;
};

//Q: does isStructureType() include struct, union, and class?




FloatingPointTypeInfo DissectFloatingPointType(const clang::Type* typeObj, bool builtIn) {
    if (typeObj == NULL) {
        FloatingPointTypeInfo info;
        info.isFloatingPoint = false;
        info.isVector = 0;
        info.typeObj = NULL;
        info.isArray = false;
        info.isPointer = false;
        return info;
    }
    FloatingPointTypeInfo info;
    info.isArray = false;
    info.isPointer = false;
//     if (const clang::ArrayType* arr = dyn_cast<clang::ArrayType>(typeObj)) {
// //        PRINT_DEBUG_MESSAGE("\t\tis array type");
//         info = DissectFloatingPointType(arr->getElementType().getTypePtrOrNull(), false);
//         info.isArray = true;
//         return info;
//     }
//     else if (const clang::PointerType* ptr = dyn_cast<clang::PointerType>(typeObj)) {
// //        PRINT_DEBUG_MESSAGE("\t\tis pointer type");
//         info = DissectFloatingPointType(ptr->getPointeeType().getTypePtrOrNull(), false);
//         info.isPointer = true;
//         return info;
//     }
//     else if (const clang::PointerType* ptr = dyn_cast<clang::PointerType>(typeObj->getCanonicalTypeInternal())) {
// //        PRINT_DEBUG_MESSAGE("\t\tis pointer type");
//         info = DissectFloatingPointType(ptr->getPointeeType().getTypePtrOrNull(), false);
//         info.isPointer = false;
//         return info;
//     }

//    PRINT_DEBUG_MESSAGE("\t\tinnermost type " << typeObj->getCanonicalTypeInternal().getAsString());

    if (const clang::BuiltinType* bltin = dyn_cast<clang::BuiltinType>(typeObj)) {
//        PRINT_DEBUG_MESSAGE("\t\tis builtin type, floating point: " << bltin->isFloatingPoint());
        info.isFloatingPoint = bltin->isFloatingPoint();
        info.isVector = 0;
        info.typeObj = typeObj;
        return info;
    }
    else if (typeObj->isStructureType() || typeObj->isClassType()){
//        PRINT_DEBUG_MESSAGE("\t\tis struct type");
        // TODO: with floating point built-in vectors and __half
        info.isFloatingPoint = false;
        info.isVector = 0;
        std::string typeStr = typeObj->getCanonicalTypeInternal().getAsString();
        if (typeStr.find("half") != std::string::npos) {
            info.isFloatingPoint = true;
            info.isVector = 0;
        } else if (typeStr.find("float8_ieee_p") != std::string::npos) {
            info.isFloatingPoint = true;
            info.isVector = 0;
        } else if (typeStr.find("bfloat") != std::string::npos) {
            info.isFloatingPoint = true;
            info.isVector = 0;
        } else {
            size_t pos = typeStr.find("struct float");
            if (pos != std::string::npos) {
                info.isFloatingPoint = true;
                info.isVector = typeStr[pos + strlen("struct float")] - '1';
            }
            pos = typeStr.find("struct double");
            if (pos != std::string::npos) {
                info.isFloatingPoint = true;
                info.isVector = typeStr[pos + strlen("struct double")] - '1';
            }
        }
        info.typeObj = typeObj;
        return info;
    }
    else {
//        PRINT_DEBUG_MESSAGE("\t\tis another type");
        info.isFloatingPoint = false;
        info.isVector = 0;
        info.typeObj = typeObj;
        info.isArray = false;
        info.isPointer = false;
        return info;
    }
}


namespace {
class TraverseGlobalVisitor : public clang::RecursiveASTVisitor<TraverseGlobalVisitor> {
public:
  explicit TraverseGlobalVisitor(ASTContext *Context) : Context(Context) {};
  bool VisitVarDecl(clang::VarDecl *val) {
    return true;
  }
private:
  clang::ASTContext *Context;
};

class TraverseVarsVisitor : public clang::RecursiveASTVisitor<TraverseVarsVisitor> {
public:
  explicit TraverseVarsVisitor(ASTContext *Context) : Context(Context) {};

  bool VisitVarDecl(clang::VarDecl *val) {
            

        return true;
  }
private:
  clang::ASTContext *Context;
};

class FuncStmtVisitor : public RecursiveASTVisitor<FuncStmtVisitor> {
public:
    explicit FuncStmtVisitor(ASTContext *Context) : Context(Context) {};

    bool VisitStmt(clang::Stmt *st) {
       
        if(DeclStmt *decSt = dyn_cast<DeclStmt>(st)) {
             SourceManager& SrcMgr = Context->getSourceManager();
             for (auto decl : decSt->decls()) {
                const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(decl->getLocation()));
         string FileName = basename((Entry->tryGetRealPathName()).data());

                if (TypeAliasDecl *typDec = dyn_cast<TypeAliasDecl>(decl)) {
                    if(typDec->getNameAsString() == "accum_type" || typDec->getNameAsString() == "gemm_type" || typDec->getNameAsString().find("trsm_type") != std::string::npos ){
                    string typeName = typDec->getUnderlyingType().getAsString();
                    typeName = filter(typeName);
                    string valueName = typDec->getNameAsString();
                    if(isBLAS(FileName)) FileName = "../../tlapack/include/tlapack/blas/" + FileName; 
                    else if(isLAPACK(FileName)) FileName = "../../tlapack/include/tlapack/lapack/" + FileName;
                    else FileName = FileName;

                if (std::find(includespace[FileName]["function"][funcName].begin(), includespace[FileName]["function"][funcName].end(), valueName) == includespace[FileName]["function"][funcName].end())
                {
                    int len = typeName.length();
                    // get line number
                    string linenum = "";
                    auto loc = typDec->getBeginLoc();
                    string source_loc = loc.printToString(Context->getSourceManager());
                    string delimiter = ":";
                    vector<string> v = split (source_loc, delimiter);
                    int count = 0;
                    for (auto i : v) {
                        count = count + 1;
                        if (count == 2) {
                            linenum = i;
                        }
                    } 

                    varCount ++;
                    event[LOCAL+to_string(varCount)]["function"] = funcName;
                    event[LOCAL+to_string(varCount)]["name"] = valueName;
                    event[LOCAL+to_string(varCount)]["type"] = typeName;
                    event[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    event[LOCAL+to_string(varCount)]["file"] = FileName;
                    event[LOCAL+to_string(varCount)]["lines"] = {linenum};

                    string next_type = typeName;
                    if (next_type.substr(0, 5) == "const") {
                        next_type.replace(6, len, "float");
                    } else {
                        next_type.replace(0, len, "float");
                    }
                    searchspace[LOCAL+to_string(varCount)]["function"] = funcName;
                    searchspace[LOCAL+to_string(varCount)]["name"] = valueName;
                     int typeIndex = -1;
                        for (int i = 0; i < TYPES.size(); i++) {
                            if (TYPES[i] == typeName) {
                                typeIndex = i;
                                break;
                            }
                        }
                        // Set searchspace[GLOBAL+to_string(varCount)]["type"] to be precisions from FP32 to typename
                        if (typeIndex != -1) {
                            std::vector<std::string> precisions(TYPES.begin(), TYPES.begin() + typeIndex + 1);
                            searchspace[LOCAL+to_string(varCount)]["type"] = TYPES;
                        } else {
                            // Handle the case when typename is not found in TYPES
                            // Set searchspace[GLOBAL+to_string(varCount)]["type"] to be {next_type, typeName}
                            searchspace[LOCAL+to_string(varCount)]["type"] = TYPES;
                        }

                    searchspace[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    searchspace[LOCAL+to_string(varCount)]["file"] = FileName;
                    searchspace[LOCAL+to_string(varCount)]["lines"] = {linenum};

                }


                    }
                }
            }

            
              
            

            
        return true;
            
        } else if(CallExpr* funcCall = dyn_cast<CallExpr>(st)) {
            if(funcName == "main"){
                SourceManager& SrcMgr = Context->getSourceManager();
                const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(funcCall->getBeginLoc()));
                string FileName = basename((Entry->tryGetRealPathName()).data());

            auto origFunc = funcCall->getDirectCallee();
            if(origFunc){
            int num_args = funcCall->getNumArgs();
                if(origFunc) {

                    if(const FunctionTemplateDecl *funcTemplate = origFunc->getPrimaryTemplate()) {
                        const TemplateArgumentList *templateArgs = origFunc->getTemplateSpecializationArgs();
                        if(templateArgs) {
                            int cnt = 0;
                            for( auto arg : templateArgs->asArray()) {
                                cnt++;
                                if(arg.getKind() == TemplateArgument::ArgKind::Type) {
                                    string valueName = funcName + "_template_arg_" + to_string(cnt);
                                    FloatingPointTypeInfo info = DissectFloatingPointType(arg.getAsType().getTypePtrOrNull(), false);
                                    if(info.isFloatingPoint){
                if (std::find(includespace[FileName]["function"][funcName].begin(), includespace[FileName]["function"][funcName].end(), valueName) == includespace[FileName]["function"][funcName].end())
                {
                    string typeName = arg.getAsType().getAsString();
                    typeName = filter(typeName);
                    int len = typeName.length();
                    
                    // get line number
                    string linenum = "";
                    auto loc = funcCall->getBeginLoc();
                    string source_loc = loc.printToString(Context->getSourceManager());
                    string delimiter = ":";
                    vector<string> v = split (source_loc, delimiter);
                    int count = 0;
                    for (auto i : v) {
                        count = count + 1;
                        if (count == 2) {
                            linenum = i;
                        }
                    } 

                    varCount ++;
                    event[LOCAL+to_string(varCount)]["function"] = funcName;
                    event[LOCAL+to_string(varCount)]["name"] = valueName;
                    event[LOCAL+to_string(varCount)]["type"] = typeName;
                    event[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    event[LOCAL+to_string(varCount)]["file"] = FileName;
                    event[LOCAL+to_string(varCount)]["lines"] = {linenum};

                    string next_type = typeName;
                    if (next_type.substr(0, 5) == "const") {
                        next_type.replace(6, len, "float");
                    } else {
                        next_type.replace(0, len, "float");
                    }
                    searchspace[LOCAL+to_string(varCount)]["function"] = funcName;
                    searchspace[LOCAL+to_string(varCount)]["name"] = valueName;
                     int typeIndex = -1;
                        for (int i = 0; i < TYPES.size(); i++) {
                            if (TYPES[i] == typeName) {
                                typeIndex = i;
                                break;
                            }
                        }
                        // Set searchspace[GLOBAL+to_string(varCount)]["type"] to be precisions from FP32 to typename
                        if (typeIndex != -1) {
                            std::vector<std::string> precisions(TYPES.begin(), TYPES.begin() + typeIndex + 1);
                            searchspace[LOCAL+to_string(varCount)]["type"] = TYPES;
                        } else {
                            // Handle the case when typename is not found in TYPES
                            // Set searchspace[GLOBAL+to_string(varCount)]["type"] to be {next_type, typeName}
                            searchspace[LOCAL+to_string(varCount)]["type"] = TYPES;
                        }

                    searchspace[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    searchspace[LOCAL+to_string(varCount)]["file"] = FileName;
                    searchspace[LOCAL+to_string(varCount)]["lines"] = {linenum};

                }


                                }
                            }
                        }
                    }
                    }
                
                

            
            }
            }
        }
        }
        return true;
    }
private:
    ASTContext *Context;
};





class TraverseFuncVisitor : public clang::RecursiveASTVisitor<TraverseFuncVisitor> {
public:
  explicit TraverseFuncVisitor(ASTContext *Context) : VarsVisitor(Context), st_visitor(Context), Context(Context) {};

  bool VisitFunctionDecl(FunctionDecl* func) {
    if (!(func->doesThisDeclarationHaveABody() && func->isDefined()))
        return true;
    funcName = func->getNameInfo().getName().getAsString();
    if((funcName == "main") || isBLAS(funcName) || (funcName == "GMRES_IR") || isLAPACK(funcName)) {
    
  
    if(funcName == "operator()") return true;

    // get file name
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(func->getLocation()));
    string FileName = basename((Entry->tryGetRealPathName()).data());

    // if this function is to consider or not
    if (includespace.contains(FileName)) {
        if (func->isTemplateInstantiation()) {
        const TemplateArgumentList* templist = func->getTemplateSpecializationArgs();
        for(auto arg : templist->asArray()) {
            if(arg.getKind() == TemplateArgument::ArgKind::Type) {
            }
        }
        }
        if (includespace[FileName]["function"].contains(funcName)) {
            VarsVisitor.TraverseStmt(func->getBody());
            st_visitor.TraverseStmt(func->getBody());
        }
    }
    }

    return true;
  }

  bool VisitFunctionDecl(CXXMethodDecl* func) {
    return true;
  }

private:
  TraverseVarsVisitor VarsVisitor;
  FuncStmtVisitor st_visitor;
  clang::ASTContext *Context;
};

class TraverseFuncVarsConsumer : public clang::ASTConsumer {
public:
  explicit TraverseFuncVarsConsumer(clang::ASTContext *Context, clang::DiagnosticsEngine *Diagnostics)
    : FuncVisitor(Context), GlobVisitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    std::ifstream ifs(INCLUDE);
    includespace = nlohmann::json::parse(ifs);
    FuncVisitor.TraverseDecl(Context.getTranslationUnitDecl());
    // Create an output file to write the updated code
    string filename = OUTPUT_PATH + OUTPUT_NAME;
    std::error_code OutErrorInfo;
    std::error_code ok;
    llvm::raw_fd_ostream outFile(llvm::StringRef(filename),
                OutErrorInfo, llvm::sys::fs::OF_None);
    if (OutErrorInfo == ok) {
                outFile << std::string(event.dump(4));
                PRINT_DEBUG_MESSAGE("Output file created - " << filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << filename);
        }

    string sp_filename = OUTPUT_PATH + "search_" + OUTPUT_NAME;
    std::error_code sp_OutErrorInfo;
    std::error_code sp_ok;
    llvm::raw_fd_ostream sp_outFile(llvm::StringRef(sp_filename),
                sp_OutErrorInfo, llvm::sys::fs::OF_None);
    if (sp_OutErrorInfo == sp_ok) {
                sp_outFile << std::string(searchspace.dump(4));
                PRINT_DEBUG_MESSAGE("Output file created - " << sp_filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << sp_filename);
        }


  }
private:
  TraverseFuncVisitor FuncVisitor;
  TraverseGlobalVisitor GlobVisitor;
};

class TraverseFuncVarsAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) {
    return std::make_unique<TraverseFuncVarsConsumer>(&CI.getASTContext(), &CI.getDiagnostics());
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    // To be written...

    
    llvm::errs() << "Plugin arg size = " << args.size() << "\n";
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (args[i] == "-output-path") {
        if (i+1 < e)
            OUTPUT_PATH = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing output path! Could not generate output file.");
        llvm::errs() << "Output path = " << OUTPUT_PATH << "\n";
      }
      if (args[i] == "-output-name") {
        if (i+1 < e)
            OUTPUT_NAME = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing output name! Could not generate output file.");
        llvm::errs() << "Output file name = " << OUTPUT_NAME << "\n";
      }
      if (args[i] == "-input-file") {
        if (i+1 < e)
            INPUT_FILE = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing input file name! Could not generate output file.");
        llvm::errs() << "Input file name = " << INPUT_FILE << "\n";
      }
    }

    return true;
  }
};

}

// Q : what is the sequence of function calls if I invoke the plugin with appropraite arguments?

static clang::FrontendPluginRegistry::Add<TraverseFuncVarsAction>
X("create-space", "find all variables in each function");
