// trans-type Clang plugin
//

#include "TransformType.h"
#include <iostream>
#include <fstream>
#include <string>
#include "../utilities/json.hpp"
#include <regex>

using namespace clang;
using namespace std;
using namespace llvm;


nlohmann::json jf;
string funcName;
string parm_Name = "";
string type_Name = "";
string new_TypeName = "";
string OUTPUT_PATH = "";
string INPUT_CONFIG = "config_temp.json";
int libcall_flag = 0;

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


size_t countSubString(const string& str, const string& sub) {
   size_t ret = 0;
   size_t loc = str.find(sub);
   while (loc != string::npos) {
      ++ret;
      loc = str.find(sub, loc+1);
   }
   return ret;
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


void PrintSourceRange(SourceRange range, ASTContext* astContext) {
    PRINT_DEBUG_MESSAGE("\toffset: " << astContext->getSourceManager().getFileOffset(range.getBegin()) << " " <<
    astContext->getSourceManager().getFileOffset(range.getEnd()));
}

void PrintStatement(string prefix, const Stmt* st, ASTContext* astContext) {
    std::string statementText;
    raw_string_ostream wrap(statementText);
    st->printPretty(wrap, NULL, PrintingPolicy(astContext->getLangOpts()));
    PRINT_DEBUG_MESSAGE(prefix << st->getStmtClassName() << ", " << statementText);
    PrintSourceRange(st->getSourceRange(), astContext);
}

string getFileName(ASTContext *Context, VarDecl *val) {
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
    string FileName = basename(Entry->getName().str());
    return FileName;
}

string getFileName(ASTContext *Context, DeclRefExpr *val) {
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
    string FileName = basename(Entry->getName().str());
    return FileName;
}


string getNewType(string valueName, string typeName, string funcName, string fileName, string location, string source_loc) {
    for (nlohmann::json::iterator it = jf.begin(); it != jf.end(); ++it) {
      if (location == "call") {
          if (valueName == jf[it.key()]["name"]
          && funcName == jf[it.key()]["function"]
          && fileName == jf[it.key()]["file"]
          && source_loc == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["switch"]) {
                return "";
            }
            else {
                return jf[it.key()]["switch"];
            }
          }
      }
      else if (location != "globalVar") {
          if (valueName == jf[it.key()]["name"]
          && funcName == jf[it.key()]["function"]
          && fileName == jf[it.key()]["file"]
          && location == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["type"]) {
                return "";
            }
            else {
                return jf[it.key()]["type"];
            }
          }
      }
      else {
          if (valueName == jf[it.key()]["name"]
          && fileName == jf[it.key()]["file"]
          && location == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["type"]) {
                return "";
            }
            else {
                return jf[it.key()]["type"];
            }
          }
      }

    }
    return "";
}

string basename(std::string path) {
    return std::string( std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(), path.end());
}

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

//q : how can I find the offset from one sourcelocation to another?

//a : 

bool TransformTypeAliasDecl(ASTContext *Context, TypeAliasDecl *typdecl) {
    SourceRange range = typdecl->getSourceRange();
    SourceManager &SrcMgr = Context->getSourceManager();
    SourceLocation ST = range.getBegin();
    SourceLocation END = range.getEnd();
    clang::QualType underlyingType = typdecl->getUnderlyingType();
    const clang::Type* typeObj = underlyingType.getTypePtrOrNull();
    
    if (!typeObj) {
        std::cerr << "Type object is null." << std::endl;
        return false;
    }

    FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);
    if (info.isFloatingPoint) {
        std::string typeStr = typeObj->getCanonicalTypeInternal().getAsString();
        typeStr = filter(typeStr);
        std::string newType = getNewType(typdecl->getNameAsString(), typeStr, funcName, funcName + ".hpp", "localVar", "");

        if (!newType.empty()) {
            rewriter.ReplaceText(range, "using " + typdecl->getNameAsString() + " = " + newType + ";");
        }
    }

    return true;
}


bool TransformTemplateParam(ASTContext *Context, CallExpr *expr) {
    SourceManager& SrcMgr = Context->getSourceManager();
    SourceRange range = expr->getSourceRange();

    if (range.isInvalid()) {
        std::cerr << "Invalid source range for CallExpr." << std::endl;
        return false;
    }

    // Get the source code of the CallExpr
    std::string exprSource = Lexer::getSourceText(CharSourceRange::getTokenRange(range), SrcMgr, Context->getLangOpts()).str();
    if(exprSource.find("GMRES_IR") != std::string::npos) { 
      

    // Find the template argument list
    
    std::regex templateArgsPattern("GMRES_IR<.*(,.)*>.n");
    std::smatch match;
    if (std::regex_search(exprSource, match, templateArgsPattern)) {
        std::string templateArgs = match.str();
        

        // Split the template arguments and replace each with a new type
        std::string newTemplateArgs = "GMRES_IR<";
        std::istringstream ss(templateArgs.substr(1, templateArgs.size() - 3)); // remove '<' and '>'
        std::string arg;
        int cnt = 0;
        while (std::getline(ss, arg, ',')) {
            cnt++;
            arg = filter(arg);
            std::string newType = getNewType("main_template_arg_" + std::to_string(cnt), arg, "main", "GMRES_IR.cpp", "localVar", "");
            if (!newType.empty()) {
                newTemplateArgs += newType + ",";
            } else {
                newTemplateArgs += arg + ",";
            }
        }
        newTemplateArgs.back() = '>'; // Replace the last comma with '>'
        newTemplateArgs += "(n";


        // Replace the template arguments in the original expression
        std::string newExprSource = std::regex_replace(exprSource, templateArgsPattern, newTemplateArgs);

        // Use the rewriter to replace the original source with the new source
        rewriter.ReplaceText(range, newExprSource);
    } else {
        std::cerr << "No template arguments found in the expression." << std::endl;
    }
    }

    return true;
}


    bool GlobVisitor::VisitVarDecl(VarDecl *val){

    return true;
}

bool VarsVisitor::VisitVarDecl(VarDecl *val){
    
    return true;
}


bool FuncStmtVisitor::VisitStmt(Stmt *st) {
//    PrintStatement("Statement: ", st, astContext);
    
    if( clang::DeclStmt* declSt = dyn_cast<clang::DeclStmt>(st)) {
        for(auto decl : declSt->decls()) {
            if ( clang::TypeAliasDecl* type = dyn_cast<clang::TypeAliasDecl>(decl)) {
                TransformTypeAliasDecl(astContext, type);

            } else {
                return true;
            }
        }
    } else if( clang::CallExpr* callSt = dyn_cast<clang::CallExpr>(st)) {
        if (libcall_flag == 1) {
            if ( clang::FunctionDecl* libcall = callSt->getDirectCallee()) {
                //need a trnasform templates function
                TransformTemplateParam(astContext, callSt);
            }
        }
    }
    return true;
}

bool FuncVisitor::VisitFunctionDecl(FunctionDecl* func) {
    if (!func->doesThisDeclarationHaveABody())
        return true;

    funcName = func->getNameInfo().getName().getAsString();
    if(funcName == "operator()") return true;
    if((funcName == "main") || isBLAS(funcName) || (funcName == "GMRES_IR") || isLAPACK(funcName)) {
        
    libcall_flag = 1;
    st_visitor->TraverseStmt(func->getBody());
    vars_visitor->TraverseStmt(func->getBody());
    }

    return true;


}



void TransTypeConsumer::HandleTranslationUnit(ASTContext &Context){
    FileID id = rewriter.getSourceMgr().getMainFileID();
    string basefilename = basename(rewriter.getSourceMgr().getFilename(rewriter.getSourceMgr().getLocForStartOfFile(id)).str());
    string filename;
    if(basefilename == "GMRES_IR.cpp") {
        filename = OUTPUT_PATH + basefilename;
    }
    else {
        filename = rewriter.getSourceMgr().getFilename(rewriter.getSourceMgr().getLocForStartOfFile(id)).str();
    }
    
//    PRINT_DEBUG_MESSAGE(filename);
    std::ifstream ifs(INPUT_CONFIG);
    jf = nlohmann::json::parse(ifs);
    glob_visitor->TraverseDecl(Context.getTranslationUnitDecl());
    func_visitor->TraverseDecl(Context.getTranslationUnitDecl());

    // Create an output file to write the updated code
    std::error_code OutErrorInfo;
    std::error_code ok;
    const RewriteBuffer *RewriteBuf = rewriter.getRewriteBufferFor(id);
    if (RewriteBuf) { 
        llvm::raw_fd_ostream outFile(llvm::StringRef(filename),
                OutErrorInfo, llvm::sys::fs::OF_None);
        if (OutErrorInfo == ok) {
                outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
                PRINT_DEBUG_MESSAGE("Output file created - " << filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << filename);
        }
    }else {
        PRINT_DEBUG_MESSAGE("No file created!");
    }
}

unique_ptr<ASTConsumer> TransTypeAction::CreateASTConsumer(CompilerInstance &CI, StringRef file) {
    return make_unique<TransTypeConsumer>(&CI);
}

bool TransTypeAction::ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    // To be written...
    llvm::errs() << "Plugin arg size = " << args.size() << "\n";
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (args[i] == "-output-path") {
        if (i+1 < e)
            OUTPUT_PATH = args[i+1]; // ./temp-NPB3.3-SER-C/CG/
        else
            PRINT_DEBUG_MESSAGE("Missing output path! Could not generate output file.");
        llvm::errs() << "Output path = " << OUTPUT_PATH << "\n";
      }
      if (args[i] == "-input-config") {
        if (i+1 < e)
            INPUT_CONFIG = args[i+1]; // config_temp.json
        else
            PRINT_DEBUG_MESSAGE("Missing input config! Could not generate output file.");
        llvm::errs() << "Input config  = " << INPUT_CONFIG << "\n";
      }
    }
    return true;
  }



//Q :How will clang call this plugin
//A : 
static clang::FrontendPluginRegistry::Add<TransTypeAction>
X("trans-type", "transform all floating point variables' type from double to float");
 