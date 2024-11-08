
enum class factorization_type { standard_LU, two_prec_LU, three_prec_LU, low_prec_store_LU, scaled_two_prec_store_LU, block_low_prec_LU};
enum class pivoting_scheme { partial, complete, none};

int set_matrix_params(int& n, float& cond, bool& is_symmetric, bool& diag_dom, nlohmann::json& outer_settings)
{

    string tmp;
    auto settings = outer_settings["matrix settings"];
    tmp = settings["n"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    n = stoi(tmp);

    tmp = settings["condition number"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    cond = stof(tmp);

    tmp = settings["symmetric"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    is_symmetric = (tmp == "true");

    tmp = settings["diag_dom"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    diag_dom = (tmp == "true");

    if(diag_dom) cout << "it is true\n";

    return 0;

}

int set_factorization_params(factorization_type& fact_type, string& lowest_prec, string& highest_prec, bool& is_rank_revealing,  pivoting_scheme& pivoting_scheme, int& num_precisions, int& block_size, bool& use_microscal, int& stopping_pos, double& switching_val, double& scaling_factor, double& tolerance, float& dropping_prob, nlohmann::json& outer_settings) 
{
    string tmp;
    auto settings = outer_settings["factorization settings"];
    tmp = settings["factorization method"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "two_prec_LU") fact_type = factorization_type::two_prec_LU;
    else if(tmp == "three_prec_LU") fact_type = factorization_type::three_prec_LU;
    else if(tmp == "low_prec_store") fact_type = factorization_type::low_prec_store_LU;
    else if(tmp == "scaled_two_prec_store") fact_type = factorization_type::scaled_two_prec_store_LU;
    else if(tmp == "block_low_prec") fact_type = factorization_type::block_low_prec_LU;
    else fact_type = factorization_type::standard_LU;

    tmp = settings["lowest precision"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    lowest_prec = tmp;

    tmp = settings["highest precision"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    highest_prec = tmp;

    tmp = settings["is rank revealing"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "true") is_rank_revealing = true;
    else is_rank_revealing = false;

    tmp = settings["pivoting scheme"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "partial") pivoting_scheme = pivoting_scheme::partial;
    else if(tmp == "full") pivoting_scheme = pivoting_scheme::complete;
    else if(tmp == "none") pivoting_scheme = pivoting_scheme::none;

    tmp = settings["num_precisions"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    num_precisions = stoi(tmp);

    tmp = settings["block size"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    block_size = stoi(tmp);

    tmp = settings["use Microscaling"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "true") use_microscal = true;
    else use_microscal = false;

    tmp = settings["stopping pos"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    stopping_pos = stoi(tmp);

    tmp = settings["switching val"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    switching_val = stof(tmp);

    tmp = settings["scaling factor"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    std::cout << "scaling factor is : " << tmp << std::endl;
    scaling_factor = stof(tmp);

    tmp = settings["tolerance"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    std::cout << "tolerance is : " << tmp << std::endl;
    tolerance = stof(tmp);

    tmp = settings["dropping_prob"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    std::cout << "dropping_prob is : " << tmp << std::endl;
    dropping_prob = stof(tmp);

    return 0;
}

int set_refinement_settings(int& max_gmres_iter, int& num_iter_1, int& total_num_iter, refinement_type& refinement_method, kernel_type& precond_kernel, int& num_IR_iter, int& inner_gmres_num, ortho_type& arnoldi_subroutine, double& conv_thresh, nlohmann::json& outer_settings)
{
    string tmp;
    auto settings = outer_settings["GMRES settings"];
    tmp = settings["max_gmres_iter"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    max_gmres_iter = stoi(tmp);

    tmp = settings["num_iter_1"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    num_iter_1 = stoi(tmp);

    tmp = settings["total_num_iter"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    total_num_iter = stoi(tmp);

    tmp = settings["refinement_method"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    std::cout << "refinement method is : " << tmp << std::endl;
    if(tmp == "NVIDIA") refinement_method = refinement_type::NVIDIA_IR;
    else if(tmp == "GMRES") refinement_method = refinement_type::GMRES_IR;
    else if(tmp == "NO_IR") refinement_method = refinement_type::NO_IR;

    tmp = settings["precond_kernel"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "RIGHT_LU") precond_kernel = kernel_type::RIGHT_LU;
    else if(tmp == "LEFT_LU") precond_kernel = kernel_type::LEFT_LU;
    else if(tmp == "RIGHT_GMRES") precond_kernel = kernel_type::RIGHT_GMRES;
    else if(tmp == "SPLIT_LU") precond_kernel = kernel_type::SPLIT_LU;
    else if(tmp == "NONE") precond_kernel = kernel_type::NONE;

    tmp = settings["num_IR_iter"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    num_IR_iter = stoi(tmp);

    tmp = settings["inner_gmres_num"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    inner_gmres_num = stoi(tmp);

    tmp = settings["arnoldi_subroutine"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    if(tmp == "MGS") arnoldi_subroutine = ortho_type::MGS;
    else if(tmp == "Deterministic Householder") arnoldi_subroutine = ortho_type::DHH;
    else if(tmp == "Random Householder") arnoldi_subroutine = ortho_type::RHH;

    tmp = settings["conv_thresh"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    conv_thresh = stof(tmp);


    return 0;


}
