///@author Sudhanva Kulkarni, UC berkeley
/// this file defines a struct that accumulates the number of flops for each data type in the code

struct n_flops {
    int n_flops_double;
    int n_flops_float;
    int n_flops_half;
    int n_flops_bfloat;
    int n_flops_fp8;

    public :
    
    n_flops() : n_flops_double(0), n_flops_float(0), n_flops_half(0), n_flops_bfloat(0), n_flops_fp8(0) {};
    void reset() {
        n_flops_double = 0;
        n_flops_float = 0;
        n_flops_half = 0;
        n_flops_bfloat = 0;
        n_flops_fp8 = 0;
    }

    void add_double_flops(int n) {
        n_flops_double += n;
    }

    void add_float_flops(int n) {
        n_flops_float += n;
    }

    void add_half_flops(int n) {
        n_flops_half += n;
    }

    void add_bfloat_flops(int n) {
        n_flops_bfloat += n;
    }

    void add_fp8_flops(int n) {
        n_flops_fp8 += n;
    }

};

