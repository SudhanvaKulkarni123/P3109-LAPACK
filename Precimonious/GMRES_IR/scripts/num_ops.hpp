/// @author Sudhanva Kulkarni
/// file with utilities to count num_flops, num_casts, num_comparisons, etc
#include <iostream>
struct n_flops {
    int n_flops_double;
    int n_flops_float;
    int n_flops_half;
    int n_flops_bfloat;
    int n_flops_fp8;

public:
    n_flops() : n_flops_double(0), n_flops_float(0), n_flops_half(0), n_flops_bfloat(0), n_flops_fp8(0) {}

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

    // Getters for the values
    int get_double_flops() const {
        return n_flops_double;
    }

    int get_float_flops() const {
        return n_flops_float;
    }

    int get_half_flops() const {
        return n_flops_half;
    }

    int get_bfloat_flops() const {
        return n_flops_bfloat;
    }

    int get_fp8_flops() const {
        return n_flops_fp8;
    }

    void print_stats() {
        std::cout << "=========================================\n";
        std::cout << "flop count per data type as follows - \n";
        std::cout << "number of fp64 flops : " << get_double_flops() << "\n";
        std::cout << "number of fp32 flops : " << get_float_flops() << "\n";
        std::cout << "number of fp16 flops : " << get_half_flops() << "\n";
        std::cout << "number of bfloat flops : " << get_bfloat_flops() << "\n";
        std::cout << "number of fp8 flops : " << get_fp8_flops() << "\n";
        std::cout << "=========================================\n";
    }
};


struct n_comp {
    int n_comps_double;
    int n_comps_float;
    int n_comps_half;
    int n_comps_bfloat;
    int n_comps_fp8;

public:
    n_comp() : n_comps_double(0), n_comps_float(0), n_comps_half(0), n_comps_bfloat(0), n_comps_fp8(0) {}

    void reset() {
        n_comps_double = 0;
        n_comps_float = 0;
        n_comps_half = 0;
        n_comps_bfloat = 0;
        n_comps_fp8 = 0;
    }

    void add_double_comps(int n) {
        n_comps_double += n;
    }

    void add_float_comps(int n) {
        n_comps_float += n;
    }

    void add_half_comps(int n) {
        n_comps_half += n;
    }

    void add_bfloat_comps(int n) {
        n_comps_bfloat += n;
    }

    void add_fp8_comps(int n) {
        n_comps_fp8 += n;
    }

    // Getters for the values
    int get_double_comps() const {
        return n_comps_double;
    }

    int get_float_comps() const {
        return n_comps_float;
    }

    int get_half_comps() const {
        return n_comps_half;
    }

    int get_bfloat_comps() const {
        return n_comps_bfloat;
    }

    int get_fp8_comps() const {
        return n_comps_fp8;
    }
};


struct n_casts {
    int n_casts_double_to_float;
    int n_casts_float_to_double;
    int n_casts_double_to_half;
    int n_casts_half_to_double;
    int n_casts_float_to_half;
    int n_casts_half_to_float;
    int n_casts_bfloat_to_fp8;
    int n_casts_fp8_to_bfloat;
    int n_casts_double_to_bfloat;
    int n_casts_bfloat_to_double;
    int n_casts_float_to_bfloat;
    int n_casts_bfloat_to_float;
    int n_casts_half_to_bfloat;
    int n_casts_bfloat_to_half;
    int n_casts_fp8_to_float;
    int n_casts_float_to_fp8;
    int n_casts_fp8_to_half;
    int n_casts_half_to_fp8;

public:
    n_casts() : n_casts_double_to_float(0), n_casts_float_to_double(0),
                n_casts_double_to_half(0), n_casts_half_to_double(0),
                n_casts_float_to_half(0), n_casts_half_to_float(0),
                n_casts_bfloat_to_fp8(0), n_casts_fp8_to_bfloat(0),
                n_casts_double_to_bfloat(0), n_casts_bfloat_to_double(0),
                n_casts_float_to_bfloat(0), n_casts_bfloat_to_float(0),
                n_casts_half_to_bfloat(0), n_casts_bfloat_to_half(0),
                n_casts_fp8_to_float(0), n_casts_float_to_fp8(0),
                n_casts_fp8_to_half(0), n_casts_half_to_fp8(0) {}

    void reset() {
        n_casts_double_to_float = 0;
        n_casts_float_to_double = 0;
        n_casts_double_to_half = 0;
        n_casts_half_to_double = 0;
        n_casts_float_to_half = 0;
        n_casts_half_to_float = 0;
        n_casts_bfloat_to_fp8 = 0;
        n_casts_fp8_to_bfloat = 0;
        n_casts_double_to_bfloat = 0;
        n_casts_bfloat_to_double = 0;
        n_casts_float_to_bfloat = 0;
        n_casts_bfloat_to_float = 0;
        n_casts_half_to_bfloat = 0;
        n_casts_bfloat_to_half = 0;
        n_casts_fp8_to_float = 0;
        n_casts_float_to_fp8 = 0;
        n_casts_fp8_to_half = 0;
        n_casts_half_to_fp8 = 0;
    }

    // Add functions for each casting type
    void add_double_to_float_cast(int n) { n_casts_double_to_float += n; }
    void add_float_to_double_cast(int n) { n_casts_float_to_double += n; }
    void add_double_to_half_cast(int n) { n_casts_double_to_half += n; }
    void add_half_to_double_cast(int n) { n_casts_half_to_double += n; }
    void add_float_to_half_cast(int n) { n_casts_float_to_half += n; }
    void add_half_to_float_cast(int n) { n_casts_half_to_float += n; }
    void add_bfloat_to_fp8_cast(int n) { n_casts_bfloat_to_fp8 += n; }
    void add_fp8_to_bfloat_cast(int n) { n_casts_fp8_to_bfloat += n; }
    void add_double_to_bfloat_cast(int n) { n_casts_double_to_bfloat += n; }
    void add_bfloat_to_double_cast(int n) { n_casts_bfloat_to_double += n; }
    void add_float_to_bfloat_cast(int n) { n_casts_float_to_bfloat += n; }
    void add_bfloat_to_float_cast(int n) { n_casts_bfloat_to_float += n; }
    void add_half_to_bfloat_cast(int n) { n_casts_half_to_bfloat += n; }
    void add_bfloat_to_half_cast(int n) { n_casts_bfloat_to_half += n; }
    void add_fp8_to_float_cast(int n) { n_casts_fp8_to_float += n; }
    void add_float_to_fp8_cast(int n) { n_casts_float_to_fp8 += n; }
    void add_fp8_to_half_cast(int n) { n_casts_fp8_to_half += n; }
    void add_half_to_fp8_cast(int n) { n_casts_half_to_fp8 += n; }

    // Getters for the values
    int get_double_to_float_casts() const { return n_casts_double_to_float; }
    int get_float_to_double_casts() const { return n_casts_float_to_double; }
    int get_double_to_half_casts() const { return n_casts_double_to_half; }
    int get_half_to_double_casts() const { return n_casts_half_to_double; }
    int get_float_to_half_casts() const { return n_casts_float_to_half; }
    int get_half_to_float_casts() const { return n_casts_half_to_float; }
    int get_bfloat_to_fp8_casts() const { return n_casts_bfloat_to_fp8; }
    int get_fp8_to_bfloat_casts() const { return n_casts_fp8_to_bfloat; }
    int get_double_to_bfloat_casts() const { return n_casts_double_to_bfloat; }
    int get_bfloat_to_double_casts() const { return n_casts_bfloat_to_double; }
    int get_float_to_bfloat_casts() const { return n_casts_float_to_bfloat; }
    int get_bfloat_to_float_casts() const { return n_casts_bfloat_to_float; }
    int get_half_to_bfloat_casts() const { return n_casts_half_to_bfloat; }
    int get_bfloat_to_half_casts() const { return n_casts_bfloat_to_half; }
    int get_fp8_to_float_casts() const { return n_casts_fp8_to_float; }
    int get_float_to_fp8_casts() const { return n_casts_float_to_fp8; }
    int get_fp8_to_half_casts() const { return n_casts_fp8_to_half; }
    int get_half_to_fp8_casts() const { return n_casts_half_to_fp8; }
};
