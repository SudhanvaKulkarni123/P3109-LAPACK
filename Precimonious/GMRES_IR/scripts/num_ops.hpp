/// @author Sudhanva Kulkarni
/// file with utilities to count num_flops, num_casts, num_comparisons, etc
#include <iostream>
#include <vector>
struct n_flops {
    long n_flops_double;
    long n_flops_float;
    long n_flops_half;
    long n_flops_bfloat;
    long n_flops_fp8;
    long n_flops_fp6;
    long n_flops_fp4;
    float work_factor;

public:
    n_flops(float WF = 2.0) : n_flops_double(0), n_flops_float(0), n_flops_half(0), n_flops_bfloat(0), n_flops_fp8(0), n_flops_fp6(0), n_flops_fp4(0), work_factor(WF) {}

    void reset() {
        n_flops_double = 0;
        n_flops_float = 0;
        n_flops_half = 0;
        n_flops_bfloat = 0;
        n_flops_fp8 = 0;
        n_flops_fp6 = 0;
        n_flops_fp4 = 0;
    }

    void add_double_flops(long n) {
        n_flops_double += n;
    }

    void add_float_flops(long n) {
        n_flops_float += n;
    }

    void add_half_flops(long n) {
        n_flops_half += n;
    }

    void add_bfloat_flops(long n) {
        n_flops_bfloat += n;
    }

    void add_fp8_flops(long n) {
        n_flops_fp8 += n;
    }

    void add_fp6_flops(long n) {
        n_flops_fp6 += n;
    }

    void add_fp4_flops(long n) {
        n_flops_fp4 += n;
    }

    // Getters for the values
    long get_double_flops() const {
        return n_flops_double;
    }

    long get_float_flops() const {
        return n_flops_float;
    }

    long get_half_flops() const {
        return n_flops_half;
    }

    long get_bfloat_flops() const {
        return n_flops_bfloat;
    }

    long get_fp8_flops() const {
        return n_flops_fp8;
    }

    long get_fp6_flops() const {
        return n_flops_fp6;
    }

    long get_fp4_flops() const {
        return n_flops_fp4;
    }


    long total_flops() {
        return get_fp4_flops() + get_fp6_flops() + get_fp8_flops() + get_bfloat_flops() + get_half_flops() + get_float_flops() + get_double_flops();
    }

    void add_struct(n_flops& other) {
        n_flops_double += other.get_double_flops();
        n_flops_float += other.get_float_flops();
        n_flops_half += other.get_half_flops();
        n_flops_bfloat += other.get_bfloat_flops();
        n_flops_fp8 += other.get_fp8_flops();
        n_flops_fp6 += other.get_fp6_flops();
        n_flops_fp4 += other.get_fp4_flops();

    }

    void print_stats(bool use_sci = false, bool ultra_low = true) {
        if(use_sci) std::cout << std::scientific;

        if(ultra_low) {
            std::cout << "==========================================================================================================================================================\n";
        std::cout << "flop count per data type as follows - \n";
        std::cout << "number of fp64 flops : " << (double)get_double_flops() <<  "          number of fp64 flops normalized to fp4 : " << (double)(16*get_double_flops()) << "\n";
        std::cout << "number of fp32 flops : " << (double)get_float_flops() <<  "          number of fp32 flops normalized to fp4 : " << (double)(8*get_float_flops()) << "\n";
        std::cout << "number of fp16 flops : " << (double)get_half_flops() <<  "          number of fp16 flops normalized to fp4 : " << (double)(4*get_half_flops()) << "\n";
        std::cout << "number of bfloat flops : " << (double)get_bfloat_flops() <<  "          number of bf16 flops normalized to fp4 : " << (double)(4*get_bfloat_flops()) << "\n";
        std::cout << "number of fp8 flops : " << (double)get_fp8_flops() <<  "          number of fp8 flops normalized to fp4 : " << (double)(2*get_fp8_flops()) << "\n";
        std::cout << "number of fp6 flops : " << (double)get_fp6_flops() <<  "          number of fp6 flops normalized to fp4 : " << (double)(3*get_fp6_flops())/2.0 << "\n";
        std::cout << "number of fp4 flops : " << (double)get_fp4_flops() <<  "          number of fp4 flops normalized to fp4 : " << (double)(get_fp4_flops()) << "\n";

        std::cout << "==========================================================================================================================================================\n";

        } else {
        std::cout << "==========================================================================================================================================================\n";
        std::cout << "flop count per data type as follows - \n";
        std::cout << "number of fp64 flops : " << (double)get_double_flops() <<  "          number of fp64 flops normalized to fp4 : " << (double)(8*get_double_flops()) << "\n";
        std::cout << "number of fp32 flops : " << (double)get_float_flops() <<  "          number of fp32 flops normalized to fp4 : " << (double)(4*get_float_flops()) << "\n";
        std::cout << "number of fp16 flops : " << (double)get_half_flops() <<  "          number of fp16 flops normalized to fp4 : " << (double)(2*get_half_flops()) << "\n";
        std::cout << "number of bfloat flops : " << (double)get_bfloat_flops() <<  "          number of bf16 flops normalized to fp4 : " << (double)(2*get_bfloat_flops()) << "\n";
        std::cout << "number of fp8 flops : " << (double)get_fp8_flops() <<  "          number of fp8 flops normalized to fp4 : " << (double)(get_fp8_flops()) << "\n";
        std::cout << "==========================================================================================================================================================\n";
        }

        return;
    }

    double compare_with(std::vector<n_flops>& algo) {
        long double_count = 0;
        long float_count = 0;
        long half_count = 0;
        long bfloat_count = 0;
        long fp8_count = 0;
        long fp6_count = 0;
        long fp4_count = 0;

        for( auto a : algo ) {
            double_count += a.get_double_flops();
            float_count += a.get_float_flops();
            half_count += a.get_half_flops();
            bfloat_count += a.get_bfloat_flops();
            fp8_count += a.get_fp8_flops();
            fp6_count += a.get_fp6_flops();
            fp4_count += a.get_fp4_flops();
        }

        long algo_work = 16*double_count + 8*float_count + 4*half_count + 4*bfloat_count + 2*fp8_count + 3*fp6_count/2  + fp4_count ;
        long curr_work = 16*get_double_flops() + 8*get_float_flops() + 4*get_half_flops() + 4*get_bfloat_flops() + 2*get_fp8_flops() + 3*get_fp6_flops()/2 + get_fp4_flops();

        return ((double) algo_work)/((double)curr_work) ;

    }

    double report_total() {
        long algo_work = 16*get_double_flops() + 8*get_float_flops() + 4*get_half_flops() + 4*get_bfloat_flops() + 2*get_fp8_flops() + 3*get_fp6_flops()/2 + get_fp4_flops();
        return (double) algo_work;
    }

    void log_results(std::ofstream& logfile, bool use_sci = true) {
        if(use_sci) std::cout << std::scientific;
        logfile << "double flops : " << (double)get_double_flops() << "\n";
        logfile << "float flops : " << (double)get_float_flops() << "\n";
        logfile << "half flops : " << (double)get_half_flops() << "\n";
        logfile << "bfloat flops : " << (double)get_bfloat_flops() << "\n";
        logfile << "fp8 flops : " << (double)get_fp8_flops() << "\n";
        logfile << "fp6 flops : " << (double)get_fp6_flops() << "\n";
        logfile << "fp4 flops : " << (double)get_fp4_flops() << "\n";
        return;
    }

    static void log_all(std::ofstream& logfile, std::vector<n_flops>& algo, bool use_sci = true) {
        if(use_sci) std::cout << std::scientific;

        long double_count = 0;
        long float_count = 0;
        long half_count = 0;
        long bfloat_count = 0;
        long fp8_count = 0;
        long fp6_count = 0;
        long fp4_count = 0;

        for( auto a : algo ) {
            double_count += a.get_double_flops();
            float_count += a.get_float_flops();
            half_count += a.get_half_flops();
            bfloat_count += a.get_bfloat_flops();
            fp8_count += a.get_fp8_flops();
            fp6_count += a.get_fp6_flops();
            fp4_count += a.get_fp4_flops();
        }

        logfile << "double flops : " << (double)double_count << "\n";
        logfile << "float flops : " << (double)float_count << "\n";
        logfile << "half flops : " << (double)half_count << "\n";
        logfile << "bfloat flops : " << (double)bfloat_count << "\n";
        logfile << "fp8 flops : " << (double)fp8_count << "\n";
        logfile << "fp6 flops : " << (double)fp6_count << "\n";
        logfile << "fp4 flops : " << (double)fp4_count << "\n";
        return;
    }
};


struct n_comp {
    long n_comps_double;
    long n_comps_float;
    long n_comps_half;
    long n_comps_bfloat;
    long n_comps_fp8;

public:
    n_comp() : n_comps_double(0), n_comps_float(0), n_comps_half(0), n_comps_bfloat(0), n_comps_fp8(0) {}

    void reset() {
        n_comps_double = 0;
        n_comps_float = 0;
        n_comps_half = 0;
        n_comps_bfloat = 0;
        n_comps_fp8 = 0;
    }

    void add_double_comps(long n) {
        n_comps_double += n;
    }

    void add_float_comps(long n) {
        n_comps_float += n;
    }

    void add_half_comps(long n) {
        n_comps_half += n;
    }

    void add_bfloat_comps(long n) {
        n_comps_bfloat += n;
    }

    void add_fp8_comps(long n) {
        n_comps_fp8 += n;
    }

    // Getters for the values
    long get_double_comps() const {
        return n_comps_double;
    }

    long get_float_comps() const {
        return n_comps_float;
    }

    long get_half_comps() const {
        return n_comps_half;
    }

    long get_bfloat_comps() const {
        return n_comps_bfloat;
    }

    long get_fp8_comps() const {
        return n_comps_fp8;
    }
};


struct n_casts {
    long n_casts_double_to_float;
    long n_casts_float_to_double;
    long n_casts_double_to_half;
    long n_casts_half_to_double;
    long n_casts_float_to_half;
    long n_casts_half_to_float;
    long n_casts_bfloat_to_fp8;
    long n_casts_fp8_to_bfloat;
    long n_casts_double_to_bfloat;
    long n_casts_bfloat_to_double;
    long n_casts_float_to_bfloat;
    long n_casts_bfloat_to_float;
    long n_casts_half_to_bfloat;
    long n_casts_bfloat_to_half;
    long n_casts_fp8_to_float;
    long n_casts_float_to_fp8;
    long n_casts_fp8_to_half;
    long n_casts_half_to_fp8;

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
    void add_double_to_float_cast(long n) { n_casts_double_to_float += n; }
    void add_float_to_double_cast(long n) { n_casts_float_to_double += n; }
    void add_double_to_half_cast(long n) { n_casts_double_to_half += n; }
    void add_half_to_double_cast(long n) { n_casts_half_to_double += n; }
    void add_float_to_half_cast(long n) { n_casts_float_to_half += n; }
    void add_half_to_float_cast(long n) { n_casts_half_to_float += n; }
    void add_bfloat_to_fp8_cast(long n) { n_casts_bfloat_to_fp8 += n; }
    void add_fp8_to_bfloat_cast(long n) { n_casts_fp8_to_bfloat += n; }
    void add_double_to_bfloat_cast(long n) { n_casts_double_to_bfloat += n; }
    void add_bfloat_to_double_cast(long n) { n_casts_bfloat_to_double += n; }
    void add_float_to_bfloat_cast(long n) { n_casts_float_to_bfloat += n; }
    void add_bfloat_to_float_cast(long n) { n_casts_bfloat_to_float += n; }
    void add_half_to_bfloat_cast(long n) { n_casts_half_to_bfloat += n; }
    void add_bfloat_to_half_cast(long n) { n_casts_bfloat_to_half += n; }
    void add_fp8_to_float_cast(long n) { n_casts_fp8_to_float += n; }
    void add_float_to_fp8_cast(long n) { n_casts_float_to_fp8 += n; }
    void add_fp8_to_half_cast(long n) { n_casts_fp8_to_half += n; }
    void add_half_to_fp8_cast(long n) { n_casts_half_to_fp8 += n; }

    // Getters for the values
    long get_double_to_float_casts() const { return n_casts_double_to_float; }
    long get_float_to_double_casts() const { return n_casts_float_to_double; }
    long get_double_to_half_casts() const { return n_casts_double_to_half; }
    long get_half_to_double_casts() const { return n_casts_half_to_double; }
    long get_float_to_half_casts() const { return n_casts_float_to_half; }
    long get_half_to_float_casts() const { return n_casts_half_to_float; }
    long get_bfloat_to_fp8_casts() const { return n_casts_bfloat_to_fp8; }
    long get_fp8_to_bfloat_casts() const { return n_casts_fp8_to_bfloat; }
    long get_double_to_bfloat_casts() const { return n_casts_double_to_bfloat; }
    long get_bfloat_to_double_casts() const { return n_casts_bfloat_to_double; }
    long get_float_to_bfloat_casts() const { return n_casts_float_to_bfloat; }
    long get_bfloat_to_float_casts() const { return n_casts_bfloat_to_float; }
    long get_half_to_bfloat_casts() const { return n_casts_half_to_bfloat; }
    long get_bfloat_to_half_casts() const { return n_casts_bfloat_to_half; }
    long get_fp8_to_float_casts() const { return n_casts_fp8_to_float; }
    long get_float_to_fp8_casts() const { return n_casts_float_to_fp8; }
    long get_fp8_to_half_casts() const { return n_casts_fp8_to_half; }
    long get_half_to_fp8_casts() const { return n_casts_half_to_fp8; }
};
