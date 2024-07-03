; ModuleID = 'funarc_8.cpp'
source_filename = "funarc_8.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"class.Eigen::symbolic::SymbolExpr" = type { i8 }
%"class.Eigen::symbolic::AddExpr" = type { %"class.Eigen::symbolic::SymbolExpr", %"class.Eigen::symbolic::ValueExpr" }
%"class.Eigen::symbolic::ValueExpr" = type { i8 }
%"class.Eigen::internal::FixedInt" = type { i8 }
%"struct.Eigen::internal::all_t" = type { i8 }
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<>::param_type" }
%"struct.std::uniform_int_distribution<>::param_type" = type { i32, i32 }
%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }
%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"struct.Eigen::bfloat16" = type { %"struct.Eigen::bfloat16_impl::bfloat16_base" }
%"struct.Eigen::bfloat16_impl::bfloat16_base" = type { %"struct.Eigen::bfloat16_impl::__bfloat16_raw" }
%"struct.Eigen::bfloat16_impl::__bfloat16_raw" = type { i16 }
%"struct.Eigen::half" = type { %"struct.Eigen::half_impl::half_base" }
%"struct.Eigen::half_impl::half_base" = type { %"struct.Eigen::half_impl::__half_raw" }
%"struct.Eigen::half_impl::__half_raw" = type { i16 }
%"class.ml_dtypes::float8_internal::float8_ieee_p" = type { %"class.ml_dtypes::float8_internal::float8_base" }
%"class.ml_dtypes::float8_internal::float8_base" = type { i8 }
%"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag" = type { i8 }

$_ZN5Eigen8symbolic10SymbolExprINS_8internal17symbolic_last_tagEEC2Ev = comdat any

$_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEEplILi1EEENS0_7AddExprIS5_NS0_9ValueExprINS3_8FixedIntIXT_EEEEEEESB_ = comdat any

$_ZNK5Eigen8internal8FixedIntILi1EEclEv = comdat any

$_ZN5Eigen8internal5all_tC2Ev = comdat any

$_ZNSt24uniform_int_distributionIiEC2Eii = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em = comdat any

$_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev = comdat any

$_Z5fun_aIfN9ml_dtypes15float8_internal13float8_ieee_pILi3EEES3_S3_S3_EvRT_RT0_RT1_RT2_RT3_ = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv = comdat any

$_ZN9ml_dtypes15float8_internallsINS0_13float8_ieee_pILi3EEEEERSoS4_RKNS0_11float8_baseIT_EE = comdat any

$_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ev = comdat any

$_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEE7derivedEv = comdat any

$_ZN5Eigen8symbolic9ValueExprINS_8internal8FixedIntILi1EEEEC2Ev = comdat any

$_ZN5Eigen8symbolic7AddExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEENS0_9ValueExprINS3_8FixedIntILi1EEEEEEC2ERKS5_RKS9_ = comdat any

$_ZNSt24uniform_int_distributionIiE10param_typeC2Eii = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm = comdat any

$_ZNSt8__detail5__modImTnT_Lm4294967296ETnS1_Lm1ETnS1_Lm0EEES1_S1_ = comdat any

$_ZNSt8__detail5__modImTnT_Lm624ETnS1_Lm1ETnS1_Lm0EEES1_S1_ = comdat any

$_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm = comdat any

$_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm = comdat any

$_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd = comdat any

$_Z4acosIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_ = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEdvERKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEIiSt9enable_ifILb1EvEEET_ = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvfEv = comdat any

$_Z3funIfET_S0_ = comdat any

$_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2ERKf = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEplERKS3_ = comdat any

$_Z4sqrtIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_ = comdat any

$_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEmiERKS2_ = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmlERKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ed = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EEES3_RKd = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2EhNS4_19ConstructFromRepTagE = comdat any

$_ZN9ml_dtypes15float8_internal11ConvertImplIdNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKdb = comdat any

$_ZN5Eigen6numext8bit_castImdEET_RKT0_ = comdat any

$_ZN5Eigen6numext5isinfIdEEbRKT_ = comdat any

$_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv = comdat any

$_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv = comdat any

$_ZN5Eigen6numext5isnanIdEEbRKT_ = comdat any

$_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE9quiet_NaNEv = comdat any

$_ZN9ml_dtypes15float8_internal16Stochastic_RoundImEET_S2_i = comdat any

$_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenImEET_S2_i = comdat any

$_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_ = comdat any

$_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_ = comdat any

$_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv = comdat any

$_ZSt3absd = comdat any

$_ZN5Eigen8internal10isinf_implIdEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_ = comdat any

$_ZSt5isinfd = comdat any

$_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE8infinityEv = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh = comdat any

$_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEhNS3_IS2_E19ConstructFromRepTagE = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEngEv = comdat any

$_ZN5Eigen8internal10isnan_implIdEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_ = comdat any

$_ZSt5isnand = comdat any

$_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE9quiet_NaNEv = comdat any

$_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_ = comdat any

$_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv = comdat any

$_ZNKSt24uniform_int_distributionIiE10param_type1bEv = comdat any

$_ZNKSt24uniform_int_distributionIiE10param_type1aEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv = comdat any

$_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE3maxEv = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2IiSt9enable_ifILb1EvEEET_ = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EfEES3_RKT1_ = comdat any

$_ZN9ml_dtypes15float8_internal11ConvertImplIfNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKfb = comdat any

$_ZN5Eigen6numext8bit_castIjfEET_RKT0_ = comdat any

$_ZN5Eigen6numext5isinfIfEEbRKT_ = comdat any

$_ZN5Eigen6numext5isnanIfEEbRKT_ = comdat any

$_ZN9ml_dtypes15float8_internal16Stochastic_RoundIjEET_S2_i = comdat any

$_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenIjEET_S2_i = comdat any

$_ZSt3absf = comdat any

$_ZN5Eigen8internal10isinf_implIfEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_ = comdat any

$_ZSt5isinff = comdat any

$_ZN5Eigen8internal10isnan_implIfEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_ = comdat any

$_ZSt5isnanf = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIfLb0ELb0EEET_RKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEfLb0ELb0EvE3runERKS3_b = comdat any

$_ZN5Eigen6numext5isinfIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_ = comdat any

$_ZN5Eigen16GenericNumTraitsIfE8infinityEv = comdat any

$_ZN5Eigen6numext5isnanIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_ = comdat any

$_ZN5Eigen16GenericNumTraitsIfE9quiet_NaNEv = comdat any

$_ZN5Eigen6numext8bit_castIfjEET_RKT0_ = comdat any

$_ZN5Eigen16GenericNumTraitsIfE7highestEv = comdat any

$_ZN9ml_dtypes15float8_internal3absILi3EEENS0_13float8_ieee_pIXT_EEERKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal5isnanILi3EEEbRKNS0_13float8_ieee_pIXT_EEE = comdat any

$_ZN5Eigen8internal10isinf_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_ = comdat any

$_ZN9ml_dtypes15float8_internal5isinfINS0_13float8_ieee_pILi3EEEEEbRKNS0_11float8_baseIT_EE = comdat any

$_ZNSt14numeric_limitsIfE8infinityEv = comdat any

$_ZN5Eigen8internal10isnan_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS7_EE17has_signaling_NaNntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_ = comdat any

$_ZNSt14numeric_limitsIfE9quiet_NaNEv = comdat any

$_ZNSt14numeric_limitsIfE3maxEv = comdat any

$_ZSt3sinf = comdat any

$_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmiERKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIdLb0ELb0EEET_RKS3_ = comdat any

$_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEdLb0ELb0EvE3runERKS3_b = comdat any

$_ZN5Eigen16GenericNumTraitsIdE8infinityEv = comdat any

$_ZN5Eigen16GenericNumTraitsIdE9quiet_NaNEv = comdat any

$_ZN5Eigen6numext8bit_castIdmEET_RKT0_ = comdat any

$_ZN5Eigen16GenericNumTraitsIdE7highestEv = comdat any

$_ZNSt14numeric_limitsIdE8infinityEv = comdat any

$_ZNSt14numeric_limitsIdE9quiet_NaNEv = comdat any

$_ZNSt14numeric_limitsIdE3maxEv = comdat any

$_ZN5Eigen3fixILi1EEE = comdat any

@_ZN5Eigen12placeholdersL4lastE = internal global %"class.Eigen::symbolic::SymbolExpr" zeroinitializer, align 1
@_ZN5Eigen12placeholdersL6lastp1E = internal global %"class.Eigen::symbolic::AddExpr" zeroinitializer, align 1
@_ZN5Eigen3fixILi1EEE = linkonce_odr dso_local constant %"class.Eigen::internal::FixedInt" zeroinitializer, comdat, align 1
@_ZN5Eigen12placeholdersL3allE = internal global %"struct.Eigen::internal::all_t" zeroinitializer, align 1
@_ZL12distribution = internal global %"class.std::uniform_int_distribution" zeroinitializer, align 4
@_ZL2mt = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [26 x i8] c" VERIFICATION SUCCESSFUL\0A\00", align 1
@.str.6 = private unnamed_addr constant [10 x i8] c" Zeta is \00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c" Error is \00", align 1
@.str.8 = private unnamed_addr constant [22 x i8] c" VERIFICATION FAILED\0A\00", align 1
@.str.9 = private unnamed_addr constant [10 x i8] c"./log.txt\00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"true\0A\00", align 1
@.str.12 = private unnamed_addr constant [7 x i8] c"false\0A\00", align 1
@.str.13 = private unnamed_addr constant [9 x i8] c"%20.13E\0A\00", align 1
@.str.14 = private unnamed_addr constant [11 x i8] c"./time.txt\00", align 1
@.str.15 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@.str.16 = private unnamed_addr constant [16 x i8] c"\04\03\02\02\01\01\01\01\00\00\00\00\00\00\00\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_funarc_8.cpp, ptr null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
  call void @_ZN5Eigen8symbolic10SymbolExprINS_8internal17symbolic_last_tagEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) @_ZN5Eigen12placeholdersL4lastE)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN5Eigen8symbolic10SymbolExprINS_8internal17symbolic_last_tagEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init.1() #0 section ".text.startup" {
  %1 = alloca %"class.Eigen::internal::FixedInt", align 1
  %2 = alloca %"class.Eigen::internal::FixedInt", align 1
  %3 = alloca %"class.Eigen::symbolic::AddExpr", align 1
  call void @_ZNK5Eigen8internal8FixedIntILi1EEclEv(ptr noundef nonnull align 1 dereferenceable(1) @_ZN5Eigen3fixILi1EEE)
  call void @_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEEplILi1EEENS0_7AddExprIS5_NS0_9ValueExprINS3_8FixedIntIXT_EEEEEEESB_(ptr noundef nonnull align 1 dereferenceable(1) @_ZN5Eigen12placeholdersL4lastE)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEEplILi1EEENS0_7AddExprIS5_NS0_9ValueExprINS3_8FixedIntIXT_EEEEEEESB_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca %"class.Eigen::symbolic::AddExpr", align 1
  %3 = alloca %"class.Eigen::internal::FixedInt", align 1
  %4 = alloca ptr, align 8
  %5 = alloca %"class.Eigen::symbolic::ValueExpr", align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  call void @_ZN5Eigen8symbolic9ValueExprINS_8internal8FixedIntILi1EEEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @_ZN5Eigen8symbolic7AddExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEENS0_9ValueExprINS3_8FixedIntILi1EEEEEEC2ERKS5_RKS9_(ptr noundef nonnull align 1 dereferenceable(2) %2, ptr noundef nonnull align 1 dereferenceable(1) %7, ptr noundef nonnull align 1 dereferenceable(1) %5)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNK5Eigen8internal8FixedIntILi1EEclEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init.2() #0 section ".text.startup" {
  call void @_ZN5Eigen8internal5all_tC2Ev(ptr noundef nonnull align 1 dereferenceable(1) @_ZN5Eigen12placeholdersL3allE)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal5all_tC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init.3() #0 section ".text.startup" {
  call void @_ZNSt24uniform_int_distributionIiEC2Eii(ptr noundef nonnull align 4 dereferenceable(8) @_ZL12distribution, i32 noundef 0, i32 noundef 1)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt24uniform_int_distributionIiEC2Eii(ptr noundef nonnull align 4 dereferenceable(8) %0, i32 noundef %1, i32 noundef %2) unnamed_addr #2 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds %"class.std::uniform_int_distribution", ptr %7, i32 0, i32 0
  %9 = load i32, ptr %5, align 4
  %10 = load i32, ptr %6, align 4
  call void @_ZNSt24uniform_int_distributionIiE10param_typeC2Eii(ptr noundef nonnull align 4 dereferenceable(8) %8, i32 noundef %9, i32 noundef %10)
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init.4() #0 section ".text.startup" {
  %1 = call i64 @time(ptr noundef null) #5
  call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL2mt, i64 noundef %1)
  ret void
}

; Function Attrs: nounwind
declare i64 @time(ptr noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(ptr noundef nonnull align 8 dereferenceable(5000) %0, i64 noundef %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load i64, ptr %4, align 8
  call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(ptr noundef nonnull align 8 dereferenceable(5000) %5, i64 noundef %6)
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init.5() #0 section ".text.startup" {
  call void @_ZNSt8ios_base4InitC1Ev(ptr noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %1 = call i32 @__cxa_atexit(ptr @_ZNSt8ios_base4InitD1Ev, ptr @_ZStL8__ioinit, ptr @__dso_handle) #5
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #3

; Function Attrs: nounwind
declare i32 @__cxa_atexit(ptr, ptr, ptr) #5

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z12measure_timefRd(float noundef %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 {
  %3 = alloca float, align 4
  %4 = alloca ptr, align 8
  store float %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load double, ptr %5, align 8
  %7 = fadd double %6, 4.000000e+00
  %8 = load ptr, ptr %4, align 8
  store double %7, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z12measure_timedRd(double noundef %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store double %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load double, ptr %5, align 8
  %7 = fadd double %6, 8.000000e+00
  %8 = load ptr, ptr %4, align 8
  store double %7, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z12measure_timeN5Eigen8bfloat16ERd(i16 %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 {
  %3 = alloca %"struct.Eigen::bfloat16", align 2
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds %"struct.Eigen::bfloat16", ptr %3, i32 0, i32 0
  %6 = getelementptr inbounds %"struct.Eigen::bfloat16_impl::bfloat16_base", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds %"struct.Eigen::bfloat16_impl::__bfloat16_raw", ptr %6, i32 0, i32 0
  store i16 %0, ptr %7, align 2
  store ptr %1, ptr %4, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = load double, ptr %8, align 8
  %10 = fadd double %9, 2.000000e+00
  %11 = load ptr, ptr %4, align 8
  store double %10, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z12measure_timeN5Eigen4halfERd(i16 %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 {
  %3 = alloca %"struct.Eigen::half", align 2
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds %"struct.Eigen::half", ptr %3, i32 0, i32 0
  %6 = getelementptr inbounds %"struct.Eigen::half_impl::half_base", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds %"struct.Eigen::half_impl::__half_raw", ptr %6, i32 0, i32 0
  store i16 %0, ptr %7, align 2
  store ptr %1, ptr %4, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = load double, ptr %8, align 8
  %10 = fadd double %9, 2.000000e+00
  %11 = load ptr, ptr %4, align 8
  store double %10, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z12measure_timeeRd(x86_fp80 noundef %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 {
  %3 = alloca x86_fp80, align 16
  %4 = alloca ptr, align 8
  store x86_fp80 %0, ptr %3, align 16
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load double, ptr %5, align 8
  %7 = fadd double %6, 1.600000e+01
  %8 = load ptr, ptr %4, align 8
  store double %7, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #6 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca double, align 8
  %7 = alloca float, align 4
  %8 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %9 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %10 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %11 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %12 = alloca double, align 8
  %13 = alloca double, align 8
  %14 = alloca i8, align 1
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %18 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %19 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %20 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store i32 0, ptr %1, align 4
  store double 1.000000e+01, ptr %2, align 8
  store double 0.000000e+00, ptr %6, align 8
  %21 = call i64 @clock() #5
  store i64 %21, ptr %4, align 8
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %8)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %9)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %10)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %11)
  call void @_Z5fun_aIfN9ml_dtypes15float8_internal13float8_ieee_pILi3EEES3_S3_S3_EvRT_RT0_RT1_RT2_RT3_(ptr noundef nonnull align 4 dereferenceable(4) %7, ptr noundef nonnull align 1 dereferenceable(1) %8, ptr noundef nonnull align 1 dereferenceable(1) %9, ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef nonnull align 1 dereferenceable(1) %11)
  store double 0x40172EDFFCFEC7AB, ptr %12, align 8
  %22 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %23 = load double, ptr %12, align 8
  %24 = fsub double %22, %23
  %25 = call double @llvm.fabs.f64(double %24)
  %26 = load double, ptr %12, align 8
  %27 = fdiv double %25, %26
  store double %27, ptr %13, align 8
  %28 = load double, ptr %13, align 8
  %29 = load double, ptr %2, align 8
  %30 = fcmp ole double %28, %29
  br i1 %30, label %31, label %40

31:                                               ; preds = %0
  store i8 1, ptr %14, align 1
  %32 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str)
  %33 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str.6)
  %34 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZN9ml_dtypes15float8_internallsINS0_13float8_ieee_pILi3EEEEERSoS4_RKNS0_11float8_baseIT_EE(ptr noundef nonnull align 8 dereferenceable(8) %33, ptr noundef nonnull align 1 dereferenceable(1) %11)
  %35 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEPFRSoS_E(ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %36 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str.7)
  %37 = load double, ptr %13, align 8
  %38 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8) %36, double noundef %37)
  %39 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEPFRSoS_E(ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %49

40:                                               ; preds = %0
  store i8 0, ptr %14, align 1
  %41 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str.8)
  %42 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str.6)
  %43 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZN9ml_dtypes15float8_internallsINS0_13float8_ieee_pILi3EEEEERSoS4_RKNS0_11float8_baseIT_EE(ptr noundef nonnull align 8 dereferenceable(8) %42, ptr noundef nonnull align 1 dereferenceable(1) %11)
  %44 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEPFRSoS_E(ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %45 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef @.str.7)
  %46 = load double, ptr %13, align 8
  %47 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8) %45, double noundef %46)
  %48 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEPFRSoS_E(ptr noundef nonnull align 8 dereferenceable(8) %47, ptr noundef @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %49

49:                                               ; preds = %40, %31
  %50 = call ptr @fopen(ptr noundef @.str.9, ptr noundef @.str.10)
  store ptr %50, ptr %15, align 8
  %51 = load i8, ptr %14, align 1
  %52 = trunc i8 %51 to i1
  %53 = zext i1 %52 to i64
  %54 = select i1 %52, ptr @.str.11, ptr @.str.12
  %55 = load ptr, ptr %15, align 8
  %56 = call i32 @fputs(ptr noundef %54, ptr noundef %55)
  %57 = load ptr, ptr %15, align 8
  %58 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %59 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %57, ptr noundef @.str.13, double noundef %58)
  %60 = load ptr, ptr %15, align 8
  %61 = load double, ptr %13, align 8
  %62 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %60, ptr noundef @.str.13, double noundef %61)
  %63 = call i64 @clock() #5
  store i64 %63, ptr %5, align 8
  %64 = call ptr @fopen(ptr noundef @.str.14, ptr noundef @.str.10)
  store ptr %64, ptr %16, align 8
  %65 = load float, ptr %7, align 4
  call void @_Z12measure_timefRd(float noundef %65, ptr noundef nonnull align 8 dereferenceable(8) %6)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %17, ptr align 1 %8, i64 1, i1 false)
  %66 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %17, i32 0, i32 0
  %67 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %66, i32 0, i32 0
  %68 = load i8, ptr %67, align 1
  call void @_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd(i8 %68, ptr noundef nonnull align 8 dereferenceable(8) %6)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %18, ptr align 1 %9, i64 1, i1 false)
  %69 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %18, i32 0, i32 0
  %70 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %69, i32 0, i32 0
  %71 = load i8, ptr %70, align 1
  call void @_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd(i8 %71, ptr noundef nonnull align 8 dereferenceable(8) %6)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %19, ptr align 1 %10, i64 1, i1 false)
  %72 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %19, i32 0, i32 0
  %73 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %72, i32 0, i32 0
  %74 = load i8, ptr %73, align 1
  call void @_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd(i8 %74, ptr noundef nonnull align 8 dereferenceable(8) %6)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %20, ptr align 1 %11, i64 1, i1 false)
  %75 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %20, i32 0, i32 0
  %76 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %75, i32 0, i32 0
  %77 = load i8, ptr %76, align 1
  call void @_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd(i8 %77, ptr noundef nonnull align 8 dereferenceable(8) %6)
  %78 = load ptr, ptr %16, align 8
  %79 = load double, ptr %6, align 8
  %80 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %78, ptr noundef @.str.15, double noundef %79)
  ret i32 0
}

; Function Attrs: nounwind
declare i64 @clock() #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #2 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %3)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z5fun_aIfN9ml_dtypes15float8_internal13float8_ieee_pILi3EEES3_S3_S3_EvRT_RT0_RT1_RT2_RT3_(ptr noundef nonnull align 4 dereferenceable(4) %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2, ptr noundef nonnull align 1 dereferenceable(1) %3, ptr noundef nonnull align 1 dereferenceable(1) %4) #2 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %17 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %18 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %19 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %20 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %21 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %22 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %23 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %24 = alloca float, align 4
  %25 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %26 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %27 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %28 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %29 = alloca float, align 4
  %30 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %31 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %32 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %33 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %34 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %35 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %7, align 8
  store ptr %2, ptr %8, align 8
  store ptr %3, ptr %9, align 8
  store ptr %4, ptr %10, align 8
  store i32 1000000, ptr %15, align 4
  store i32 0, ptr %11, align 4
  br label %36

36:                                               ; preds = %108, %5
  %37 = load i32, ptr %11, align 4
  %38 = icmp slt i32 %37, 1
  br i1 %38, label %39, label %111

39:                                               ; preds = %36
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %16, double noundef -1.000000e+00)
  %40 = load ptr, ptr %7, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %40, ptr align 1 %16, i64 1, i1 false)
  %41 = load ptr, ptr %7, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %18, ptr align 1 %41, i64 1, i1 false)
  %42 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %18, i32 0, i32 0
  %43 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %42, i32 0, i32 0
  %44 = load i8, ptr %43, align 1
  %45 = call i8 @_Z4acosIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_(i8 %44)
  %46 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %17, i32 0, i32 0
  %47 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %46, i32 0, i32 0
  store i8 %45, ptr %47, align 1
  %48 = load ptr, ptr %9, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %48, ptr align 1 %17, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %19, double noundef 0.000000e+00)
  %49 = load ptr, ptr %10, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %49, ptr align 1 %19, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %20, double noundef 0.000000e+00)
  %50 = load ptr, ptr %7, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %50, ptr align 1 %20, i64 1, i1 false)
  %51 = load ptr, ptr %9, align 8
  %52 = load i32, ptr %15, align 4
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEIiSt9enable_ifILb1EvEEET_(ptr noundef nonnull align 1 dereferenceable(1) %22, i32 noundef %52)
  %53 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEdvERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %51, ptr noundef nonnull align 1 dereferenceable(1) %22)
  %54 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %21, i32 0, i32 0
  %55 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %54, i32 0, i32 0
  store i8 %53, ptr %55, align 1
  %56 = call noundef float @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvfEv(ptr noundef nonnull align 1 dereferenceable(1) %21)
  %57 = load ptr, ptr %6, align 8
  store float %56, ptr %57, align 4
  store i32 1, ptr %12, align 4
  br label %58

58:                                               ; preds = %104, %39
  %59 = load i32, ptr %12, align 4
  %60 = load i32, ptr %15, align 4
  %61 = icmp sle i32 %59, %60
  br i1 %61, label %62, label %107

62:                                               ; preds = %58
  %63 = load i32, ptr %12, align 4
  %64 = sitofp i32 %63 to float
  %65 = load ptr, ptr %6, align 8
  %66 = load float, ptr %65, align 4
  %67 = fmul float %64, %66
  %68 = call noundef float @_Z3funIfET_S0_(float noundef %67)
  store float %68, ptr %24, align 4
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2ERKf(ptr noundef nonnull align 1 dereferenceable(1) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
  %69 = load ptr, ptr %8, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %69, ptr align 1 %23, i64 1, i1 false)
  %70 = load ptr, ptr %10, align 8
  %71 = load ptr, ptr %6, align 8
  %72 = load float, ptr %71, align 4
  %73 = load ptr, ptr %6, align 8
  %74 = load float, ptr %73, align 4
  %75 = fmul float %72, %74
  store float %75, ptr %29, align 4
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2ERKf(ptr noundef nonnull align 1 dereferenceable(1) %28, ptr noundef nonnull align 4 dereferenceable(4) %29)
  %76 = load ptr, ptr %8, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %32, ptr align 1 %76, i64 1, i1 false)
  %77 = load ptr, ptr %7, align 8
  %78 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEmiERKS2_(ptr noundef nonnull align 1 dereferenceable(1) %32, ptr noundef nonnull align 1 dereferenceable(1) %77)
  %79 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %31, i32 0, i32 0
  %80 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %79, i32 0, i32 0
  store i8 %78, ptr %80, align 1
  %81 = load ptr, ptr %8, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %34, ptr align 1 %81, i64 1, i1 false)
  %82 = load ptr, ptr %7, align 8
  %83 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEmiERKS2_(ptr noundef nonnull align 1 dereferenceable(1) %34, ptr noundef nonnull align 1 dereferenceable(1) %82)
  %84 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %33, i32 0, i32 0
  %85 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %84, i32 0, i32 0
  store i8 %83, ptr %85, align 1
  %86 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmlERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %31, ptr noundef nonnull align 1 dereferenceable(1) %33)
  %87 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %30, i32 0, i32 0
  %88 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %87, i32 0, i32 0
  store i8 %86, ptr %88, align 1
  %89 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEplERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %28, ptr noundef nonnull align 1 dereferenceable(1) %30)
  %90 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %27, i32 0, i32 0
  %91 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %90, i32 0, i32 0
  store i8 %89, ptr %91, align 1
  %92 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %27, i32 0, i32 0
  %93 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %92, i32 0, i32 0
  %94 = load i8, ptr %93, align 1
  %95 = call i8 @_Z4sqrtIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_(i8 %94)
  %96 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %26, i32 0, i32 0
  %97 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %96, i32 0, i32 0
  store i8 %95, ptr %97, align 1
  %98 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEplERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %70, ptr noundef nonnull align 1 dereferenceable(1) %26)
  %99 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %25, i32 0, i32 0
  %100 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %99, i32 0, i32 0
  store i8 %98, ptr %100, align 1
  %101 = load ptr, ptr %10, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %101, ptr align 1 %25, i64 1, i1 false)
  %102 = load ptr, ptr %8, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %35, ptr align 1 %102, i64 1, i1 false)
  %103 = load ptr, ptr %7, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %103, ptr align 1 %35, i64 1, i1 false)
  br label %104

104:                                              ; preds = %62
  %105 = load i32, ptr %12, align 4
  %106 = add nsw i32 %105, 1
  store i32 %106, ptr %12, align 4
  br label %58, !llvm.loop !6

107:                                              ; preds = %58
  br label %108

108:                                              ; preds = %107
  %109 = load i32, ptr %11, align 4
  %110 = add nsw i32 %109, 1
  store i32 %110, ptr %11, align 4
  br label %36, !llvm.loop !8

111:                                              ; preds = %36
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %5 = call noundef double @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIdLb0ELb0EEET_RKS3_(ptr noundef nonnull align 1 dereferenceable(1) %4)
  ret double %5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #7

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) #4

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZN9ml_dtypes15float8_internallsINS0_13float8_ieee_pILi3EEEEERSoS4_RKNS0_11float8_baseIT_EE(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = call noundef float @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvfEv(ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8) %5, float noundef %8)
  %10 = load ptr, ptr %3, align 8
  ret ptr %10
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEPFRSoS_E(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8)) #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8), double noundef) #4

declare ptr @fopen(ptr noundef, ptr noundef) #4

declare i32 @fputs(ptr noundef, ptr noundef) #4

declare i32 @fprintf(ptr noundef, ptr noundef, ...) #4

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_Z12measure_timeILi3EEvN9ml_dtypes15float8_internal13float8_ieee_pIXT_EEERd(i8 %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #1 comdat {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  store i8 %0, ptr %6, align 1
  store ptr %1, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load double, ptr %7, align 8
  %9 = fadd double %8, 1.000000e+00
  %10 = load ptr, ptr %4, align 8
  store double %9, ptr %10, align 8
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #8

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 0, ptr %4, align 1
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @_ZNK5Eigen8symbolic8BaseExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN5Eigen8symbolic9ValueExprINS_8internal8FixedIntILi1EEEEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN5Eigen8symbolic7AddExprINS0_10SymbolExprINS_8internal17symbolic_last_tagEEENS0_9ValueExprINS3_8FixedIntILi1EEEEEEC2ERKS5_RKS9_(ptr noundef nonnull align 1 dereferenceable(2) %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #1 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds %"class.Eigen::symbolic::AddExpr", ptr %7, i32 0, i32 0
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds %"class.Eigen::symbolic::AddExpr", ptr %7, i32 0, i32 1
  %11 = load ptr, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt24uniform_int_distributionIiE10param_typeC2Eii(ptr noundef nonnull align 4 dereferenceable(8) %0, i32 noundef %1, i32 noundef %2) unnamed_addr #1 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds %"struct.std::uniform_int_distribution<>::param_type", ptr %7, i32 0, i32 0
  %9 = load i32, ptr %5, align 4
  store i32 %9, ptr %8, align 4
  %10 = getelementptr inbounds %"struct.std::uniform_int_distribution<>::param_type", ptr %7, i32 0, i32 1
  %11 = load i32, ptr %6, align 4
  store i32 %11, ptr %10, align 4
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(ptr noundef nonnull align 8 dereferenceable(5000) %0, i64 noundef %1) #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = load i64, ptr %4, align 8
  %9 = call noundef i64 @_ZNSt8__detail5__modImTnT_Lm4294967296ETnS1_Lm1ETnS1_Lm0EEES1_S1_(i64 noundef %8)
  %10 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %7, i32 0, i32 0
  %11 = getelementptr inbounds [624 x i64], ptr %10, i64 0, i64 0
  store i64 %9, ptr %11, align 8
  store i64 1, ptr %5, align 8
  br label %12

12:                                               ; preds = %36, %2
  %13 = load i64, ptr %5, align 8
  %14 = icmp ult i64 %13, 624
  br i1 %14, label %15, label %39

15:                                               ; preds = %12
  %16 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %7, i32 0, i32 0
  %17 = load i64, ptr %5, align 8
  %18 = sub i64 %17, 1
  %19 = getelementptr inbounds [624 x i64], ptr %16, i64 0, i64 %18
  %20 = load i64, ptr %19, align 8
  store i64 %20, ptr %6, align 8
  %21 = load i64, ptr %6, align 8
  %22 = lshr i64 %21, 30
  %23 = load i64, ptr %6, align 8
  %24 = xor i64 %23, %22
  store i64 %24, ptr %6, align 8
  %25 = load i64, ptr %6, align 8
  %26 = mul i64 %25, 1812433253
  store i64 %26, ptr %6, align 8
  %27 = load i64, ptr %5, align 8
  %28 = call noundef i64 @_ZNSt8__detail5__modImTnT_Lm624ETnS1_Lm1ETnS1_Lm0EEES1_S1_(i64 noundef %27)
  %29 = load i64, ptr %6, align 8
  %30 = add i64 %29, %28
  store i64 %30, ptr %6, align 8
  %31 = load i64, ptr %6, align 8
  %32 = call noundef i64 @_ZNSt8__detail5__modImTnT_Lm4294967296ETnS1_Lm1ETnS1_Lm0EEES1_S1_(i64 noundef %31)
  %33 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %7, i32 0, i32 0
  %34 = load i64, ptr %5, align 8
  %35 = getelementptr inbounds [624 x i64], ptr %33, i64 0, i64 %34
  store i64 %32, ptr %35, align 8
  br label %36

36:                                               ; preds = %15
  %37 = load i64, ptr %5, align 8
  %38 = add i64 %37, 1
  store i64 %38, ptr %5, align 8
  br label %12, !llvm.loop !9

39:                                               ; preds = %12
  %40 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %7, i32 0, i32 1
  store i64 624, ptr %40, align 8
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt8__detail5__modImTnT_Lm4294967296ETnS1_Lm1ETnS1_Lm0EEES1_S1_(i64 noundef %0) #2 comdat {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  store i64 1, ptr %3, align 8
  %4 = load i64, ptr %2, align 8
  %5 = call noundef i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64 noundef %4)
  ret i64 %5
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt8__detail5__modImTnT_Lm624ETnS1_Lm1ETnS1_Lm0EEES1_S1_(i64 noundef %0) #2 comdat {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  store i64 1, ptr %3, align 8
  %4 = load i64, ptr %2, align 8
  %5 = call noundef i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64 noundef %4)
  ret i64 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64 noundef %0) #1 comdat align 2 {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %4 = load i64, ptr %2, align 8
  %5 = mul i64 1, %4
  %6 = add i64 %5, 0
  store i64 %6, ptr %3, align 8
  %7 = load i64, ptr %3, align 8
  %8 = urem i64 %7, 4294967296
  store i64 %8, ptr %3, align 8
  %9 = load i64, ptr %3, align 8
  ret i64 %9
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64 noundef %0) #1 comdat align 2 {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %4 = load i64, ptr %2, align 8
  %5 = mul i64 1, %4
  %6 = add i64 %5, 0
  store i64 %6, ptr %3, align 8
  %7 = load i64, ptr %3, align 8
  %8 = urem i64 %7, 624
  store i64 %8, ptr %3, align 8
  %9 = load i64, ptr %3, align 8
  ret i64 %9
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %0, double noundef %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store ptr %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load double, ptr %4, align 8
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ed(ptr noundef nonnull align 1 dereferenceable(1) %5, double noundef %6)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_Z4acosIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_(i8 %0) #2 comdat {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %4, i32 0, i32 0
  store i8 %0, ptr %5, align 1
  %6 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %7 = call double @acos(double noundef %6) #5
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %2, double noundef %7)
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %8, i32 0, i32 0
  %10 = load i8, ptr %9, align 1
  ret i8 %10
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEdvERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = load ptr, ptr %5, align 8
  %10 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %9)
  %11 = fdiv double %8, %10
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %3, double noundef %11)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  %14 = load i8, ptr %13, align 1
  ret i8 %14
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEIiSt9enable_ifILb1EvEEET_(ptr noundef nonnull align 1 dereferenceable(1) %0, i32 noundef %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load ptr, ptr %3, align 8
  %6 = load i32, ptr %4, align 4
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2IiSt9enable_ifILb1EvEEET_(ptr noundef nonnull align 1 dereferenceable(1) %5, i32 noundef %6)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef float @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvfEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %5 = call noundef float @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIfLb0ELb0EEET_RKS3_(ptr noundef nonnull align 1 dereferenceable(1) %4)
  ret float %5
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef float @_Z3funIfET_S0_(float noundef %0) #2 comdat {
  %2 = alloca float, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  store float %0, ptr %2, align 4
  store i32 5, ptr %4, align 4
  store float 1.000000e+00, ptr %5, align 4
  store float 1.000000e+00, ptr %6, align 4
  %7 = load float, ptr %2, align 4
  store float %7, ptr %5, align 4
  store i32 1, ptr %3, align 4
  br label %8

8:                                                ; preds = %23, %1
  %9 = load i32, ptr %3, align 4
  %10 = load i32, ptr %4, align 4
  %11 = icmp sle i32 %9, %10
  br i1 %11, label %12, label %26

12:                                               ; preds = %8
  %13 = load float, ptr %6, align 4
  %14 = fmul float 2.000000e+00, %13
  store float %14, ptr %6, align 4
  %15 = load float, ptr %5, align 4
  %16 = load float, ptr %6, align 4
  %17 = load float, ptr %2, align 4
  %18 = fmul float %16, %17
  %19 = call noundef float @_ZSt3sinf(float noundef %18)
  %20 = load float, ptr %6, align 4
  %21 = fdiv float %19, %20
  %22 = fadd float %15, %21
  store float %22, ptr %5, align 4
  br label %23

23:                                               ; preds = %12
  %24 = load i32, ptr %3, align 4
  %25 = add nsw i32 %24, 1
  store i32 %25, ptr %3, align 4
  br label %8, !llvm.loop !10

26:                                               ; preds = %8
  %27 = load float, ptr %5, align 4
  ret float %27
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2ERKf(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 4 dereferenceable(4) %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EfEES3_RKT1_(ptr noundef nonnull align 4 dereferenceable(4) %7)
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %5, i32 0, i32 0
  %10 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %9, i32 0, i32 0
  store i8 %8, ptr %10, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %6, ptr align 1 %5, i64 1, i1 false)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEplERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = load ptr, ptr %5, align 8
  %10 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %9)
  %11 = fadd double %8, %10
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %3, double noundef %11)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  %14 = load i8, ptr %13, align 1
  ret i8 %14
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_Z4sqrtIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEET_S4_(i8 %0) #2 comdat {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %4, i32 0, i32 0
  store i8 %0, ptr %5, align 1
  %6 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %7 = call double @sqrt(double noundef %6) #5
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %2, double noundef %7)
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %8, i32 0, i32 0
  %10 = load i8, ptr %9, align 1
  ret i8 %10
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEmiERKS2_(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmiERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %6, ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %10 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %9, i32 0, i32 0
  store i8 %8, ptr %10, align 1
  %11 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %11, i32 0, i32 0
  %13 = load i8, ptr %12, align 1
  ret i8 %13
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmlERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = load ptr, ptr %5, align 8
  %10 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %9)
  %11 = fmul double %8, %10
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %3, double noundef %11)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  %14 = load i8, ptr %13, align 1
  ret i8 %14
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2Ed(ptr noundef nonnull align 1 dereferenceable(1) %0, double noundef %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %6 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  store ptr %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EEES3_RKd(ptr noundef nonnull align 8 dereferenceable(8) %4)
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %5, i32 0, i32 0
  %10 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %9, i32 0, i32 0
  store i8 %8, ptr %10, align 1
  %11 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2EhNS4_19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %7, i8 noundef zeroext %11)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EEES3_RKd(ptr noundef nonnull align 8 dereferenceable(8) %0) #2 comdat align 2 {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call i8 @_ZN9ml_dtypes15float8_internal11ConvertImplIdNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKdb(ptr noundef nonnull align 8 dereferenceable(8) %4, i1 noundef zeroext false)
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %7 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %6, i32 0, i32 0
  store i8 %5, ptr %7, align 1
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %8, i32 0, i32 0
  %10 = load i8, ptr %9, align 1
  ret i8 %10
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  %5 = load i8, ptr %4, align 1
  ret i8 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2EhNS4_19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %0, i8 noundef zeroext %1) unnamed_addr #1 comdat align 2 {
  %3 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  %4 = alloca ptr, align 8
  %5 = alloca i8, align 1
  store ptr %0, ptr %4, align 8
  store i8 %1, ptr %5, align 1
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %6, i32 0, i32 0
  %8 = load i8, ptr %5, align 1
  store i8 %8, ptr %7, align 1
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal11ConvertImplIdNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKdb(ptr noundef nonnull align 8 dereferenceable(8) %0, i1 noundef zeroext %1) #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %5 = alloca ptr, align 8
  %6 = alloca i8, align 1
  %7 = alloca i8, align 1
  %8 = alloca i64, align 8
  %9 = alloca double, align 8
  %10 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %11 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  %16 = alloca i32, align 4
  %17 = alloca i64, align 8
  %18 = alloca i8, align 1
  %19 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %20 = alloca i64, align 8
  %21 = alloca i8, align 1
  %22 = alloca i64, align 8
  %23 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %24 = alloca i64, align 8
  %25 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %26 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %5, align 8
  %27 = zext i1 %1 to i8
  store i8 %27, ptr %6, align 1
  %28 = load ptr, ptr %5, align 8
  %29 = call noundef i64 @_ZN5Eigen6numext8bit_castImdEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %28)
  %30 = lshr i64 %29, 63
  %31 = icmp ne i64 %30, 0
  %32 = zext i1 %31 to i8
  store i8 %32, ptr %7, align 1
  %33 = load ptr, ptr %5, align 8
  store ptr %33, ptr %3, align 8
  %34 = load ptr, ptr %3, align 8
  %35 = load double, ptr %34, align 8
  %36 = call noundef double @_ZSt3absd(double noundef %35)
  store double %36, ptr %9, align 8
  %37 = call noundef i64 @_ZN5Eigen6numext8bit_castImdEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %9)
  store i64 %37, ptr %8, align 8
  %38 = load ptr, ptr %5, align 8
  %39 = call noundef zeroext i1 @_ZN5Eigen6numext5isinfIdEEbRKT_(ptr noundef nonnull align 8 dereferenceable(8) %38)
  br i1 %39, label %40, label %55

40:                                               ; preds = %2
  %41 = load i8, ptr %7, align 1
  %42 = trunc i8 %41 to i1
  br i1 %42, label %43, label %50

43:                                               ; preds = %40
  %44 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %45 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %10, i32 0, i32 0
  %46 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %45, i32 0, i32 0
  store i8 %44, ptr %46, align 1
  %47 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %10)
  %48 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %49 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %48, i32 0, i32 0
  store i8 %47, ptr %49, align 1
  br label %54

50:                                               ; preds = %40
  %51 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %52 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %53 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %52, i32 0, i32 0
  store i8 %51, ptr %53, align 1
  br label %54

54:                                               ; preds = %50, %43
  br label %183

55:                                               ; preds = %2
  %56 = load ptr, ptr %5, align 8
  %57 = call noundef zeroext i1 @_ZN5Eigen6numext5isnanIdEEbRKT_(ptr noundef nonnull align 8 dereferenceable(8) %56)
  br i1 %57, label %58, label %62

58:                                               ; preds = %55
  %59 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE9quiet_NaNEv()
  %60 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %61 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %60, i32 0, i32 0
  store i8 %59, ptr %61, align 1
  br label %183

62:                                               ; preds = %55
  %63 = load i64, ptr %8, align 8
  %64 = icmp eq i64 %63, 0
  br i1 %64, label %65, label %74

65:                                               ; preds = %62
  %66 = load i8, ptr %7, align 1
  %67 = trunc i8 %66 to i1
  br i1 %67, label %68, label %72

68:                                               ; preds = %65
  call void @llvm.memset.p0.i64(ptr align 1 %11, i8 0, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %69 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %70 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %71 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %70, i32 0, i32 0
  store i8 %69, ptr %71, align 1
  br label %73

72:                                               ; preds = %65
  call void @llvm.memset.p0.i64(ptr align 1 %4, i8 0, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %4)
  br label %73

73:                                               ; preds = %72, %68
  br label %183

74:                                               ; preds = %62
  %75 = load i64, ptr %8, align 8
  %76 = lshr i64 %75, 52
  %77 = trunc i64 %76 to i32
  store i32 %77, ptr %12, align 4
  %78 = load i32, ptr %12, align 4
  %79 = sub nsw i32 %78, 1023
  store i32 %79, ptr %13, align 4
  %80 = load i32, ptr %13, align 4
  %81 = add nsw i32 %80, 16
  store i32 %81, ptr %14, align 4
  %82 = load i32, ptr %14, align 4
  %83 = icmp sle i32 %82, 0
  br i1 %83, label %84, label %132

84:                                               ; preds = %74
  %85 = load i32, ptr %12, align 4
  %86 = icmp sgt i32 %85, 0
  %87 = zext i1 %86 to i64
  %88 = select i1 %86, i32 1, i32 0
  %89 = sext i32 %88 to i64
  store i64 %89, ptr %15, align 8
  %90 = load i32, ptr %14, align 4
  %91 = sub nsw i32 50, %90
  %92 = sext i32 %91 to i64
  %93 = load i64, ptr %15, align 8
  %94 = add i64 %92, %93
  %95 = trunc i64 %94 to i32
  store i32 %95, ptr %16, align 4
  %96 = load i64, ptr %8, align 8
  %97 = and i64 %96, 4503599627370495
  %98 = load i64, ptr %15, align 8
  %99 = shl i64 %98, 52
  %100 = or i64 %97, %99
  store i64 %100, ptr %17, align 8
  store i8 0, ptr %18, align 1
  %101 = load i32, ptr %16, align 4
  %102 = icmp sle i32 %101, 53
  br i1 %102, label %103, label %120

103:                                              ; preds = %84
  %104 = load i8, ptr %6, align 1
  %105 = trunc i8 %104 to i1
  br i1 %105, label %106, label %110

106:                                              ; preds = %103
  %107 = load i64, ptr %17, align 8
  %108 = load i32, ptr %16, align 4
  %109 = call noundef i64 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundImEET_S2_i(i64 noundef %107, i32 noundef %108)
  store i64 %109, ptr %17, align 8
  br label %114

110:                                              ; preds = %103
  %111 = load i64, ptr %17, align 8
  %112 = load i32, ptr %16, align 4
  %113 = call noundef i64 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenImEET_S2_i(i64 noundef %111, i32 noundef %112)
  store i64 %113, ptr %17, align 8
  br label %114

114:                                              ; preds = %110, %106
  %115 = load i64, ptr %17, align 8
  %116 = load i32, ptr %16, align 4
  %117 = zext i32 %116 to i64
  %118 = lshr i64 %115, %117
  %119 = trunc i64 %118 to i8
  store i8 %119, ptr %18, align 1
  br label %120

120:                                              ; preds = %114, %84
  %121 = call i8 @_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %18)
  %122 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %19, i32 0, i32 0
  %123 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %122, i32 0, i32 0
  store i8 %121, ptr %123, align 1
  %124 = load i8, ptr %7, align 1
  %125 = trunc i8 %124 to i1
  br i1 %125, label %126, label %130

126:                                              ; preds = %120
  %127 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %19)
  %128 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %129 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %128, i32 0, i32 0
  store i8 %127, ptr %129, align 1
  br label %131

130:                                              ; preds = %120
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %19, i64 1, i1 false)
  br label %131

131:                                              ; preds = %130, %126
  br label %183

132:                                              ; preds = %74
  %133 = load i64, ptr %8, align 8
  store i64 %133, ptr %20, align 8
  %134 = load i8, ptr %6, align 1
  %135 = trunc i8 %134 to i1
  br i1 %135, label %136, label %139

136:                                              ; preds = %132
  %137 = load i64, ptr %8, align 8
  %138 = call noundef i64 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundImEET_S2_i(i64 noundef %137, i32 noundef 50)
  store i64 %138, ptr %20, align 8
  br label %142

139:                                              ; preds = %132
  %140 = load i64, ptr %8, align 8
  %141 = call noundef i64 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenImEET_S2_i(i64 noundef %140, i32 noundef 50)
  store i64 %141, ptr %20, align 8
  br label %142

142:                                              ; preds = %139, %136
  %143 = load i64, ptr %20, align 8
  %144 = and i64 %143, -1125899906842624
  store i64 %144, ptr %20, align 8
  %145 = load i64, ptr %20, align 8
  %146 = add i64 %145, -4535124824762089472
  store i64 %146, ptr %20, align 8
  %147 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv()
  %148 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %23, i32 0, i32 0
  %149 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %148, i32 0, i32 0
  store i8 %147, ptr %149, align 1
  %150 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %23)
  %151 = zext i8 %150 to i64
  store i64 %151, ptr %22, align 8
  %152 = load i64, ptr %22, align 8
  store i64 %152, ptr %24, align 8
  %153 = load i64, ptr %24, align 8
  %154 = shl i64 %153, 50
  store i64 %154, ptr %24, align 8
  %155 = load i64, ptr %20, align 8
  %156 = lshr i64 %155, 50
  %157 = trunc i64 %156 to i8
  store i8 %157, ptr %21, align 1
  %158 = call i8 @_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %21)
  %159 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %25, i32 0, i32 0
  %160 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %159, i32 0, i32 0
  store i8 %158, ptr %160, align 1
  %161 = load i64, ptr %20, align 8
  %162 = load i64, ptr %24, align 8
  %163 = icmp ugt i64 %161, %162
  br i1 %163, label %164, label %174

164:                                              ; preds = %142
  br i1 false, label %165, label %169

165:                                              ; preds = %164
  %166 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv()
  %167 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %26, i32 0, i32 0
  %168 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %167, i32 0, i32 0
  store i8 %166, ptr %168, align 1
  br label %173

169:                                              ; preds = %164
  %170 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %171 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %26, i32 0, i32 0
  %172 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %171, i32 0, i32 0
  store i8 %170, ptr %172, align 1
  br label %173

173:                                              ; preds = %169, %165
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %25, ptr align 1 %26, i64 1, i1 false)
  br label %174

174:                                              ; preds = %173, %142
  %175 = load i8, ptr %7, align 1
  %176 = trunc i8 %175 to i1
  br i1 %176, label %177, label %181

177:                                              ; preds = %174
  %178 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %25)
  %179 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %180 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %179, i32 0, i32 0
  store i8 %178, ptr %180, align 1
  br label %182

181:                                              ; preds = %174
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %25, i64 1, i1 false)
  br label %182

182:                                              ; preds = %181, %177
  br label %183

183:                                              ; preds = %182, %131, %73, %58, %54
  %184 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %185 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %184, i32 0, i32 0
  %186 = load i8, ptr %185, align 1
  ret i8 %186
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZN5Eigen6numext8bit_castImdEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = load double, ptr %5, align 8
  store double %6, ptr %4, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %4, i64 8, i1 false)
  %7 = load i64, ptr %3, align 8
  ret i64 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isinfIdEEbRKT_(ptr noundef nonnull align 8 dereferenceable(8) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIdEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 8 dereferenceable(8) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE8infinityEv()
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %4)
  %6 = zext i8 %5 to i32
  %7 = and i32 %6, 127
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %10

9:                                                ; preds = %1
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %2, ptr align 1 %4, i64 1, i1 false)
  br label %14

10:                                               ; preds = %1
  %11 = call i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEngEv(ptr noundef nonnull align 1 dereferenceable(1) %4)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  store i8 %11, ptr %13, align 1
  br label %14

14:                                               ; preds = %10, %9
  %15 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %16 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %15, i32 0, i32 0
  %17 = load i8, ptr %16, align 1
  ret i8 %17
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isnanIdEEbRKT_(ptr noundef nonnull align 8 dereferenceable(8) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIdEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 8 dereferenceable(8) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE9quiet_NaNEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE9quiet_NaNEv()
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #9

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundImEET_S2_i(i64 noundef %0, i32 noundef %1) #2 comdat {
  %3 = alloca i64, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store i64 %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  store i32 1, ptr %5, align 4
  %10 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_(ptr noundef nonnull align 4 dereferenceable(8) @_ZL12distribution, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL2mt)
  store i32 %10, ptr %6, align 4
  %11 = load i32, ptr %5, align 4
  %12 = zext i32 %11 to i64
  %13 = shl i64 1, %12
  %14 = sub i64 %13, 1
  store i64 %14, ptr %7, align 8
  %15 = load i32, ptr %6, align 4
  %16 = sext i32 %15 to i64
  %17 = load i64, ptr %7, align 8
  %18 = and i64 %16, %17
  store i64 %18, ptr %8, align 8
  %19 = load i64, ptr %3, align 8
  %20 = load i64, ptr %8, align 8
  %21 = load i32, ptr %4, align 4
  %22 = load i32, ptr %5, align 4
  %23 = sub nsw i32 %21, %22
  %24 = zext i32 %23 to i64
  %25 = shl i64 %20, %24
  %26 = add i64 %19, %25
  store i64 %26, ptr %9, align 8
  %27 = load i64, ptr %9, align 8
  ret i64 %27
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenImEET_S2_i(i64 noundef %0, i32 noundef %1) #1 comdat {
  %3 = alloca i64, align 8
  %4 = alloca i32, align 4
  %5 = alloca i64, align 8
  store i64 %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %6 = load i32, ptr %4, align 4
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %2
  br label %21

9:                                                ; preds = %2
  %10 = load i64, ptr %3, align 8
  %11 = load i32, ptr %4, align 4
  %12 = zext i32 %11 to i64
  %13 = lshr i64 %10, %12
  %14 = and i64 %13, 1
  %15 = load i32, ptr %4, align 4
  %16 = sub nsw i32 %15, 1
  %17 = zext i32 %16 to i64
  %18 = shl i64 1, %17
  %19 = add i64 %14, %18
  %20 = sub i64 %19, 1
  br label %21

21:                                               ; preds = %9, %8
  %22 = phi i64 [ 0, %8 ], [ %20, %9 ]
  store i64 %22, ptr %5, align 8
  %23 = load i64, ptr %3, align 8
  %24 = load i64, ptr %5, align 8
  %25 = add i64 %23, %24
  ret i64 %25
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  %4 = alloca i8, align 1
  store ptr %0, ptr %3, align 8
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2)
  %5 = load ptr, ptr %3, align 8
  %6 = load i8, ptr %5, align 1
  store i8 %6, ptr %4, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %2, ptr align 1 %4, i64 1, i1 false)
  %7 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %7, i32 0, i32 0
  %9 = load i8, ptr %8, align 1
  ret i8 %9
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i8, align 1
  %4 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %5, i64 1, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %4, i64 1, i1 false)
  %6 = load i8, ptr %3, align 1
  ret i8 %6
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE3maxEv()
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZSt3absd(double noundef %0) #1 comdat {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = call double @llvm.fabs.f64(double %3)
  ret double %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIdEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 8 dereferenceable(8) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load double, ptr %3, align 8
  %5 = call noundef zeroext i1 @_ZSt5isinfd(double noundef %4)
  ret i1 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZSt5isinfd(double noundef %0) #1 comdat {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = call i1 @llvm.is.fpclass.f64(double %3, i32 516)
  ret i1 %4
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg) #7

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE8infinityEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh(i8 noundef zeroext 127)
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh(i8 noundef zeroext %0) #2 comdat align 2 {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca i8, align 1
  %4 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  store i8 %0, ptr %3, align 1
  %5 = load i8, ptr %3, align 1
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEhNS3_IS2_E19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %2, i8 noundef zeroext %5)
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %7 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %6, i32 0, i32 0
  %8 = load i8, ptr %7, align 1
  ret i8 %8
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEhNS3_IS2_E19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %0, i8 noundef zeroext %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  %4 = alloca ptr, align 8
  %5 = alloca i8, align 1
  store ptr %0, ptr %4, align 8
  store i8 %1, ptr %5, align 1
  %6 = load ptr, ptr %4, align 8
  %7 = load i8, ptr %5, align 1
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2EhNS4_19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %6, i8 noundef zeroext %7)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEngEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  %4 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %5)
  %7 = zext i8 %6 to i32
  %8 = xor i32 %7, 128
  %9 = trunc i32 %8 to i8
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEhNS3_IS2_E19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %2, i8 noundef zeroext %9)
  %10 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %11 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %10, i32 0, i32 0
  %12 = load i8, ptr %11, align 1
  ret i8 %12
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIdEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 8 dereferenceable(8) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load double, ptr %3, align 8
  %5 = call noundef zeroext i1 @_ZSt5isnand(double noundef %4)
  ret i1 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZSt5isnand(double noundef %0) #1 comdat {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = call i1 @llvm.is.fpclass.f64(double %3, i32 3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE9quiet_NaNEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh(i8 noundef zeroext -128)
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1) #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds %"class.std::uniform_int_distribution", ptr %5, i32 0, i32 0
  %8 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(5000) %6, ptr noundef nonnull align 4 dereferenceable(8) %7)
  ret i32 %8
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) #2 comdat align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca %"struct.std::uniform_int_distribution<>::param_type", align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = load ptr, ptr %5, align 8
  %20 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv()
  store i64 %20, ptr %7, align 8
  %21 = load ptr, ptr %5, align 8
  %22 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv()
  store i64 %22, ptr %8, align 8
  %23 = load i64, ptr %8, align 8
  %24 = load i64, ptr %7, align 8
  %25 = sub i64 %23, %24
  store i64 %25, ptr %9, align 8
  %26 = load ptr, ptr %6, align 8
  %27 = call noundef i32 @_ZNKSt24uniform_int_distributionIiE10param_type1bEv(ptr noundef nonnull align 4 dereferenceable(8) %26)
  %28 = sext i32 %27 to i64
  %29 = load ptr, ptr %6, align 8
  %30 = call noundef i32 @_ZNKSt24uniform_int_distributionIiE10param_type1aEv(ptr noundef nonnull align 4 dereferenceable(8) %29)
  %31 = sext i32 %30 to i64
  %32 = sub i64 %28, %31
  store i64 %32, ptr %10, align 8
  %33 = load i64, ptr %9, align 8
  %34 = load i64, ptr %10, align 8
  %35 = icmp ugt i64 %33, %34
  br i1 %35, label %36, label %58

36:                                               ; preds = %3
  %37 = load i64, ptr %10, align 8
  %38 = add i64 %37, 1
  store i64 %38, ptr %12, align 8
  %39 = load i64, ptr %9, align 8
  %40 = load i64, ptr %12, align 8
  %41 = udiv i64 %39, %40
  store i64 %41, ptr %13, align 8
  %42 = load i64, ptr %12, align 8
  %43 = load i64, ptr %13, align 8
  %44 = mul i64 %42, %43
  store i64 %44, ptr %14, align 8
  br label %45

45:                                               ; preds = %50, %36
  %46 = load ptr, ptr %5, align 8
  %47 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %46)
  %48 = load i64, ptr %7, align 8
  %49 = sub i64 %47, %48
  store i64 %49, ptr %11, align 8
  br label %50

50:                                               ; preds = %45
  %51 = load i64, ptr %11, align 8
  %52 = load i64, ptr %14, align 8
  %53 = icmp uge i64 %51, %52
  br i1 %53, label %45, label %54, !llvm.loop !11

54:                                               ; preds = %50
  %55 = load i64, ptr %13, align 8
  %56 = load i64, ptr %11, align 8
  %57 = udiv i64 %56, %55
  store i64 %57, ptr %11, align 8
  br label %98

58:                                               ; preds = %3
  %59 = load i64, ptr %9, align 8
  %60 = load i64, ptr %10, align 8
  %61 = icmp ult i64 %59, %60
  br i1 %61, label %62, label %92

62:                                               ; preds = %58
  br label %63

63:                                               ; preds = %89, %62
  %64 = load i64, ptr %9, align 8
  %65 = add i64 %64, 1
  store i64 %65, ptr %16, align 8
  %66 = load i64, ptr %16, align 8
  %67 = load ptr, ptr %5, align 8
  %68 = load i64, ptr %10, align 8
  %69 = load i64, ptr %16, align 8
  %70 = udiv i64 %68, %69
  %71 = trunc i64 %70 to i32
  call void @_ZNSt24uniform_int_distributionIiE10param_typeC2Eii(ptr noundef nonnull align 4 dereferenceable(8) %17, i32 noundef 0, i32 noundef %71)
  %72 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(5000) %67, ptr noundef nonnull align 4 dereferenceable(8) %17)
  %73 = sext i32 %72 to i64
  %74 = mul i64 %66, %73
  store i64 %74, ptr %15, align 8
  %75 = load i64, ptr %15, align 8
  %76 = load ptr, ptr %5, align 8
  %77 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %76)
  %78 = load i64, ptr %7, align 8
  %79 = sub i64 %77, %78
  %80 = add i64 %75, %79
  store i64 %80, ptr %11, align 8
  br label %81

81:                                               ; preds = %63
  %82 = load i64, ptr %11, align 8
  %83 = load i64, ptr %10, align 8
  %84 = icmp ugt i64 %82, %83
  br i1 %84, label %89, label %85

85:                                               ; preds = %81
  %86 = load i64, ptr %11, align 8
  %87 = load i64, ptr %15, align 8
  %88 = icmp ult i64 %86, %87
  br label %89

89:                                               ; preds = %85, %81
  %90 = phi i1 [ true, %81 ], [ %88, %85 ]
  br i1 %90, label %63, label %91, !llvm.loop !12

91:                                               ; preds = %89
  br label %97

92:                                               ; preds = %58
  %93 = load ptr, ptr %5, align 8
  %94 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %93)
  %95 = load i64, ptr %7, align 8
  %96 = sub i64 %94, %95
  store i64 %96, ptr %11, align 8
  br label %97

97:                                               ; preds = %92, %91
  br label %98

98:                                               ; preds = %97, %54
  %99 = load i64, ptr %11, align 8
  %100 = load ptr, ptr %6, align 8
  %101 = call noundef i32 @_ZNKSt24uniform_int_distributionIiE10param_type1aEv(ptr noundef nonnull align 4 dereferenceable(8) %100)
  %102 = sext i32 %101 to i64
  %103 = add i64 %99, %102
  %104 = trunc i64 %103 to i32
  ret i32 %104
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv() #1 comdat align 2 {
  ret i64 0
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv() #1 comdat align 2 {
  ret i64 4294967295
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNKSt24uniform_int_distributionIiE10param_type1bEv(ptr noundef nonnull align 4 dereferenceable(8) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"struct.std::uniform_int_distribution<>::param_type", ptr %3, i32 0, i32 1
  %5 = load i32, ptr %4, align 4
  ret i32 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNKSt24uniform_int_distributionIiE10param_type1aEv(ptr noundef nonnull align 4 dereferenceable(8) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"struct.std::uniform_int_distribution<>::param_type", ptr %3, i32 0, i32 0
  %5 = load i32, ptr %4, align 4
  ret i32 %5
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %0) #2 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %4, i32 0, i32 1
  %6 = load i64, ptr %5, align 8
  %7 = icmp uge i64 %6, 624
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(ptr noundef nonnull align 8 dereferenceable(5000) %4)
  br label %9

9:                                                ; preds = %8, %1
  %10 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %4, i32 0, i32 0
  %11 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %4, i32 0, i32 1
  %12 = load i64, ptr %11, align 8
  %13 = add i64 %12, 1
  store i64 %13, ptr %11, align 8
  %14 = getelementptr inbounds [624 x i64], ptr %10, i64 0, i64 %12
  %15 = load i64, ptr %14, align 8
  store i64 %15, ptr %3, align 8
  %16 = load i64, ptr %3, align 8
  %17 = lshr i64 %16, 11
  %18 = and i64 %17, 4294967295
  %19 = load i64, ptr %3, align 8
  %20 = xor i64 %19, %18
  store i64 %20, ptr %3, align 8
  %21 = load i64, ptr %3, align 8
  %22 = shl i64 %21, 7
  %23 = and i64 %22, 2636928640
  %24 = load i64, ptr %3, align 8
  %25 = xor i64 %24, %23
  store i64 %25, ptr %3, align 8
  %26 = load i64, ptr %3, align 8
  %27 = shl i64 %26, 15
  %28 = and i64 %27, 4022730752
  %29 = load i64, ptr %3, align 8
  %30 = xor i64 %29, %28
  store i64 %30, ptr %3, align 8
  %31 = load i64, ptr %3, align 8
  %32 = lshr i64 %31, 18
  %33 = load i64, ptr %3, align 8
  %34 = xor i64 %33, %32
  store i64 %34, ptr %3, align 8
  %35 = load i64, ptr %3, align 8
  ret i64 %35
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(ptr noundef nonnull align 8 dereferenceable(5000) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %10 = load ptr, ptr %2, align 8
  store i64 -2147483648, ptr %3, align 8
  store i64 2147483647, ptr %4, align 8
  store i64 0, ptr %5, align 8
  br label %11

11:                                               ; preds = %44, %1
  %12 = load i64, ptr %5, align 8
  %13 = icmp ult i64 %12, 227
  br i1 %13, label %14, label %47

14:                                               ; preds = %11
  %15 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %16 = load i64, ptr %5, align 8
  %17 = getelementptr inbounds [624 x i64], ptr %15, i64 0, i64 %16
  %18 = load i64, ptr %17, align 8
  %19 = and i64 %18, -2147483648
  %20 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %21 = load i64, ptr %5, align 8
  %22 = add i64 %21, 1
  %23 = getelementptr inbounds [624 x i64], ptr %20, i64 0, i64 %22
  %24 = load i64, ptr %23, align 8
  %25 = and i64 %24, 2147483647
  %26 = or i64 %19, %25
  store i64 %26, ptr %6, align 8
  %27 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %28 = load i64, ptr %5, align 8
  %29 = add i64 %28, 397
  %30 = getelementptr inbounds [624 x i64], ptr %27, i64 0, i64 %29
  %31 = load i64, ptr %30, align 8
  %32 = load i64, ptr %6, align 8
  %33 = lshr i64 %32, 1
  %34 = xor i64 %31, %33
  %35 = load i64, ptr %6, align 8
  %36 = and i64 %35, 1
  %37 = icmp ne i64 %36, 0
  %38 = zext i1 %37 to i64
  %39 = select i1 %37, i64 2567483615, i64 0
  %40 = xor i64 %34, %39
  %41 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %42 = load i64, ptr %5, align 8
  %43 = getelementptr inbounds [624 x i64], ptr %41, i64 0, i64 %42
  store i64 %40, ptr %43, align 8
  br label %44

44:                                               ; preds = %14
  %45 = load i64, ptr %5, align 8
  %46 = add i64 %45, 1
  store i64 %46, ptr %5, align 8
  br label %11, !llvm.loop !13

47:                                               ; preds = %11
  store i64 227, ptr %7, align 8
  br label %48

48:                                               ; preds = %81, %47
  %49 = load i64, ptr %7, align 8
  %50 = icmp ult i64 %49, 623
  br i1 %50, label %51, label %84

51:                                               ; preds = %48
  %52 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %53 = load i64, ptr %7, align 8
  %54 = getelementptr inbounds [624 x i64], ptr %52, i64 0, i64 %53
  %55 = load i64, ptr %54, align 8
  %56 = and i64 %55, -2147483648
  %57 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %58 = load i64, ptr %7, align 8
  %59 = add i64 %58, 1
  %60 = getelementptr inbounds [624 x i64], ptr %57, i64 0, i64 %59
  %61 = load i64, ptr %60, align 8
  %62 = and i64 %61, 2147483647
  %63 = or i64 %56, %62
  store i64 %63, ptr %8, align 8
  %64 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %65 = load i64, ptr %7, align 8
  %66 = add i64 %65, -227
  %67 = getelementptr inbounds [624 x i64], ptr %64, i64 0, i64 %66
  %68 = load i64, ptr %67, align 8
  %69 = load i64, ptr %8, align 8
  %70 = lshr i64 %69, 1
  %71 = xor i64 %68, %70
  %72 = load i64, ptr %8, align 8
  %73 = and i64 %72, 1
  %74 = icmp ne i64 %73, 0
  %75 = zext i1 %74 to i64
  %76 = select i1 %74, i64 2567483615, i64 0
  %77 = xor i64 %71, %76
  %78 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %79 = load i64, ptr %7, align 8
  %80 = getelementptr inbounds [624 x i64], ptr %78, i64 0, i64 %79
  store i64 %77, ptr %80, align 8
  br label %81

81:                                               ; preds = %51
  %82 = load i64, ptr %7, align 8
  %83 = add i64 %82, 1
  store i64 %83, ptr %7, align 8
  br label %48, !llvm.loop !14

84:                                               ; preds = %48
  %85 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %86 = getelementptr inbounds [624 x i64], ptr %85, i64 0, i64 623
  %87 = load i64, ptr %86, align 8
  %88 = and i64 %87, -2147483648
  %89 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %90 = getelementptr inbounds [624 x i64], ptr %89, i64 0, i64 0
  %91 = load i64, ptr %90, align 8
  %92 = and i64 %91, 2147483647
  %93 = or i64 %88, %92
  store i64 %93, ptr %9, align 8
  %94 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %95 = getelementptr inbounds [624 x i64], ptr %94, i64 0, i64 396
  %96 = load i64, ptr %95, align 8
  %97 = load i64, ptr %9, align 8
  %98 = lshr i64 %97, 1
  %99 = xor i64 %96, %98
  %100 = load i64, ptr %9, align 8
  %101 = and i64 %100, 1
  %102 = icmp ne i64 %101, 0
  %103 = zext i1 %102 to i64
  %104 = select i1 %102, i64 2567483615, i64 0
  %105 = xor i64 %99, %104
  %106 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 0
  %107 = getelementptr inbounds [624 x i64], ptr %106, i64 0, i64 623
  store i64 %105, ptr %107, align 8
  %108 = getelementptr inbounds %"class.std::mersenne_twister_engine", ptr %10, i32 0, i32 1
  store i64 0, ptr %108, align 8
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE3maxEv() #2 comdat align 2 {
  %1 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %2 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh(i8 noundef zeroext 126)
  %3 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %4 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %3, i32 0, i32 0
  store i8 %2, ptr %4, align 1
  %5 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %1, i32 0, i32 0
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %5, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  ret i8 %7
}

; Function Attrs: nounwind
declare double @acos(double noundef) #3

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2IiSt9enable_ifILb1EvEEET_(ptr noundef nonnull align 1 dereferenceable(1) %0, i32 noundef %1) unnamed_addr #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %6 = alloca float, align 4
  %7 = alloca %"struct.ml_dtypes::float8_internal::float8_base<ml_dtypes::float8_internal::float8_ieee_p<3>>::ConstructFromRepTag", align 1
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %8 = load ptr, ptr %3, align 8
  %9 = load i32, ptr %4, align 4
  %10 = sitofp i32 %9 to float
  store float %10, ptr %6, align 4
  %11 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EfEES3_RKT1_(ptr noundef nonnull align 4 dereferenceable(4) %6)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %5, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  store i8 %11, ptr %13, align 1
  %14 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEC2EhNS4_19ConstructFromRepTagE(ptr noundef nonnull align 1 dereferenceable(1) %8, i8 noundef zeroext %14)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE11ConvertFromILb0ELb0EfEES3_RKT1_(ptr noundef nonnull align 4 dereferenceable(4) %0) #2 comdat align 2 {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call i8 @_ZN9ml_dtypes15float8_internal11ConvertImplIfNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKfb(ptr noundef nonnull align 4 dereferenceable(4) %4, i1 noundef zeroext false)
  %6 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %7 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %6, i32 0, i32 0
  store i8 %5, ptr %7, align 1
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %8, i32 0, i32 0
  %10 = load i8, ptr %9, align 1
  ret i8 %10
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal11ConvertImplIfNS0_13float8_ieee_pILi3EEELb0ELb0EvE3runERKfb(ptr noundef nonnull align 4 dereferenceable(4) %0, i1 noundef zeroext %1) #2 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %5 = alloca ptr, align 8
  %6 = alloca i8, align 1
  %7 = alloca i8, align 1
  %8 = alloca i32, align 4
  %9 = alloca float, align 4
  %10 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %11 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i8, align 1
  %19 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %20 = alloca i32, align 4
  %21 = alloca i8, align 1
  %22 = alloca i32, align 4
  %23 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %24 = alloca i32, align 4
  %25 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %26 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %5, align 8
  %27 = zext i1 %1 to i8
  store i8 %27, ptr %6, align 1
  %28 = load ptr, ptr %5, align 8
  %29 = call noundef i32 @_ZN5Eigen6numext8bit_castIjfEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %28)
  %30 = lshr i32 %29, 31
  %31 = icmp ne i32 %30, 0
  %32 = zext i1 %31 to i8
  store i8 %32, ptr %7, align 1
  %33 = load ptr, ptr %5, align 8
  store ptr %33, ptr %3, align 8
  %34 = load ptr, ptr %3, align 8
  %35 = load float, ptr %34, align 4
  %36 = call noundef float @_ZSt3absf(float noundef %35)
  store float %36, ptr %9, align 4
  %37 = call noundef i32 @_ZN5Eigen6numext8bit_castIjfEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %9)
  store i32 %37, ptr %8, align 4
  %38 = load ptr, ptr %5, align 8
  %39 = call noundef zeroext i1 @_ZN5Eigen6numext5isinfIfEEbRKT_(ptr noundef nonnull align 4 dereferenceable(4) %38)
  br i1 %39, label %40, label %55

40:                                               ; preds = %2
  %41 = load i8, ptr %7, align 1
  %42 = trunc i8 %41 to i1
  br i1 %42, label %43, label %50

43:                                               ; preds = %40
  %44 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %45 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %10, i32 0, i32 0
  %46 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %45, i32 0, i32 0
  store i8 %44, ptr %46, align 1
  %47 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %10)
  %48 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %49 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %48, i32 0, i32 0
  store i8 %47, ptr %49, align 1
  br label %54

50:                                               ; preds = %40
  %51 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %52 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %53 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %52, i32 0, i32 0
  store i8 %51, ptr %53, align 1
  br label %54

54:                                               ; preds = %50, %43
  br label %178

55:                                               ; preds = %2
  %56 = load ptr, ptr %5, align 8
  %57 = call noundef zeroext i1 @_ZN5Eigen6numext5isnanIfEEbRKT_(ptr noundef nonnull align 4 dereferenceable(4) %56)
  br i1 %57, label %58, label %62

58:                                               ; preds = %55
  %59 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE9quiet_NaNEv()
  %60 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %61 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %60, i32 0, i32 0
  store i8 %59, ptr %61, align 1
  br label %178

62:                                               ; preds = %55
  %63 = load i32, ptr %8, align 4
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %74

65:                                               ; preds = %62
  %66 = load i8, ptr %7, align 1
  %67 = trunc i8 %66 to i1
  br i1 %67, label %68, label %72

68:                                               ; preds = %65
  call void @llvm.memset.p0.i64(ptr align 1 %11, i8 0, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %69 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %11)
  %70 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %71 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %70, i32 0, i32 0
  store i8 %69, ptr %71, align 1
  br label %73

72:                                               ; preds = %65
  call void @llvm.memset.p0.i64(ptr align 1 %4, i8 0, i64 1, i1 false)
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EEC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %4)
  br label %73

73:                                               ; preds = %72, %68
  br label %178

74:                                               ; preds = %62
  %75 = load i32, ptr %8, align 4
  %76 = lshr i32 %75, 23
  store i32 %76, ptr %12, align 4
  %77 = load i32, ptr %12, align 4
  %78 = sub nsw i32 %77, 127
  store i32 %78, ptr %13, align 4
  %79 = load i32, ptr %13, align 4
  %80 = add nsw i32 %79, 16
  store i32 %80, ptr %14, align 4
  %81 = load i32, ptr %14, align 4
  %82 = icmp sle i32 %81, 0
  br i1 %82, label %83, label %127

83:                                               ; preds = %74
  %84 = load i32, ptr %12, align 4
  %85 = icmp sgt i32 %84, 0
  %86 = zext i1 %85 to i64
  %87 = select i1 %85, i32 1, i32 0
  store i32 %87, ptr %15, align 4
  %88 = load i32, ptr %14, align 4
  %89 = sub nsw i32 21, %88
  %90 = load i32, ptr %15, align 4
  %91 = add i32 %89, %90
  store i32 %91, ptr %16, align 4
  %92 = load i32, ptr %8, align 4
  %93 = and i32 %92, 8388607
  %94 = load i32, ptr %15, align 4
  %95 = shl i32 %94, 23
  %96 = or i32 %93, %95
  store i32 %96, ptr %17, align 4
  store i8 0, ptr %18, align 1
  %97 = load i32, ptr %16, align 4
  %98 = icmp sle i32 %97, 24
  br i1 %98, label %99, label %115

99:                                               ; preds = %83
  %100 = load i8, ptr %6, align 1
  %101 = trunc i8 %100 to i1
  br i1 %101, label %102, label %106

102:                                              ; preds = %99
  %103 = load i32, ptr %17, align 4
  %104 = load i32, ptr %16, align 4
  %105 = call noundef i32 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundIjEET_S2_i(i32 noundef %103, i32 noundef %104)
  store i32 %105, ptr %17, align 4
  br label %110

106:                                              ; preds = %99
  %107 = load i32, ptr %17, align 4
  %108 = load i32, ptr %16, align 4
  %109 = call noundef i32 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenIjEET_S2_i(i32 noundef %107, i32 noundef %108)
  store i32 %109, ptr %17, align 4
  br label %110

110:                                              ; preds = %106, %102
  %111 = load i32, ptr %17, align 4
  %112 = load i32, ptr %16, align 4
  %113 = lshr i32 %111, %112
  %114 = trunc i32 %113 to i8
  store i8 %114, ptr %18, align 1
  br label %115

115:                                              ; preds = %110, %83
  %116 = call i8 @_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %18)
  %117 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %19, i32 0, i32 0
  %118 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %117, i32 0, i32 0
  store i8 %116, ptr %118, align 1
  %119 = load i8, ptr %7, align 1
  %120 = trunc i8 %119 to i1
  br i1 %120, label %121, label %125

121:                                              ; preds = %115
  %122 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %19)
  %123 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %124 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %123, i32 0, i32 0
  store i8 %122, ptr %124, align 1
  br label %126

125:                                              ; preds = %115
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %19, i64 1, i1 false)
  br label %126

126:                                              ; preds = %125, %121
  br label %178

127:                                              ; preds = %74
  %128 = load i32, ptr %8, align 4
  store i32 %128, ptr %20, align 4
  %129 = load i8, ptr %6, align 1
  %130 = trunc i8 %129 to i1
  br i1 %130, label %131, label %134

131:                                              ; preds = %127
  %132 = load i32, ptr %8, align 4
  %133 = call noundef i32 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundIjEET_S2_i(i32 noundef %132, i32 noundef 21)
  store i32 %133, ptr %20, align 4
  br label %137

134:                                              ; preds = %127
  %135 = load i32, ptr %8, align 4
  %136 = call noundef i32 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenIjEET_S2_i(i32 noundef %135, i32 noundef 21)
  store i32 %136, ptr %20, align 4
  br label %137

137:                                              ; preds = %134, %131
  %138 = load i32, ptr %20, align 4
  %139 = and i32 %138, -2097152
  store i32 %139, ptr %20, align 4
  %140 = load i32, ptr %20, align 4
  %141 = add i32 %140, -931135488
  store i32 %141, ptr %20, align 4
  %142 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv()
  %143 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %23, i32 0, i32 0
  %144 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %143, i32 0, i32 0
  store i8 %142, ptr %144, align 1
  %145 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %23)
  %146 = zext i8 %145 to i32
  store i32 %146, ptr %22, align 4
  %147 = load i32, ptr %22, align 4
  store i32 %147, ptr %24, align 4
  %148 = load i32, ptr %24, align 4
  %149 = shl i32 %148, 21
  store i32 %149, ptr %24, align 4
  %150 = load i32, ptr %20, align 4
  %151 = lshr i32 %150, 21
  %152 = trunc i32 %151 to i8
  store i8 %152, ptr %21, align 1
  %153 = call i8 @_ZN5Eigen6numext8bit_castIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEhEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %21)
  %154 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %25, i32 0, i32 0
  %155 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %154, i32 0, i32 0
  store i8 %153, ptr %155, align 1
  %156 = load i32, ptr %20, align 4
  %157 = load i32, ptr %24, align 4
  %158 = icmp ugt i32 %156, %157
  br i1 %158, label %159, label %169

159:                                              ; preds = %137
  br i1 false, label %160, label %164

160:                                              ; preds = %159
  %161 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE7highestEv()
  %162 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %26, i32 0, i32 0
  %163 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %162, i32 0, i32 0
  store i8 %161, ptr %163, align 1
  br label %168

164:                                              ; preds = %159
  %165 = call i8 @_ZN5Eigen16GenericNumTraitsIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEE8infinityEv()
  %166 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %26, i32 0, i32 0
  %167 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %166, i32 0, i32 0
  store i8 %165, ptr %167, align 1
  br label %168

168:                                              ; preds = %164, %160
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %25, ptr align 1 %26, i64 1, i1 false)
  br label %169

169:                                              ; preds = %168, %137
  %170 = load i8, ptr %7, align 1
  %171 = trunc i8 %170 to i1
  br i1 %171, label %172, label %176

172:                                              ; preds = %169
  %173 = call i8 @_ZNK9ml_dtypes15float8_internal13float8_ieee_pILi3EEngEv(ptr noundef nonnull align 1 dereferenceable(1) %25)
  %174 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %175 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %174, i32 0, i32 0
  store i8 %173, ptr %175, align 1
  br label %177

176:                                              ; preds = %169
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %4, ptr align 1 %25, i64 1, i1 false)
  br label %177

177:                                              ; preds = %176, %172
  br label %178

178:                                              ; preds = %177, %126, %73, %58, %54
  %179 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %180 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %179, i32 0, i32 0
  %181 = load i8, ptr %180, align 1
  ret i8 %181
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZN5Eigen6numext8bit_castIjfEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca float, align 4
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = load float, ptr %5, align 4
  store float %6, ptr %4, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %4, i64 4, i1 false)
  %7 = load i32, ptr %3, align 4
  ret i32 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isinfIfEEbRKT_(ptr noundef nonnull align 4 dereferenceable(4) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIfEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 4 dereferenceable(4) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isnanIfEEbRKT_(ptr noundef nonnull align 4 dereferenceable(4) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIfEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 4 dereferenceable(4) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZN9ml_dtypes15float8_internal16Stochastic_RoundIjEET_S2_i(i32 noundef %0, i32 noundef %1) #2 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  store i32 1, ptr %5, align 4
  %10 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_(ptr noundef nonnull align 4 dereferenceable(8) @_ZL12distribution, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL2mt)
  store i32 %10, ptr %6, align 4
  %11 = load i32, ptr %5, align 4
  %12 = shl i32 1, %11
  %13 = sub i32 %12, 1
  store i32 %13, ptr %7, align 4
  %14 = load i32, ptr %6, align 4
  %15 = load i32, ptr %7, align 4
  %16 = and i32 %14, %15
  store i32 %16, ptr %8, align 4
  %17 = load i32, ptr %3, align 4
  %18 = load i32, ptr %8, align 4
  %19 = load i32, ptr %4, align 4
  %20 = load i32, ptr %5, align 4
  %21 = sub nsw i32 %19, %20
  %22 = shl i32 %18, %21
  %23 = add i32 %17, %22
  store i32 %23, ptr %9, align 4
  %24 = load i32, ptr %9, align 4
  ret i32 %24
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZN9ml_dtypes15float8_internal22RoundBitsToNearestEvenIjEET_S2_i(i32 noundef %0, i32 noundef %1) #1 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %6 = load i32, ptr %4, align 4
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %2
  br label %19

9:                                                ; preds = %2
  %10 = load i32, ptr %3, align 4
  %11 = load i32, ptr %4, align 4
  %12 = lshr i32 %10, %11
  %13 = and i32 %12, 1
  %14 = load i32, ptr %4, align 4
  %15 = sub nsw i32 %14, 1
  %16 = shl i32 1, %15
  %17 = add i32 %13, %16
  %18 = sub i32 %17, 1
  br label %19

19:                                               ; preds = %9, %8
  %20 = phi i32 [ 0, %8 ], [ %18, %9 ]
  store i32 %20, ptr %5, align 4
  %21 = load i32, ptr %3, align 4
  %22 = load i32, ptr %5, align 4
  %23 = add i32 %21, %22
  ret i32 %23
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZSt3absf(float noundef %0) #1 comdat {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = call float @llvm.fabs.f32(float %3)
  ret float %4
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #7

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIfEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 4 dereferenceable(4) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load float, ptr %3, align 4
  %5 = call noundef zeroext i1 @_ZSt5isinff(float noundef %4)
  ret i1 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZSt5isinff(float noundef %0) #1 comdat {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = call i1 @llvm.is.fpclass.f32(float %3, i32 516)
  ret i1 %4
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #7

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIfEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS3_EE17has_signaling_NaNntsr9NumTraitsIS3_EE9IsComplexEbE4typeERKS3_(ptr noundef nonnull align 4 dereferenceable(4) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load float, ptr %3, align 4
  %5 = call noundef zeroext i1 @_ZSt5isnanf(float noundef %4)
  ret i1 %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZSt5isnanf(float noundef %0) #1 comdat {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = call i1 @llvm.is.fpclass.f32(float %3, i32 3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef float @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIfLb0ELb0EEET_RKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef float @_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEfLb0ELb0EvE3runERKS3_b(ptr noundef nonnull align 1 dereferenceable(1) %3, i1 noundef zeroext false)
  ret float %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef float @_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEfLb0ELb0EvE3runERKS3_b(ptr noundef nonnull align 1 dereferenceable(1) %0, i1 noundef zeroext %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca float, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i8, align 1
  %8 = alloca i8, align 1
  %9 = alloca i8, align 1
  %10 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca float, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  %20 = alloca float, align 4
  %21 = alloca i32, align 4
  %22 = alloca float, align 4
  store ptr %0, ptr %6, align 8
  %23 = zext i1 %1 to i8
  store i8 %23, ptr %7, align 1
  %24 = load ptr, ptr %6, align 8
  %25 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %24)
  %26 = zext i8 %25 to i32
  %27 = ashr i32 %26, 7
  %28 = icmp ne i32 %27, 0
  %29 = zext i1 %28 to i8
  store i8 %29, ptr %8, align 1
  %30 = load ptr, ptr %6, align 8
  store ptr %30, ptr %4, align 8
  %31 = load ptr, ptr %4, align 8
  %32 = call i8 @_ZN9ml_dtypes15float8_internal3absILi3EEENS0_13float8_ieee_pIXT_EEERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %31)
  store i8 %32, ptr %3, align 1
  %33 = load i8, ptr %3, align 1
  %34 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %10, i32 0, i32 0
  %35 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %34, i32 0, i32 0
  store i8 %33, ptr %35, align 1
  %36 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %10)
  store i8 %36, ptr %9, align 1
  %37 = load ptr, ptr %6, align 8
  %38 = call noundef zeroext i1 @_ZN5Eigen6numext5isinfIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %37)
  br i1 %38, label %39, label %47

39:                                               ; preds = %2
  %40 = load i8, ptr %8, align 1
  %41 = trunc i8 %40 to i1
  %42 = zext i1 %41 to i64
  %43 = call noundef float @_ZN5Eigen16GenericNumTraitsIfE8infinityEv()
  %44 = fneg float %43
  %45 = call noundef float @_ZN5Eigen16GenericNumTraitsIfE8infinityEv()
  %46 = select i1 %41, float %44, float %45
  store float %46, ptr %5, align 4
  br label %125

47:                                               ; preds = %2
  %48 = load ptr, ptr %6, align 8
  %49 = call noundef zeroext i1 @_ZN5Eigen6numext5isnanIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %48)
  br i1 %49, label %50, label %52

50:                                               ; preds = %47
  %51 = call noundef float @_ZN5Eigen16GenericNumTraitsIfE9quiet_NaNEv()
  store float %51, ptr %5, align 4
  br label %125

52:                                               ; preds = %47
  %53 = load i8, ptr %9, align 1
  %54 = zext i8 %53 to i32
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %56, label %61

56:                                               ; preds = %52
  %57 = load i8, ptr %8, align 1
  %58 = trunc i8 %57 to i1
  %59 = zext i1 %58 to i64
  %60 = select i1 %58, float -0.000000e+00, float 0.000000e+00
  store float %60, ptr %5, align 4
  br label %125

61:                                               ; preds = %52
  %62 = load i8, ptr %9, align 1
  %63 = zext i8 %62 to i32
  %64 = ashr i32 %63, 2
  store i32 %64, ptr %11, align 4
  %65 = load i32, ptr %11, align 4
  %66 = icmp eq i32 %65, 0
  br i1 %66, label %67, label %104

67:                                               ; preds = %61
  %68 = load i8, ptr %9, align 1
  %69 = zext i8 %68 to i32
  store i32 %69, ptr %12, align 4
  %70 = load i8, ptr %9, align 1
  %71 = call noundef i32 @_ZN9ml_dtypes15float8_internalL11countl_zeroEh(i8 noundef zeroext %70)
  %72 = sub nsw i32 %71, 6
  %73 = add nsw i32 %72, 1
  store i32 %73, ptr %13, align 4
  %74 = load i32, ptr %13, align 4
  %75 = sub nsw i32 111, %74
  %76 = add nsw i32 %75, 1
  store i32 %76, ptr %14, align 4
  %77 = load i32, ptr %14, align 4
  %78 = icmp sle i32 %77, 0
  br i1 %78, label %79, label %80

79:                                               ; preds = %67
  br label %90

80:                                               ; preds = %67
  %81 = load i32, ptr %13, align 4
  %82 = load i32, ptr %12, align 4
  %83 = shl i32 %82, %81
  store i32 %83, ptr %12, align 4
  %84 = load i32, ptr %12, align 4
  %85 = and i32 %84, -5
  store i32 %85, ptr %12, align 4
  %86 = load i32, ptr %14, align 4
  %87 = shl i32 %86, 2
  %88 = load i32, ptr %12, align 4
  %89 = or i32 %88, %87
  store i32 %89, ptr %12, align 4
  br label %90

90:                                               ; preds = %80, %79
  %91 = load i32, ptr %12, align 4
  %92 = shl i32 %91, 21
  store i32 %92, ptr %12, align 4
  %93 = load i32, ptr %12, align 4
  store i32 %93, ptr %16, align 4
  %94 = call noundef float @_ZN5Eigen6numext8bit_castIfjEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %16)
  store float %94, ptr %15, align 4
  %95 = load i8, ptr %8, align 1
  %96 = trunc i8 %95 to i1
  br i1 %96, label %97, label %100

97:                                               ; preds = %90
  %98 = load float, ptr %15, align 4
  %99 = fneg float %98
  br label %102

100:                                              ; preds = %90
  %101 = load float, ptr %15, align 4
  br label %102

102:                                              ; preds = %100, %97
  %103 = phi float [ %99, %97 ], [ %101, %100 ]
  store float %103, ptr %5, align 4
  br label %125

104:                                              ; preds = %61
  %105 = load i8, ptr %9, align 1
  %106 = zext i8 %105 to i32
  store i32 %106, ptr %17, align 4
  %107 = load i32, ptr %17, align 4
  %108 = add i32 %107, 444
  store i32 %108, ptr %17, align 4
  %109 = call noundef float @_ZN5Eigen16GenericNumTraitsIfE7highestEv()
  store float %109, ptr %20, align 4
  %110 = call noundef i32 @_ZN5Eigen6numext8bit_castIjfEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %20)
  store i32 %110, ptr %19, align 4
  %111 = load i32, ptr %19, align 4
  store i32 %111, ptr %21, align 4
  %112 = load i32, ptr %17, align 4
  %113 = shl i32 %112, 21
  store i32 %113, ptr %17, align 4
  %114 = load i32, ptr %17, align 4
  store i32 %114, ptr %18, align 4
  %115 = call noundef float @_ZN5Eigen6numext8bit_castIfjEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %18)
  store float %115, ptr %22, align 4
  %116 = load i8, ptr %8, align 1
  %117 = trunc i8 %116 to i1
  br i1 %117, label %118, label %121

118:                                              ; preds = %104
  %119 = load float, ptr %22, align 4
  %120 = fneg float %119
  br label %123

121:                                              ; preds = %104
  %122 = load float, ptr %22, align 4
  br label %123

123:                                              ; preds = %121, %118
  %124 = phi float [ %120, %118 ], [ %122, %121 ]
  store float %124, ptr %5, align 4
  br label %125

125:                                              ; preds = %123, %102, %56, %50, %39
  %126 = load float, ptr %5, align 4
  ret float %126
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isinfIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_(ptr noundef nonnull align 1 dereferenceable(1) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZN5Eigen16GenericNumTraitsIfE8infinityEv() #1 comdat align 2 {
  %1 = call noundef float @_ZNSt14numeric_limitsIfE8infinityEv() #5
  ret float %1
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen6numext5isnanIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS7_EE17has_signaling_NaNntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_(ptr noundef nonnull align 1 dereferenceable(1) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZN5Eigen16GenericNumTraitsIfE9quiet_NaNEv() #1 comdat align 2 {
  %1 = call noundef float @_ZNSt14numeric_limitsIfE9quiet_NaNEv() #5
  ret float %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef i32 @_ZN9ml_dtypes15float8_internalL11countl_zeroEh(i8 noundef zeroext %0) #1 {
  %2 = alloca i8, align 1
  %3 = alloca i32, align 4
  store i8 %0, ptr %2, align 1
  store i32 4, ptr %3, align 4
  %4 = load i8, ptr %2, align 1
  %5 = zext i8 %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %8, label %15

8:                                                ; preds = %1
  %9 = load i32, ptr %3, align 4
  %10 = sub nsw i32 %9, 4
  store i32 %10, ptr %3, align 4
  %11 = load i8, ptr %2, align 1
  %12 = zext i8 %11 to i32
  %13 = ashr i32 %12, 4
  %14 = trunc i32 %13 to i8
  store i8 %14, ptr %2, align 1
  br label %15

15:                                               ; preds = %8, %1
  %16 = load i8, ptr %2, align 1
  %17 = zext i8 %16 to i64
  %18 = getelementptr inbounds [16 x i8], ptr @.str.16, i64 0, i64 %17
  %19 = load i8, ptr %18, align 1
  %20 = sext i8 %19 to i32
  %21 = load i32, ptr %3, align 4
  %22 = add nsw i32 %20, %21
  ret i32 %22
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZN5Eigen6numext8bit_castIfjEET_RKT0_(ptr noundef nonnull align 4 dereferenceable(4) %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca float, align 4
  %4 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = load i32, ptr %5, align 4
  store i32 %6, ptr %4, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %4, i64 4, i1 false)
  %7 = load float, ptr %3, align 4
  ret float %7
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZN5Eigen16GenericNumTraitsIfE7highestEv() #1 comdat align 2 {
  %1 = call noundef float @_ZNSt14numeric_limitsIfE3maxEv() #5
  ret float %1
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZN9ml_dtypes15float8_internal3absILi3EEENS0_13float8_ieee_pIXT_EEERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call noundef zeroext i1 @_ZN9ml_dtypes15float8_internal5isnanILi3EEEbRKNS0_13float8_ieee_pIXT_EEE(ptr noundef nonnull align 1 dereferenceable(1) %4)
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = load ptr, ptr %3, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %2, ptr align 1 %7, i64 1, i1 false)
  br label %17

8:                                                ; preds = %1
  %9 = load ptr, ptr %3, align 8
  %10 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %9)
  %11 = zext i8 %10 to i32
  %12 = and i32 %11, 127
  %13 = trunc i32 %12 to i8
  %14 = call i8 @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7FromRepEh(i8 noundef zeroext %13)
  %15 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %16 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %15, i32 0, i32 0
  store i8 %14, ptr %16, align 1
  br label %17

17:                                               ; preds = %8, %6
  %18 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %2, i32 0, i32 0
  %19 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %18, i32 0, i32 0
  %20 = load i8, ptr %19, align 1
  ret i8 %20
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN9ml_dtypes15float8_internal5isnanILi3EEEbRKNS0_13float8_ieee_pIXT_EEE(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %5 = zext i8 %4 to i32
  %6 = icmp eq i32 %5, 128
  ret i1 %6
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isinf_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaasr3std14numeric_limitsIT_EE12has_infinityntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN9ml_dtypes15float8_internal5isinfINS0_13float8_ieee_pILi3EEEEEbRKNS0_11float8_baseIT_EE(ptr noundef nonnull align 1 dereferenceable(1) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN9ml_dtypes15float8_internal5isinfINS0_13float8_ieee_pILi3EEEEEbRKNS0_11float8_baseIT_EE(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %5)
  %7 = call i8 @_ZN9ml_dtypes15float8_internal3absILi3EEENS0_13float8_ieee_pIXT_EEERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %9 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %8, i32 0, i32 0
  store i8 %7, ptr %9, align 1
  %10 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %3)
  %11 = zext i8 %10 to i32
  %12 = call i8 @_ZN9ml_dtypes15float8_internal28numeric_limits_float8_ieee_pILi3EE8infinityEv()
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %4, i32 0, i32 0
  %14 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %13, i32 0, i32 0
  store i8 %12, ptr %14, align 1
  %15 = call noundef zeroext i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE3repEv(ptr noundef nonnull align 1 dereferenceable(1) %4)
  %16 = zext i8 %15 to i32
  %17 = icmp eq i32 %11, %16
  ret i1 %17
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZNSt14numeric_limitsIfE8infinityEv() #1 comdat align 2 {
  ret float 0x7FF0000000000000
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN5Eigen8internal10isnan_implIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEENSt9enable_ifIXaaoosr3std14numeric_limitsIT_EE13has_quiet_NaNsr3std14numeric_limitsIS7_EE17has_signaling_NaNntsr9NumTraitsIS7_EE9IsComplexEbE4typeERKS7_(ptr noundef nonnull align 1 dereferenceable(1) %0) #1 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef zeroext i1 @_ZN9ml_dtypes15float8_internal5isnanILi3EEEbRKNS0_13float8_ieee_pIXT_EEE(ptr noundef nonnull align 1 dereferenceable(1) %3)
  ret i1 %4
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZNSt14numeric_limitsIfE9quiet_NaNEv() #1 comdat align 2 {
  ret float 0x7FF8000000000000
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZNSt14numeric_limitsIfE3maxEv() #1 comdat align 2 {
  ret float 0x47EFFFFFE0000000
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZSt3sinf(float noundef %0) #1 comdat {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = call float @sinf(float noundef %3) #5
  ret float %4
}

; Function Attrs: nounwind
declare float @sinf(float noundef) #3

; Function Attrs: nounwind
declare double @sqrt(double noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local i8 @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEmiERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef nonnull align 1 dereferenceable(1) %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE7derivedEv(ptr noundef nonnull align 1 dereferenceable(1) %6)
  %8 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %7)
  %9 = load ptr, ptr %5, align 8
  %10 = call noundef double @_ZNK9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEEcvdEv(ptr noundef nonnull align 1 dereferenceable(1) %9)
  %11 = fsub double %8, %10
  call void @_ZN9ml_dtypes15float8_internal13float8_ieee_pILi3EECI2NS0_11float8_baseIS2_EEEd(ptr noundef nonnull align 1 dereferenceable(1) %3, double noundef %11)
  %12 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %3, i32 0, i32 0
  %13 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %12, i32 0, i32 0
  %14 = load i8, ptr %13, align 1
  ret i8 %14
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef double @_ZN9ml_dtypes15float8_internal11float8_baseINS0_13float8_ieee_pILi3EEEE9ConvertToIdLb0ELb0EEET_RKS3_(ptr noundef nonnull align 1 dereferenceable(1) %0) #2 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef double @_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEdLb0ELb0EvE3runERKS3_b(ptr noundef nonnull align 1 dereferenceable(1) %3, i1 noundef zeroext false)
  ret double %4
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef double @_ZN9ml_dtypes15float8_internal11ConvertImplINS0_13float8_ieee_pILi3EEEdLb0ELb0EvE3runERKS3_b(ptr noundef nonnull align 1 dereferenceable(1) %0, i1 noundef zeroext %1) #2 comdat align 2 {
  %3 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i8, align 1
  %8 = alloca i8, align 1
  %9 = alloca i8, align 1
  %10 = alloca %"class.ml_dtypes::float8_internal::float8_ieee_p", align 1
  %11 = alloca i32, align 4
  %12 = alloca i64, align 8
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca double, align 8
  %16 = alloca i64, align 8
  %17 = alloca i64, align 8
  %18 = alloca i64, align 8
  %19 = alloca i64, align 8
  %20 = alloca double, align 8
  %21 = alloca i64, align 8
  %22 = alloca double, align 8
  store ptr %0, ptr %6, align 8
  %23 = zext i1 %1 to i8
  store i8 %23, ptr %7, align 1
  %24 = load ptr, ptr %6, align 8
  %25 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %24)
  %26 = zext i8 %25 to i32
  %27 = ashr i32 %26, 7
  %28 = icmp ne i32 %27, 0
  %29 = zext i1 %28 to i8
  store i8 %29, ptr %8, align 1
  %30 = load ptr, ptr %6, align 8
  store ptr %30, ptr %4, align 8
  %31 = load ptr, ptr %4, align 8
  %32 = call i8 @_ZN9ml_dtypes15float8_internal3absILi3EEENS0_13float8_ieee_pIXT_EEERKS3_(ptr noundef nonnull align 1 dereferenceable(1) %31)
  store i8 %32, ptr %3, align 1
  %33 = load i8, ptr %3, align 1
  %34 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_ieee_p", ptr %10, i32 0, i32 0
  %35 = getelementptr inbounds %"class.ml_dtypes::float8_internal::float8_base", ptr %34, i32 0, i32 0
  store i8 %33, ptr %35, align 1
  %36 = call noundef zeroext i8 @_ZN5Eigen6numext8bit_castIhN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEET_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %10)
  store i8 %36, ptr %9, align 1
  %37 = load ptr, ptr %6, align 8
  %38 = call noundef zeroext i1 @_ZN5Eigen6numext5isinfIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %37)
  br i1 %38, label %39, label %47

39:                                               ; preds = %2
  %40 = load i8, ptr %8, align 1
  %41 = trunc i8 %40 to i1
  %42 = zext i1 %41 to i64
  %43 = call noundef double @_ZN5Eigen16GenericNumTraitsIdE8infinityEv()
  %44 = fneg double %43
  %45 = call noundef double @_ZN5Eigen16GenericNumTraitsIdE8infinityEv()
  %46 = select i1 %41, double %44, double %45
  store double %46, ptr %5, align 8
  br label %127

47:                                               ; preds = %2
  %48 = load ptr, ptr %6, align 8
  %49 = call noundef zeroext i1 @_ZN5Eigen6numext5isnanIN9ml_dtypes15float8_internal13float8_ieee_pILi3EEEEEbRKT_(ptr noundef nonnull align 1 dereferenceable(1) %48)
  br i1 %49, label %50, label %52

50:                                               ; preds = %47
  %51 = call noundef double @_ZN5Eigen16GenericNumTraitsIdE9quiet_NaNEv()
  store double %51, ptr %5, align 8
  br label %127

52:                                               ; preds = %47
  %53 = load i8, ptr %9, align 1
  %54 = zext i8 %53 to i32
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %56, label %61

56:                                               ; preds = %52
  %57 = load i8, ptr %8, align 1
  %58 = trunc i8 %57 to i1
  %59 = zext i1 %58 to i64
  %60 = select i1 %58, double -0.000000e+00, double 0.000000e+00
  store double %60, ptr %5, align 8
  br label %127

61:                                               ; preds = %52
  %62 = load i8, ptr %9, align 1
  %63 = zext i8 %62 to i32
  %64 = ashr i32 %63, 2
  store i32 %64, ptr %11, align 4
  %65 = load i32, ptr %11, align 4
  %66 = icmp eq i32 %65, 0
  br i1 %66, label %67, label %106

67:                                               ; preds = %61
  %68 = load i8, ptr %9, align 1
  %69 = zext i8 %68 to i64
  store i64 %69, ptr %12, align 8
  %70 = load i8, ptr %9, align 1
  %71 = call noundef i32 @_ZN9ml_dtypes15float8_internalL11countl_zeroEh(i8 noundef zeroext %70)
  %72 = sub nsw i32 %71, 6
  %73 = add nsw i32 %72, 1
  store i32 %73, ptr %13, align 4
  %74 = load i32, ptr %13, align 4
  %75 = sub nsw i32 1007, %74
  %76 = add nsw i32 %75, 1
  store i32 %76, ptr %14, align 4
  %77 = load i32, ptr %14, align 4
  %78 = icmp sle i32 %77, 0
  br i1 %78, label %79, label %80

79:                                               ; preds = %67
  br label %92

80:                                               ; preds = %67
  %81 = load i32, ptr %13, align 4
  %82 = load i64, ptr %12, align 8
  %83 = zext i32 %81 to i64
  %84 = shl i64 %82, %83
  store i64 %84, ptr %12, align 8
  %85 = load i64, ptr %12, align 8
  %86 = and i64 %85, -5
  store i64 %86, ptr %12, align 8
  %87 = load i32, ptr %14, align 4
  %88 = sext i32 %87 to i64
  %89 = shl i64 %88, 2
  %90 = load i64, ptr %12, align 8
  %91 = or i64 %90, %89
  store i64 %91, ptr %12, align 8
  br label %92

92:                                               ; preds = %80, %79
  %93 = load i64, ptr %12, align 8
  %94 = shl i64 %93, 50
  store i64 %94, ptr %12, align 8
  %95 = load i64, ptr %12, align 8
  store i64 %95, ptr %16, align 8
  %96 = call noundef double @_ZN5Eigen6numext8bit_castIdmEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %16)
  store double %96, ptr %15, align 8
  %97 = load i8, ptr %8, align 1
  %98 = trunc i8 %97 to i1
  br i1 %98, label %99, label %102

99:                                               ; preds = %92
  %100 = load double, ptr %15, align 8
  %101 = fneg double %100
  br label %104

102:                                              ; preds = %92
  %103 = load double, ptr %15, align 8
  br label %104

104:                                              ; preds = %102, %99
  %105 = phi double [ %101, %99 ], [ %103, %102 ]
  store double %105, ptr %5, align 8
  br label %127

106:                                              ; preds = %61
  %107 = load i8, ptr %9, align 1
  %108 = zext i8 %107 to i64
  store i64 %108, ptr %17, align 8
  %109 = load i64, ptr %17, align 8
  %110 = add i64 %109, 4028
  store i64 %110, ptr %17, align 8
  %111 = call noundef double @_ZN5Eigen16GenericNumTraitsIdE7highestEv()
  store double %111, ptr %20, align 8
  %112 = call noundef i64 @_ZN5Eigen6numext8bit_castImdEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %20)
  store i64 %112, ptr %19, align 8
  %113 = load i64, ptr %19, align 8
  store i64 %113, ptr %21, align 8
  %114 = load i64, ptr %17, align 8
  %115 = shl i64 %114, 50
  store i64 %115, ptr %17, align 8
  %116 = load i64, ptr %17, align 8
  store i64 %116, ptr %18, align 8
  %117 = call noundef double @_ZN5Eigen6numext8bit_castIdmEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %18)
  store double %117, ptr %22, align 8
  %118 = load i8, ptr %8, align 1
  %119 = trunc i8 %118 to i1
  br i1 %119, label %120, label %123

120:                                              ; preds = %106
  %121 = load double, ptr %22, align 8
  %122 = fneg double %121
  br label %125

123:                                              ; preds = %106
  %124 = load double, ptr %22, align 8
  br label %125

125:                                              ; preds = %123, %120
  %126 = phi double [ %122, %120 ], [ %124, %123 ]
  store double %126, ptr %5, align 8
  br label %127

127:                                              ; preds = %125, %104, %56, %50, %39
  %128 = load double, ptr %5, align 8
  ret double %128
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZN5Eigen16GenericNumTraitsIdE8infinityEv() #1 comdat align 2 {
  %1 = call noundef double @_ZNSt14numeric_limitsIdE8infinityEv() #5
  ret double %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZN5Eigen16GenericNumTraitsIdE9quiet_NaNEv() #1 comdat align 2 {
  %1 = call noundef double @_ZNSt14numeric_limitsIdE9quiet_NaNEv() #5
  ret double %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZN5Eigen6numext8bit_castIdmEET_RKT0_(ptr noundef nonnull align 8 dereferenceable(8) %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = load i64, ptr %5, align 8
  store i64 %6, ptr %4, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %3, ptr align 1 %4, i64 8, i1 false)
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZN5Eigen16GenericNumTraitsIdE7highestEv() #1 comdat align 2 {
  %1 = call noundef double @_ZNSt14numeric_limitsIdE3maxEv() #5
  ret double %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZNSt14numeric_limitsIdE8infinityEv() #1 comdat align 2 {
  ret double 0x7FF0000000000000
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZNSt14numeric_limitsIdE9quiet_NaNEv() #1 comdat align 2 {
  ret double 0x7FF8000000000000
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef double @_ZNSt14numeric_limitsIdE3maxEv() #1 comdat align 2 {
  ret double 0x7FEFFFFFFFFFFFFF
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8), float noundef) #4

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_funarc_8.cpp() #0 section ".text.startup" {
  call void @__cxx_global_var_init()
  call void @__cxx_global_var_init.1()
  call void @__cxx_global_var_init.2()
  call void @__cxx_global_var_init.3()
  call void @__cxx_global_var_init.4()
  call void @__cxx_global_var_init.5()
  ret void
}

attributes #0 = { noinline uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }
attributes #6 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 18.1.6 (++20240518023231+1118c2e05e67-1~exp1~20240518143320.131)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
