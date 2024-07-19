; ModuleID = 'funarc.c'
source_filename = "funarc.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque

@.str = private unnamed_addr constant [26 x i8] c" VERIFICATION SUCCESSFUL\0A\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c" Zeta is    %20.13E\0A\00", align 1
@.str.2 = private unnamed_addr constant [21 x i8] c" Error is   %20.13E\0A\00", align 1
@.str.3 = private unnamed_addr constant [22 x i8] c" VERIFICATION FAILED\0A\00", align 1
@.str.4 = private unnamed_addr constant [30 x i8] c" Zeta                %20.13E\0A\00", align 1
@.str.5 = private unnamed_addr constant [30 x i8] c" The correct zeta is %20.13E\0A\00", align 1
@.str.6 = private unnamed_addr constant [10 x i8] c"./log.txt\00", align 1
@.str.7 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.8 = private unnamed_addr constant [6 x i8] c"true\0A\00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"false\0A\00", align 1
@.str.10 = private unnamed_addr constant [9 x i8] c"%20.13E\0A\00", align 1
@.str.11 = private unnamed_addr constant [11 x i8] c"./time.txt\00", align 1
@.str.12 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local double @fun(double %0) #0 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store double %0, double* %2, align 8
  store i32 5, i32* %4, align 4
  store double 1.000000e+00, double* %5, align 8
  store double 1.000000e+00, double* %6, align 8
  %7 = load double, double* %2, align 8
  store double %7, double* %5, align 8
  store i32 1, i32* %3, align 4
  br label %8

8:                                                ; preds = %23, %1
  %9 = load i32, i32* %3, align 4
  %10 = load i32, i32* %4, align 4
  %11 = icmp sle i32 %9, %10
  br i1 %11, label %12, label %26

12:                                               ; preds = %8
  %13 = load double, double* %6, align 8
  %14 = fmul double 2.000000e+00, %13
  store double %14, double* %6, align 8
  %15 = load double, double* %5, align 8
  %16 = load double, double* %6, align 8
  %17 = load double, double* %2, align 8
  %18 = fmul double %16, %17
  %19 = call double @sin(double %18) #4
  %20 = load double, double* %6, align 8
  %21 = fdiv double %19, %20
  %22 = fadd double %15, %21
  store double %22, double* %5, align 8
  br label %23

23:                                               ; preds = %12
  %24 = load i32, i32* %3, align 4
  %25 = add nsw i32 %24, 1
  store i32 %25, i32* %3, align 4
  br label %8, !llvm.loop !2

26:                                               ; preds = %8
  %27 = load double, double* %5, align 8
  ret double %27
}

; Function Attrs: nounwind
declare dso_local double @sin(double) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca double, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca double, align 8
  %12 = alloca double, align 8
  %13 = alloca double, align 8
  %14 = alloca double, align 8
  %15 = alloca double, align 8
  %16 = alloca double, align 8
  %17 = alloca double, align 8
  %18 = alloca i8, align 1
  %19 = alloca %struct._IO_FILE*, align 8
  %20 = alloca %struct._IO_FILE*, align 8
  store i32 0, i32* %1, align 4
  store double 1.000000e-04, double* %2, align 8
  %21 = call i64 @clock() #4
  store i64 %21, i64* %4, align 8
  %22 = call float @sqrtf(float 0.000000e+00) #4
  %23 = call float @acosf(float 0.000000e+00) #4
  %24 = call float @sinf(float 0.000000e+00) #4
  store i32 1000000, i32* %10, align 4
  store i32 0, i32* %3, align 4
  br label %25

25:                                               ; preds = %64, %0
  %26 = load i32, i32* %3, align 4
  %27 = icmp slt i32 %26, 10
  br i1 %27, label %28, label %67

28:                                               ; preds = %25
  store double -1.000000e+00, double* %12, align 8
  %29 = load double, double* %12, align 8
  %30 = call double @acos(double %29) #4
  store double %30, double* %14, align 8
  store double 0.000000e+00, double* %15, align 8
  store double 0.000000e+00, double* %12, align 8
  %31 = load double, double* %14, align 8
  %32 = load i32, i32* %10, align 4
  %33 = sitofp i32 %32 to double
  %34 = fdiv double %31, %33
  store double %34, double* %11, align 8
  store i32 1, i32* %7, align 4
  br label %35

35:                                               ; preds = %60, %28
  %36 = load i32, i32* %7, align 4
  %37 = load i32, i32* %10, align 4
  %38 = icmp sle i32 %36, %37
  br i1 %38, label %39, label %63

39:                                               ; preds = %35
  %40 = load i32, i32* %7, align 4
  %41 = sitofp i32 %40 to double
  %42 = load double, double* %11, align 8
  %43 = fmul double %41, %42
  %44 = call double @fun(double %43)
  store double %44, double* %13, align 8
  %45 = load double, double* %15, align 8
  %46 = load double, double* %11, align 8
  %47 = load double, double* %11, align 8
  %48 = fmul double %46, %47
  %49 = load double, double* %13, align 8
  %50 = load double, double* %12, align 8
  %51 = fsub double %49, %50
  %52 = load double, double* %13, align 8
  %53 = load double, double* %12, align 8
  %54 = fsub double %52, %53
  %55 = fmul double %51, %54
  %56 = fadd double %48, %55
  %57 = call double @sqrt(double %56) #4
  %58 = fadd double %45, %57
  store double %58, double* %15, align 8
  %59 = load double, double* %13, align 8
  store double %59, double* %12, align 8
  br label %60

60:                                               ; preds = %39
  %61 = load i32, i32* %7, align 4
  %62 = add nsw i32 %61, 1
  store i32 %62, i32* %7, align 4
  br label %35, !llvm.loop !4

63:                                               ; preds = %35
  br label %64

64:                                               ; preds = %63
  %65 = load i32, i32* %3, align 4
  %66 = add nsw i32 %65, 1
  store i32 %66, i32* %3, align 4
  br label %25, !llvm.loop !5

67:                                               ; preds = %25
  store double 0x40172EDFFCFEC7AB, double* %16, align 8
  %68 = load double, double* %15, align 8
  %69 = load double, double* %16, align 8
  %70 = fsub double %68, %69
  %71 = call double @llvm.fabs.f64(double %70)
  %72 = load double, double* %16, align 8
  %73 = fdiv double %71, %72
  store double %73, double* %17, align 8
  %74 = load double, double* %17, align 8
  %75 = load double, double* %2, align 8
  %76 = fcmp ole double %74, %75
  br i1 %76, label %77, label %83

77:                                               ; preds = %67
  store i8 1, i8* %18, align 1
  %78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str, i64 0, i64 0))
  %79 = load double, double* %15, align 8
  %80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), double %79)
  %81 = load double, double* %17, align 8
  %82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.2, i64 0, i64 0), double %81)
  br label %89

83:                                               ; preds = %67
  store i8 0, i8* %18, align 1
  %84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.3, i64 0, i64 0))
  %85 = load double, double* %15, align 8
  %86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.4, i64 0, i64 0), double %85)
  %87 = load double, double* %16, align 8
  %88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.5, i64 0, i64 0), double %87)
  br label %89

89:                                               ; preds = %83, %77
  %90 = call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i64 0, i64 0))
  store %struct._IO_FILE* %90, %struct._IO_FILE** %19, align 8
  %91 = load i8, i8* %18, align 1
  %92 = trunc i8 %91 to i1
  %93 = zext i1 %92 to i64
  %94 = select i1 %92, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i64 0, i64 0)
  %95 = load %struct._IO_FILE*, %struct._IO_FILE** %19, align 8
  %96 = call i32 @fputs(i8* %94, %struct._IO_FILE* %95)
  %97 = load %struct._IO_FILE*, %struct._IO_FILE** %19, align 8
  %98 = load double, double* %15, align 8
  %99 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %97, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.10, i64 0, i64 0), double %98)
  %100 = load %struct._IO_FILE*, %struct._IO_FILE** %19, align 8
  %101 = load double, double* %17, align 8
  %102 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %100, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.10, i64 0, i64 0), double %101)
  %103 = call i64 @clock() #4
  store i64 %103, i64* %5, align 8
  %104 = load i64, i64* %5, align 8
  %105 = load i64, i64* %4, align 8
  %106 = sub nsw i64 %104, %105
  %107 = sitofp i64 %106 to double
  %108 = fdiv double %107, 1.000000e+06
  store double %108, double* %6, align 8
  %109 = call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.11, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i64 0, i64 0))
  store %struct._IO_FILE* %109, %struct._IO_FILE** %20, align 8
  %110 = load %struct._IO_FILE*, %struct._IO_FILE** %20, align 8
  %111 = load double, double* %6, align 8
  %112 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %110, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.12, i64 0, i64 0), double %111)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i64 @clock() #1

; Function Attrs: nounwind
declare dso_local float @sqrtf(float) #1

; Function Attrs: nounwind
declare dso_local float @acosf(float) #1

; Function Attrs: nounwind
declare dso_local float @sinf(float) #1

; Function Attrs: nounwind
declare dso_local double @acos(double) #1

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double) #2

declare dso_local i32 @printf(i8*, ...) #3

declare dso_local %struct._IO_FILE* @fopen(i8*, i8*) #3

declare dso_local i32 @fputs(i8*, %struct._IO_FILE*) #3

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #3

attributes #0 = { noinline nounwind optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (https://github.com/llvm/llvm-project/ fed41342a82f5a3a9201819a82bf7a48313e296b)"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !3}
!5 = distinct !{!5, !3}
