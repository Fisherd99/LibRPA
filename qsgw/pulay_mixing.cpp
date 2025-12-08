#include "pulay_mixing.h"
#include "lapack_connector.h"
#include <deque>
#include <iostream>
#include <vector>
#include <stdexcept>

// 手动声明 LAPACK 函数，因为 LapackConnector 中缺失这些函数
// 注意：Fortran 函数名通常在 C++ 中需要加下划线
extern "C" {
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b, const int* ldb, int* info);
}

// 构造函数
PulayMixer::PulayMixer(int max_history, double mixing_beta) 
    : max_history_(max_history), current_step_(0), mixing_beta_(mixing_beta), 
      initialized_(false), nrows_(0), ncols_(0) {}

// 初始化混合器
void PulayMixer::initialize(const matrix& initial_guess) {
    nrows_ = initial_guess.nr;
    ncols_ = initial_guess.nc;
    input_history_.clear();
    residual_history_.clear();
    
    // 存储初始猜测
    input_history_.push_back(initial_guess);
    initialized_ = true;
    current_step_ = 0;

    std::cout << "[PulayMixer] Initialized with matrix of size " 
              << nrows_ << "x" << ncols_ << std::endl;
}

// 执行 Pulay 混合
matrix PulayMixer::mix(const matrix& current_output) {
    if (!initialized_) {
        throw std::runtime_error("[PulayMixer] Not initialized. Call initialize() first.");
    }
    
    if (current_output.nr != nrows_ || current_output.nc != ncols_) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions do not match initialization.");
    }
    
    current_step_++;
    
    // 计算当前残差
    matrix current_residual = current_output - input_history_.back();
    
    // 将残差加入历史
    residual_history_.push_back(current_residual);
    
    // 如果历史记录过多，删除最旧的
    // 修复：vector 没有 pop_front，使用 erase(begin())
    if (residual_history_.size() > max_history_) {
        residual_history_.erase(residual_history_.begin());
        input_history_.erase(input_history_.begin());
    }
    
    int history_size = residual_history_.size();
    bool use_linear_mixing = (history_size <= 1);
    matrix alpha;

    // 尝试 Pulay 混合
    if (!use_linear_mixing) {
        try {
            // Pulay 混合：构建并求解线性方程组
            matrix B(history_size + 1, history_size + 1, true);  // 内积矩阵（带约束）
            matrix rhs(history_size + 1, 1, true);               // 右端项
            
            // 构建内积矩阵 B
            for (int i = 0; i < history_size; i++) {
                for (int j = i; j < history_size; j++) {
                    double inner_prod = matrix_inner_product(residual_history_[i], residual_history_[j]);
                    B(i, j) = inner_prod;
                    B(j, i) = inner_prod;  // 对称矩阵
                }
                // 约束条件对应的列和行：Sum(alpha) = 1
                // 使用 -1.0 是标准推导形式，对应拉格朗日乘子 -lambda
                B(i, history_size) = -1.0;
                B(history_size, i) = -1.0;
            }
            B(history_size, history_size) = 0.0;  // 约束位置的0
            
            // 构建右端项 [0, 0, ..., 0, -1]^T
            rhs(history_size, 0) = -1.0;
            
            // 求解线性方程组 B * alpha = rhs
            alpha = solve_linear_system(B, rhs);
            
        } catch (const std::exception& e) {
            std::cerr << "[PulayMixer] Warning: Pulay mixing failed (" << e.what() << "). Resetting history and falling back to linear mixing." << std::endl;
            
            // 混合失败（通常是矩阵奇异），重置历史，仅保留当前步
            // 修复：使用 erase 清理历史
            while (residual_history_.size() > 1) {
                residual_history_.erase(residual_history_.begin());
                input_history_.erase(input_history_.begin());
            }
            history_size = 1;
            use_linear_mixing = true;
        }
    }
    
    if (use_linear_mixing) {
        // 简单线性混合 (Linear Mixing)
        matrix new_input = input_history_.back() + mixing_beta_ * current_residual;
        input_history_.push_back(new_input);
        std::cout << "[PulayMixer] Performed simple linear mixing." << std::endl;
        return new_input;
    }
    
    // Pulay 混合：计算新的输入矩阵
    matrix new_input(nrows_, ncols_, true);
    for (int i = 0; i < history_size; i++) {
        matrix term = input_history_[i] + mixing_beta_ * residual_history_[i];
        new_input += alpha(i, 0) * term;
    }
    
    // 将新输入加入历史
    input_history_.push_back(new_input);

    std::cout << "[PulayMixer] Performed Pulay mixing at step " << current_step_ << "." << std::endl;
    return new_input;
}

// 获取当前历史记录大小
int PulayMixer::get_history_size() const {
    return residual_history_.size();
}

// 获取当前迭代步数
int PulayMixer::get_current_step() const {
    return current_step_;
}

// 重置混合器
void PulayMixer::reset() {
    initialized_ = false;
    input_history_.clear();
    residual_history_.clear();
    nrows_ = 0;
    ncols_ = 0;
    current_step_ = 0;

    std::cout << "[PulayMixer] Reset mixer." << std::endl;
}

// 设置混合参数
void PulayMixer::set_mixing_beta(double beta) {
    mixing_beta_ = beta;
}

// 获取混合参数
double PulayMixer::get_mixing_beta() const {
    return mixing_beta_;
}

// 私有成员函数实现

// 计算两个矩阵的内积（视为向量的点积）
double PulayMixer::matrix_inner_product(const matrix& A, const matrix& B) {
    if (A.nr != B.nr || A.nc != B.nc) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions must match for inner product.");
    }
    
    double result = 0.0;
    for (int i = 0; i < A.size; i++) {
        result += A.c[i] * B.c[i];
    }
    return result;
}

// 求解线性方程组 Ax = b
matrix PulayMixer::solve_linear_system(const matrix& A, const matrix& b) {
    if (A.nr != A.nc) {
        throw std::runtime_error("[PulayMixer] Matrix A must be square for linear system solving.");
    }
    if (b.nc != 1) {
        throw std::runtime_error("[PulayMixer] Right-hand side must be a column vector.");
    }
    if (A.nr != b.nr) {
        throw std::runtime_error("[PulayMixer] Matrix A and vector b must have compatible dimensions.");
    }
    
    int n = A.nr;
    matrix A_copy = A;  // 工作副本
    matrix x = b;       // 解向量
    
    // 使用LU分解求解
    int* ipiv = new int[n];
    int info;
    
    // 修复：直接调用 LAPACK dgetrf_
    dgetrf_(&n, &n, A_copy.c, &n, ipiv, &info);
    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("[PulayMixer] LU factorization failed.");
    }
    
    // 修复：直接调用 LAPACK dgetrs_
    char trans = 'N';
    int nrhs = 1;
    dgetrs_(&trans, &n, &nrhs, A_copy.c, &n, ipiv, x.c, &n, &info);
    
    delete[] ipiv;
    
    if (info != 0) {
        throw std::runtime_error("[PulayMixer] Linear system solving failed.");
    }
    
    return x;
}