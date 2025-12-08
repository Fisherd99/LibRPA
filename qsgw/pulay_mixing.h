#ifndef PULAY_MIXER_H
#define PULAY_MIXER_H

#include "matrix.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>

class PulayMixer {
private:
    int max_history_;           // 最大历史记录数
    int current_step_;          // 当前步数
    double mixing_beta_;        // 混合参数
    bool initialized_;          // 是否已初始化
    
    // 历史数据存储
    std::vector<matrix> input_history_;   // 输入矩阵历史
    std::vector<matrix> residual_history_; // 残差矩阵历史
    
    // 矩阵尺寸
    int nrows_;
    int ncols_;
    
    // 私有成员函数声明
    double matrix_inner_product(const matrix& A, const matrix& B);
    matrix solve_linear_system(const matrix& A, const matrix& b);
    
public:
    // 构造函数
    PulayMixer(int max_history = 5, double mixing_beta = 0.1);
    
    /**
     * @brief 初始化混合器，必须在第一次调用 mix 前调用
     * @param initial_guess 初始猜测矩阵
     */
    void initialize(const matrix& initial_guess);
    
    /**
     * @brief 执行 Pulay 混合
     * @param current_output 当前迭代的输出矩阵
     * @return 混合后的新输入矩阵
     */
    matrix mix(const matrix& current_output);
    
    /**
     * @brief 获取当前历史记录大小
     */
    int get_history_size() const;
    
    /**
     * @brief 获取当前迭代步数
     */
    int get_current_step() const;
    
    /**
     * @brief 重置混合器
     */
    void reset();
    
    /**
     * @brief 设置混合参数
     */
    void set_mixing_beta(double beta);
    
    /**
     * @brief 获取混合参数
     */
    double get_mixing_beta() const;
};

#endif // PULAY_MIXER_H
