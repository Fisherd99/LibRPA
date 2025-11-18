/*!
 * @file      analycont.h
 * @brief     Utilities for analytic continuation
 * @author    Min-Ye Zhang
 * @date      2024-04-23
 */
#pragma once
#include <vector>

#include "base_utility.h"
#include "complexmatrix.h"  
namespace LIBRPA
{

class AnalyContPade
{
private:
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;

public:
    AnalyContPade(int n_pars_in,
                  const std::vector<cplxdb> &xs,
                  const std::vector<cplxdb> &data);

    cplxdb get(const cplxdb &x) const;
};

// 新增的Nevanlinna类声明
class AnalyContNevanlinna 
{
private:
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;
    std::vector<cplxdb> phis_;
    std::vector<ComplexMatrix> abcds_;

public:
    AnalyContNevanlinna(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data);
    cplxdb get(const cplxdb &x) const;
};

// 新增的自能专用Nevanlinna类声明
class AnalyContNevanlinnaSelfEnergy 
{
private:
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;
    std::vector<cplxdb> phis_;
    std::vector<ComplexMatrix> abcds_;
    bool apply_physical_constraints_;

public:
    AnalyContNevanlinnaSelfEnergy(int n_pars_in, const std::vector<cplxdb> &xs, 
                                 const std::vector<cplxdb> &data, bool apply_constraints = true);
    cplxdb get(const cplxdb &x) const;
};

} // namespace LIBRPA