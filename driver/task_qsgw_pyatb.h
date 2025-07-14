#ifndef TASK_QSGW_pyatb_H
#define TASK_QSGW_pyatb_H
#include <map>
#include <vector>

#include "complexmatrix.h"
#include "vector3_order.h"

// 声明 qsgw_pyatb 计算任务的函数
void task_qsgw_pyatb(std::map<Vector3_Order<double>, ComplexMatrix>& sinvS);
#endif  // TASK_qsgw_pyatb_H
