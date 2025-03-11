/**
 * @brief 定义求解器需要的错误码
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-03-11
 */

#ifndef __GCGE_INCLUDE_ERRCODE_H__
#define __GCGE_INCLUDE_ERRCODE_H__

#ifdef __cplusplus
extern "C" {
#endif

// 兼容 C 和 C++ 的 enum 定义
typedef enum {
    GCGE_SUCCESS = 0,  // 成功
    GCGE_ERR_NULLPTR = 101,  // 空指针错误
    GCGE_ERR_MEM = 102,   // 内存分配错误
    GCGE_ERR_FILE = 103,     // 文件读写错误
    GCGE_ERR_INPUT = 104,    // 输入参数错误
    GCGE_ERR_OUTPUT = 105,   // 输出参数错误
    GCGE_ERR_INTERNAL = 106, // 内部错误
    GCGE_ERR_UNKNOWN = 107,   // 未知错误

    GCGE_ERR_NOT_CONVERGED  = 201,  /* solver did not converge */
} GcgeErrCode;

#ifdef __cplusplus
}
#endif

#endif // __GCGE_INCLUDE_ERRCODE_H__
