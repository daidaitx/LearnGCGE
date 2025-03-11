/**
 * @brief 读取MTX文件的工具
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-02-18
 */

#ifndef __BEF_GCGE_SRC_IO_MMIO_READER_H__
#define __BEF_GCGE_SRC_IO_MMIO_READER_H__

#ifdef __cplusplus
extern "C"{
#endif

#include "app_ccs.h"

/**
 * @brief 根据MTX文件创建CCS格式的稀疏矩阵
 * 
 * @param ccs_matA GCGE中定义个CCS矩阵的结构体对象
 * @param file_matrix mtx文件的路径
 * @return int 错误码
 */
int CreateCCSFromMTX(CCSMAT *ccs_matA, char* file_matrix);

#ifdef __cplusplus
}
#endif
#endif