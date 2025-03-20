/**
 * @file ops_multi_grid.c
 * 
 * Chinese Encoding Format: UTF-8
 * 
 * Updated on 2025-03-20 by 吴卓轩
 * 
 * @brief 实现了多重网格算法中不同网格层之间的数据传递操作
 * 
 * 本文档的两个函数实现了多重网格算法中不同网格层之间的数据传递操作，包括：
 * 		插值延拓：将向量从粗网格插值到细网格
 * 		限制投影：将向量从细网格投影到粗网格
 * 支持单向量 DefaultVecFromItoJ 和多向量 DefaultMultiVecFromItoJ 两种操作模式
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ops.h"

/**
 * @todo 本代码库未使用 DefaultVecFromItoJ 函数
 * @todo 在 ops_lin_sol.c 中使用了 DefaultMultiVecFromItoJ 函数，但似乎没有真的调用，这需要检查文档 ops_lin_sol.c 的具体实现
 */

/**
 * @brief 【从粗层到细层的插值】和【从细层到粗层的限制】的默认函数（单向量版本）
 * 
 * @param P_array [in]   转移矩阵数组，P_array[i]表示从第 i+1 层到第 i 层的转移矩阵
 * @param level_i [in]  【源层】的层号
 * @param level_j [in]  【目标层】的层号
 * @param vec_i   [in]  【源层】的向量
 * @param vec_j   [out] 【目标层】的向量
 * 
 * @param vec_ws  [out] 向量的工作空间，用于存储中间结果
 * @param ops     [in]  集合了线性代数操作
 * 
 * @note 层数编号从 0 开始，最细网格层号为 0，最粗网格层号为 L-1
 */
void DefaultVecFromItoJ(void **P_array, int level_i, int level_j, void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops) {
	/**
	 * @param from_vec 【源层】的向量
	 * @param to_vec   【目标层】的向量
	 * 
	 * @param k        循环变量
	 */
	void *from_vec, *to_vec;
	int k = 0;

	// 判定【源层】和【目标层】的层号大小，以确定是插值还是限制
	if (level_i > level_j) {
		// 如果【源层】的层号大于【目标层】的层号，则从粗层到细层，插值延拓
		// 计算 vec_j = P[j] * P[j+1] * ... * P[i-2] * P[i-1] * vec_i
		/**
		 * 层号顺序：j < i
		 * 定义：
		 *     vec_ws[i] = vec_i
		 *     vec_ws[j] = vec_j
		 * 计算：
		 *     vec_ws[i-1] = P[i-1] * vec_ws[i  ];
		 *     vec_ws[i-2] = P[i-2] * vec_ws[i-1];
		 *     ...
		 *     vec_ws[j+1] = P[j+1] * vec_ws[j+2];
		 *     vec_ws[j  ] = P[j  ] * vec_ws[j+1];
		 */
		for (k = level_i; k > level_j; --k) {
			if (k == level_i) {
				from_vec = vec_i;
			} else {
				from_vec = vec_ws[k];
			}
			if (k == level_j + 1) {
				to_vec = vec_j;
			} else {
				to_vec = vec_ws[k - 1];
			}
			ops->MatDotVec(P_array[k - 1], from_vec, to_vec, ops);
		}
	} else if (level_i < level_j) {
		// 如果【源层】的层号小于【目标层】的层号，则从细层到粗层，限制投影
		// 计算 vec_j = P[j-1]' * P[j-2]' * ... * P[i+1]' * P[i]' * vec_i
		/**
		 * 层号顺序：i < j
		 * 定义：
		 *     vec_ws[i] = vec_i
		 *     vec_ws[j] = vec_j
		 * 计算：
		 *     vec_ws[i+1] = P[i  ]' * vec_ws[i  ];
		 *     vec_ws[i+2] = P[i+1]' * vec_ws[i+1];
		 *     ...
		 *     vec_ws[j-1] = P[j-2]' * vec_ws[j-2];
		 *     vec_ws[j  ] = P[j-1]' * vec_ws[j-1];
		 */
		for (k = level_i; k < level_j; ++k) {
			if (k == level_i) {
				from_vec = vec_i;
			} else {
				from_vec = vec_ws[k];
			}
			if (k == level_j - 1) {
				to_vec = vec_j;
			} else {
				to_vec = vec_ws[k + 1];
			}
			ops->MatTransDotVec(P_array[k], from_vec, to_vec, ops);
		}
	} else {
		// 若源层和目标层为同层，利用axpby直接赋值：vec_j = vec_i;
		ops->VecAxpby(1.0, vec_i, 0.0, vec_j, ops);
	}
	return;
}

/**
 * @brief 【从粗层到细层的插值】和【从细层到粗层的限制】的默认函数（多向量版本）
 * 
 * @param P_array      [in]  转移矩阵数组，P_array[i]表示从第 i+1 层到第 i 层的转移矩阵
 * @param level_i      [in]  【源层】的层号
 * @param level_j      [in]  【目标层】的层号
 * @param multi_vec_i  [in]  【源层】的多向量
 * @param multi_vec_j  [out] 【目标层】的多向量
 * 
 * @param startIJ      [in]  多向量的起始索引
 * @param endIJ        [in]  多向量的终止索引
 * 
 * @param multi_vec_ws [out] 多向量的工作空间，用于存储中间结果
 * @param ops          [in]  集合了线性代数操作
 * 
 * @note 层数编号从 0 开始，最细网格层号为 0，最粗网格层号为 L-1
 */
void DefaultMultiVecFromItoJ(void **P_array, int level_i, int level_j, void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ, void ***multi_vec_ws, struct OPS_ *ops)
{
	/**
	 * @param from_vecs 【源层】的多向量
	 * @param to_vecs   【目标层】的多向量
	 * 
	 * @param k         循环变量
	 * 
	 * @param start     多向量的起始索引
	 * @param end       多向量的终止索引
	 */
	void **from_vecs, **to_vecs;
	int k = 0, start[2], end[2];

	// 判定【源层】和【目标层】的层号大小，以确定是插值还是限制
	if (level_i > level_j) {
		// 如果【源层】的层号大于【目标层】的层号，则从粗层到细层，插值延拓
		// 计算 vec_j = P[j] * P[j+1] * ... * P[i-2] * P[i-1] * vec_i
		/**
		 * 层号顺序：j < i
		 * 定义：
		 *     multi_vec_ws[i] = multi_vec_i
		 *     multi_vec_ws[j] = multi_vec_j
		 * 计算：
		 *     multi_vec_ws[i-1] = P[i-1] * multi_vec_ws[i  ];
		 *     multi_vec_ws[i-2] = P[i-2] * multi_vec_ws[i-1];
		 *     ...
		 *     multi_vec_ws[j+1] = P[j+1] * multi_vec_ws[j+2];
		 *     multi_vec_ws[j  ] = P[j  ] * multi_vec_ws[j+1];
		 */
		for (k = level_i; k > level_j; --k) {
			if (k == level_i) {
				from_vecs = multi_vec_i;
				start[0] = startIJ[0];
				end[0] = endIJ[0];
			} else {
				from_vecs = multi_vec_ws[k];
				start[0] = 0;
				end[0] = endIJ[0] - startIJ[0];
			}
			if (k == level_j + 1) {
				to_vecs = multi_vec_j;
				start[1] = startIJ[1];
				end[1] = endIJ[1];
			} else {
				to_vecs = multi_vec_ws[k - 1];
				start[1] = 0;
				end[1] = endIJ[0] - startIJ[0];
			}
			ops->MatDotMultiVec(P_array[k - 1], from_vecs, to_vecs, start, end, ops);
		}
	} else if (level_i < level_j) {
		// 如果【源层】的层号小于【目标层】的层号，则从细层到粗层，限制投影
		// 计算 vec_j = P[j-1]' * P[j-2]' * ... * P[i+1]' * P[i]' * vec_i
		/**
		 * 层号顺序：i < j
		 * 定义：
		 *     multi_vec_ws[i] = multi_vec_i
		 *     multi_vec_ws[j] = multi_vec_j
		 * 计算：
		 *     multi_vec_ws[i+1] = P[i  ]' * multi_vec_ws[i  ];
		 *     multi_vec_ws[i+2] = P[i+1]' * multi_vec_ws[i+1];
		 *     ...
		 *     multi_vec_ws[j-1] = P[j-2]' * multi_vec_ws[j-2];
		 *     multi_vec_ws[j  ] = P[j-1]' * multi_vec_ws[j-1];
		 */
		for (k = level_i; k < level_j; ++k) {
			if (k == level_i) {
				from_vecs = multi_vec_i;
				start[0] = startIJ[0];
				end[0] = endIJ[0];
			} else {
				from_vecs = multi_vec_ws[k];
				start[0] = 0;
				end[0] = endIJ[0] - startIJ[0];
			}
			if (k == level_j - 1) {
				to_vecs = multi_vec_j;
				start[1] = startIJ[1];
				end[1] = endIJ[1];
			} else {
				to_vecs = multi_vec_ws[k + 1];
				start[1] = 0;
				end[1] = endIJ[0] - startIJ[0];
			}
			ops->MatTransDotMultiVec(P_array[k], from_vecs, to_vecs, start, end, ops);
		}
	} else {
		// 若源层和目标层为同层，利用axpby直接赋值：multi_vec_j = multi_vec_i;
		ops->MultiVecAxpby(1.0, multi_vec_i, 0.0, multi_vec_j, startIJ, endIJ, ops);
	}
	return;
}
