/**
 * @brief Kernels for computing t-SNE repulsive forces with barnes hut approximation.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_BH_REP_FORCES_H_
#define SRC_INCLUDE_KERNELS_BH_REP_FORCES_H_

#include "include/common.h"

#ifdef __KEPLER__
#define REPULSIVE_FORCES_THREADS 1024
#define REPULSIVE_FORCE_BLOCKS 2
#else
#define REPULSIVE_FORCES_THREADS 256
#define REPULSIVE_FORCES_BLOCKS 5
#endif

namespace tsnecuda {
namespace bh {



/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(REPULSIVE_FORCES_THREADS, REPULSIVE_FORCES_BLOCKS)
void ForceCalculationKernel(volatile int * __restrict__ errd,
                                          volatile float * __restrict__ x_vel_device,
                                          volatile float * __restrict__ y_vel_device,
                                          volatile float * __restrict__ normalization_vec_device,
                                          const int * __restrict__ cell_sorted,
                                          const int * __restrict__ children,
                                          const float * __restrict__ cell_mass,
                                          volatile float * __restrict__ x_pos_device,
                                          volatile float * __restrict__ y_pos_device,
                                          const float theta,
                                          const float epsilon,
                                          const uint32_t num_nodes,
                                          const uint32_t num_points);

void ComputeRepulsiveForces(thrust::device_vector<int> &errd,
                                          thrust::device_vector<float> &repulsive_forces,
                                          thrust::device_vector<float> &normalization_vec,
                                          thrust::device_vector<int> &cell_sorted,
                                          thrust::device_vector<int> &children,
                                          thrust::device_vector<float> &cell_mass,
                                          thrust::device_vector<float> &points,
                                          const float theta,
                                          const float epsilon,
                                          const uint32_t num_nodes,
                                          const uint32_t num_points,
                                          const uint32_t num_blocks);
 
}
}

#endif