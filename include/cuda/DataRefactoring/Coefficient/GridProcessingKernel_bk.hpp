/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_GRID_PROCESSING_KERNEL_2_TEMPLATE
#define MGRAD_CUDA_GRID_PROCESSING_KERNEL_2_TEMPLATE

#include "../../CommonInternal.h"
#include "GPKFunctor.h"
#include "GridProcessingKernel3D.h"

#include "../../Functor.h"
#include "../../AutoTuners/AutoTuner.h"
#include "../../Task.h"
#include "../../DeviceAdapters/DeviceAdapter.h"

namespace mgard_cuda {
template <DIM D_GLOBAL, DIM D_LOCAL, typename T, SIZE R, SIZE C, SIZE F, 
          bool INTERPOLATION, bool CALC_COEFF, int TYPE, typename DeviceType>
class GpkReoFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT GpkReoFunctor(SubArray<1, SIZE, DeviceType> shape, 
                            SubArray<1, SIZE, DeviceType> shape_c,  
                            DIM unprocessed_n, SubArray<1, DIM, DeviceType> unprocessed_dims, 
                            DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                            SubArray<1, T, DeviceType> ratio_r, 
                            SubArray<1, T, DeviceType> ratio_c, 
                            SubArray<1, T, DeviceType> ratio_f,
                            SubArray<D_GLOBAL, T, DeviceType> v,
                            SubArray<D_GLOBAL, T, DeviceType> w,
                            SubArray<D_GLOBAL, T, DeviceType> wf,
                            SubArray<D_GLOBAL, T, DeviceType> wc,
                            SubArray<D_GLOBAL, T, DeviceType> wr,
                            SubArray<D_GLOBAL, T, DeviceType> wcf,
                            SubArray<D_GLOBAL, T, DeviceType> wrf,
                            SubArray<D_GLOBAL, T, DeviceType> wrc,
                            SubArray<D_GLOBAL, T, DeviceType> wrcf):
                            shape(shape), shape_c(shape_c),
                            unprocessed_n(unprocessed_n), unprocessed_dims(unprocessed_dims),
                            curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f), 
                            ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f),
                            v(v), w(w), wf(wf), wc(wc), wr(wr), wcf(wcf),
                            wrf(wrf), wrc(wrc), wrcf(wrcf)
                            {
                              Functor<DeviceType>();
                            }

  MGARDm_EXEC void
  Operation1() {

    // bool debug = false;
    // if (blockIdx.x == 0 && blockIdx.y ==0 && blockIdx.z == 0 &&
    //     threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) debug =
    //     false;

    // volatile clock_t start = 0;
    // volatile clock_t end = 0;
    // volatile unsigned long long sum_time = 0;

    threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

  
    in_next = true;

    T *sm = SharedMemory<T>();
    ldsm1 = (F/2) * 2 + 1;
    ldsm2 = (C/2) * 2 + 1;

    v_sm = sm; sm += ((F/2) * 2 + 1) * ((C/2) * 2 + 1) * ((R/2) * 2 + 1);
    ratio_f_sm = sm; sm += (F/2) * 2;
    ratio_c_sm = sm; sm += (C/2) * 2;
    ratio_r_sm = sm; sm += (R/2) * 2;

    sm_size = (SIZE*)sm;
    shape_sm = sm_size; sm_size += D_GLOBAL;
    shape_c_sm = sm_size; sm_size += D_GLOBAL;
    // ldvs_sm = sm_size; sm_size += D_GLOBAL;
    // ldws_sm = sm_size; sm_size += D_GLOBAL;
    sm = (T*)sm_size;

    sm_dim = (DIM*)sm;
    unprocessed_dims_sm = sm_dim; sm_dim += D_GLOBAL;
    sm = (T*)sm_dim;

    if (threadId < D_GLOBAL) {
      shape_sm[threadId] = *shape(threadId);
      shape_c_sm[threadId] = *shape_c(threadId);
      // ldvs_sm[threadId] = ldvs[threadId];
      // ldws_sm[threadId] = ldws[threadId];
    }

    if (threadId < unprocessed_n) {
      unprocessed_dims_sm[threadId] = *unprocessed_dims(threadId);
    }
    __syncthreads();

    for (DIM d = 0; d < D_GLOBAL; d++)
      idx[d] = 0;

    nr = shape_sm[curr_dim_r];
    nc = shape_sm[curr_dim_c];
    nf = shape_sm[curr_dim_f];

    nr_c = shape_c_sm[curr_dim_r];
    nc_c = shape_c_sm[curr_dim_c];
    nf_c = shape_c_sm[curr_dim_f];

    if (D_LOCAL < 3) {
      nr = 1;
      nr_c = 1;
    }
    if (D_LOCAL < 2) {
      nc = 1;
      nc_c = 1;
    }

    r = blockIdx.z * blockDim.z;
    c = blockIdx.y * blockDim.y;
    SIZE bidx = blockIdx.x;
    SIZE firstD = div_roundup(shape_sm[0] - 1, blockDim.x);
    f = (bidx % firstD) * blockDim.x;

    bidx /= firstD;

    // if (debug) printf("n: %d %d %d rcf: %d %d %d\n", nr, nc, nf, r, c, f);
    rest_r = nr - r;
    rest_c = nc - c;
    rest_f = nf - f;

    nr_p = nr;
    nc_p = nc;
    nf_p = nf;

    rest_r_p = rest_r;
    rest_c_p = rest_c;
    rest_f_p = rest_f;

    if (nr % 2 == 0) {
      nr_p = nr + 1;
      rest_r_p = nr_p - r;
    }
    if (nc % 2 == 0) {
      nc_p = nc + 1;
      rest_c_p = nc_p - c;
    }
    if (nf % 2 == 0) {
      nf_p = nf + 1;
      rest_f_p = nf_p - f;
    }

    for (DIM d = 0; d < D_GLOBAL; d++) {
      if (D_LOCAL == 3 && d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        idx[d] = bidx % shape_sm[d];
        bidx /= shape_sm[d];
        if (idx[d] >= shape_c_sm[d])
          in_next = false;
      }
      if (D_LOCAL == 2 && d != curr_dim_c && d != curr_dim_f) {
        idx[d] = bidx % shape_sm[d];
        bidx /= shape_sm[d];
        if (idx[d] >= shape_c_sm[d])
          in_next = false;
      }
    }

    skip = 0;
    #pragma unroll 1
    for (DIM t = 0; t < D_GLOBAL; t++) {
      for (DIM k = 0; k < unprocessed_n; k++) {
        if (t == unprocessed_dims_sm[k] &&
            (shape_sm[t] % 2 == 1 && idx[t] % 2 == 1 ||
             shape_sm[t] % 2 == 0 && idx[t] % 2 == 1 &&
                 idx[t] != shape_sm[t] - 1)) {
          skip = 1;
        }
      }
    }

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //   printf("total_idx_sm: %d %d %d %d (skip: %d)\n", idx[3], idx[2], idx[1],
    //   idx[0], skip);
    // }
    // }

    // LENGTH other_offset_v = get_idx<D_GLOBAL>(ldvs_sm, idx);
    // LENGTH other_offset_w = get_idx<D_GLOBAL>(ldws_sm, idx);

    v.offset(idx);
    w.offset(idx);
    wf.offset(idx);
    wc.offset(idx);
    wr.offset(idx);
    wcf.offset(idx);
    wrf.offset(idx);
    wrc.offset(idx);
    wrcf.offset(idx);
    // dv = dv + other_offset_v;
    // dw = dw + other_offset_w;
    // dwr = dwr + other_offset_w;
    // dwc = dwc + other_offset_w;
    // dwf = dwf + other_offset_w;
    // dwrf = dwrf + other_offset_w;
    // dwrc = dwrc + other_offset_w;
    // dwcf = dwcf + other_offset_w;
    // dwrcf = dwrcf + other_offset_w;

    // if (idx[1] == 0 && idx[2] == 0 && threadId == 0) printf("before wf(00): %lld, w: %lld\n", wf((IDX)0), w((IDX)0));
    // if (idx[1] == 1 && idx[2] == 0 && threadId == 0) printf("before wf(01): %lld, w: %lld\n", wf((IDX)0), w((IDX)0));

    
    if (TYPE == 2) {
      // if (idx[1] == 0 && idx[2] == 0 && threadId == 0) printf("before wf: %lld, w: %lld\n", wf((IDX)0), w((IDX)0));
      wf = w;
      wcf = wc;
      wrf = wr;
      wrcf = wrc;
      // if (idx[1] == 0 && idx[2] == 0 && threadId == 0) printf("after wf: %lld, w: %lld\n", wf((IDX)0), w((IDX)0));
    }
    __syncthreads();

    

    // if (!skip)
    {
      r_sm = threadIdx.z;
      c_sm = threadIdx.y;
      f_sm = threadIdx.x;

      r_sm_ex = (R/2) * 2;
      c_sm_ex = (C/2) * 2;
      f_sm_ex = (F/2) * 2;

      r_gl = r + r_sm;
      r_gl_ex = r + (R/2) * 2;
      c_gl = c + c_sm;
      c_gl_ex = c + (C/2) * 2;
      f_gl = f + f_sm;
      f_gl_ex = f + (F/2) * 2;

      //  __syncthreads();
      // if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
      //   //printf("setting zeros\n");
      //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
      //     for (int j = 0; j < (C/2) * 2 + 1; j++) {
      //       for (int k = 0; k < (F/2) * 2 + 1; k++) {
      //         v_sm[get_idx(ldsm1, ldsm2, i, j, k)] = 0.0;
      //       }
      //     }
      //   }
      //   //printf("done zeros\n");
      // }
      //  __syncthreads();
      /* Load v */
      // loading extra rules
      // case 1: input = odd (non-padding required)
      //    case 1.a: block size < rest (need to load extra);
      //    case 1.b: block size > rest (NO need to load extra);
      // case 2: input = even (padding requried)
      //    case 2.a: block size < rest (need to load extra);
      //    case 2.b: block size >= rest (NO need to load extra, but need
      //    padding);

      // Load from dv
      if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {

        // load cubic
        // asm volatile("membar.cta;");
        // start = clock64();
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            *v(r_gl, c_gl, f_gl);
        // if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0) {
        //   printf("load (%d %d %d) %f <- %d+(%d %d %d) (ld: %d %d)\n",
        //           r_sm, c_sm, f_sm,
        //           dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)],
        //           other_offset_v+r_gl, c_gl, f_gl, lddv1, lddv2);
        // }
        if (r_sm == 0) {
          if (rest_r > (R/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] =
                *v(r_gl_ex, c_gl, f_gl);
          }
        }
        if (c_sm == 0) {
          if (rest_c > (C/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
                *v(r_gl, c_gl_ex, f_gl);
          }
        }
        if (f_sm == 0) {
          if (rest_f > (F/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] =
                *v(r_gl, c_gl, f_gl_ex);
          }
        }
        if (c_sm == 0 && f_sm == 0) {
          if (rest_c > (C/2) * 2 && rest_f > (F/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
                *v(r_gl, c_gl_ex, f_gl_ex);
          }
        }
        if (r_sm == 0 && f_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_f > (F/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
                *v(r_gl_ex, c_gl, f_gl_ex);
          }
        }
        if (r_sm == 0 && c_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_c > (C/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
                *v(r_gl_ex, c_gl_ex, f_gl);
          }
        }
        if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_c > (C/2) * 2 && rest_f > (F/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
                *v(r_gl_ex, c_gl_ex, f_gl_ex);
          }
        }
      }

      __syncthreads();

      // apply padding is necessary
      if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {

        // printf("load main[%d %d %d]:%f --> [%d %d %d] (%d %d %d)\n", r_gl,
        // c_gl, f_gl,
        //     dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)], r_sm, c_sm, f_sm, nr,
        //     nc, nf);

        // asm volatile("membar.cta;");
        // start = clock64() - start;
        // printf("[load main] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
        // blockIdx.y, blockIdx.x, start); start = clock64();

        // load extra surface

        if (r_sm == 0) {
          if (rest_r > (R/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] =
            //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)];
            // printf("load-r[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl, f_gl,
            //   dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)], r_sm_ex, c_sm,
            //   f_sm);
          } else if (nr % 2 == 0) {
            // if (r == 16 && c == 0 && f == 0) {
            //   printf("padding (%d %d %d) %f <- (%f %f %f)\n", rest_r_p - 1,
            //   c_sm, f_sm,
            //         v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)], rest_r
            //         - 1, c_sm, f_sm);
            //   padded = true;
            //   aa = v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)];
            //   bb = v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)];
            // }
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)];
          }
        }

        if (c_sm == 0) {
          if (rest_c > (C/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
            //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
            // printf("load-c[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex, f_gl,
            //   dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)], r_sm, c_sm_ex,
            //   f_sm);
          } else if (nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm)];
          }
        }

        if (f_sm == 0) {
          if (rest_f > (F/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] =
            //     dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)];
            // printf("load-f[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl, f_gl_ex,
            //   dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)], r_sm, c_sm,
            //   f_sm_ex);
          } else if (nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f - 1)];
          }
        }

        // load extra edges
        if (c_sm == 0 && f_sm == 0) {
          if (rest_c > (C/2) * 2 && rest_f > (F/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
            //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
            // printf("load-cf[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex,
            // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)], r_sm,
            // c_sm_ex, f_sm_ex);
          } else if (rest_c <= (C/2) * 2 && rest_f <= (F/2) * 2 && nc % 2 == 0 &&
                     nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)];
          } else if (rest_c > (C/2) * 2 && rest_f <= (F/2) * 2 && nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f - 1)];
          } else if (rest_c <= (C/2) * 2 && rest_f > (F/2) * 2 && nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm_ex)];
          }
        }

        if (r_sm == 0 && f_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_f > (F/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
            //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
            // printf("load-rf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl,
            // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)],
            // r_sm_ex, c_sm, f_sm_ex);
          } else if (rest_r <= (R/2) * 2 && rest_f <= (F/2) * 2 && nr % 2 == 0 &&
                     nf % 2 == 0) {
            // printf("padding (%d %d %d) <- (%d %d %d)\n", rest_r_p - 1, c_sm,
            // rest_f_p - 1, rest_r - 1, c_sm, rest_f - 1);
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)];
          } else if (rest_r > (R/2) * 2 && rest_f <= (F/2) * 2 && nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f - 1)];
          } else if (rest_r <= (R/2) * 2 && rest_f > (F/2) * 2 && nr % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm_ex)];
          }
        }

        if (r_sm == 0 && c_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_c > (C/2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
            //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
            // printf("load-rc[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
            // f_gl, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)], r_sm_ex,
            // c_sm_ex, f_sm);
          } else if (rest_r <= (R/2) * 2 && rest_c <= (C/2) * 2 && nr % 2 == 0 &&
                     nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)];
            // printf("padding (%d %d %d) <- (%d %d %d): %f\n", rest_r_p - 1,
            // rest_c_p - 1, f_sm, rest_r - 1, rest_c - 1, f_sm,
            // v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)]);
          } else if (rest_r > (R/2) * 2 && rest_c <= (C/2) * 2 && nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm)];
          } else if (rest_r <= (R/2) * 2 && rest_c > (C/2) * 2 && nr % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm)];
          }
        }
        // load extra vertex

        if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
          if (rest_r > (R/2) * 2 && rest_c > (C/2) * 2 && rest_f > (F/2) * 2) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
                *v(r_gl_ex, c_gl_ex, f_gl_ex);
            // printf("load-rcf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
            // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)],
            // r_sm_ex, c_sm_ex, f_sm_ex);
          } else if (rest_r <= (R/2) * 2 && rest_c <= (C/2) * 2 && rest_f <= (F/2) * 2 &&
                     nr % 2 == 0 && nc % 2 == 0 && nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                         rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)];
          } else if (rest_r > (R/2) * 2 && rest_c > (C/2) * 2 && rest_f <= (F/2) * 2 &&
                     nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f - 1)];
          } else if (rest_r > (R/2) * 2 && rest_c <= (C/2) * 2 && rest_f > (F/2) * 2 &&
                     nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm_ex)];
          } else if (rest_r > (R/2) * 2 && rest_c <= (C/2) * 2 && rest_f <= (F/2) * 2 &&
                     nc % 2 == 0 && nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, rest_f - 1)];
          } else if (rest_r <= (R/2) * 2 && rest_c > (C/2) * 2 && rest_f > (F/2) * 2 &&
                     nr % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm_ex)];
          } else if (rest_r <= (R/2) * 2 && rest_c > (C/2) * 2 && rest_f <= (F/2) * 2 &&
                     nr % 2 == 0 && nf % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, rest_f - 1)];
          } else if (rest_r <= (R/2) * 2 && rest_c <= (C/2) * 2 && rest_f > (F/2) * 2 &&
                     nr % 2 == 0 && nc % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)] =
                v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm_ex)];
          }
        }

        // asm volatile("membar.cta;");
        // start = clock64() - start;
        // printf("[load extra] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
        // blockIdx.y, blockIdx.x, start); start = clock64();

        // load dist
        if (c_sm == 0 && f_sm == 0 && r_sm < rest_r_p - 2) {
          // printf("%d/%d load %f\n", r_sm, rest_r - 2, dratio_r[r + r_sm]);
          ratio_r_sm[r_sm] = *ratio_r(r + r_sm);
          // if (nr % 2 == 0 && (R/2) * 2 + 1 >= rest_r_p && r_sm == 0) {
          //   ratio_r_sm[rest_r_p - 3] = 0.5;
          // }
        }
        if (r_sm == 0 && f_sm == 0 && c_sm < rest_c_p - 2) {
          ratio_c_sm[c_sm] = *ratio_c(c + c_sm);
          // if (nc % 2 == 0 && (C/2) * 2 + 1 >= rest_c_p && c_sm == 0) {
          //   ratio_c_sm[rest_c_p - 3] = 0.5;
          // }
        }
        if (c_sm == 0 && r_sm == 0 && f_sm < rest_f_p - 2) {
          ratio_f_sm[f_sm] = *ratio_f(f + f_sm);
          // if (nf % 2 == 0 && (F/2) * 2 + 1 >= rest_f_p && f_sm == 0) {
          //   ratio_f_sm[rest_f_p - 3] = 0.5;
          // }
        }

        // if (r == 0 && c == 0 && f == 0 && r_sm == 0 && c_sm == 0 && f_sm == 0)
        // {
        //   printf("ratio:");
        //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
        //     printf("%2.2f ", ratio_r_sm[i]);
        //   }
        //   printf("\n");
        // }

      } // restrict boundary

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[load ratio] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();

      // __syncthreads();
      // // debug print
      // if (debug) {
      //   printf("in config: %d %d %d (%d %d %d)\n", (R/2), (C/2), (F/2), r,c,f);
      //   printf("rest_p: %d %d %d\n", rest_r_p, rest_c_p, rest_f_p);
      //   bool print = false;
      //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
      //     for (int j = 0; j < (C/2) * 2 + 1; j++) {
      //       for (int k = 0; k < (F/2) * 2 + 1; k++) {
      //         // if (abs(v_sm[get_idx(ldsm1, ldsm2, i, j, k)]) > 10000) {
      //           // print = true;
      //           // printf("(block %d %d %d) %2.2f \n", r,c,f,
      //           v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
      //         // printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
      //         // }
      //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
      //       }
      //       printf("\n");
      //     }
      //     printf("\n");
      //   }
      // }
      __syncthreads();

      if (!w.isNull() && threadId < (R/2) * (C/2) * (F/2)) {
        r_sm = (threadId / ((C/2) * (F/2))) * 2;
        c_sm = ((threadId % ((C/2) * (F/2))) / (F/2)) * 2;
        f_sm = ((threadId % ((C/2) * (F/2))) % (F/2)) * 2;
        r_gl = r / 2 + threadId / ((C/2) * (F/2));
        c_gl = c / 2 + threadId % ((C/2) * (F/2)) / (F/2);
        f_gl = f / 2 + threadId % ((C/2) * (F/2)) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: (%d %d %d) <- %f (%d %d %d)\n",
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
      }

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[store coarse] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();
      int base = 0;
      // printf("TYPE =%d \n", TYPE);
      // printf("%d == %d && %llu >= %d && %llu < %d\n", r + (R/2) * 2, nr_p - 1,
      // threadId, base, threadId, base + (C/2) * (F/2));

      if (!w.isNull() && r + (R/2) * 2 == nr_p - 1 && threadId >= base &&
          threadId < base + (C/2) * (F/2)) {
        r_sm = (R/2) * 2;
        c_sm = ((threadId - base) / (F/2)) * 2;
        f_sm = ((threadId - base) % (F/2)) * 2;
        r_gl = r / 2 + (R/2);
        c_gl = c / 2 + (threadId - base) / (F/2);
        f_gl = f / 2 + (threadId - base) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }

            // if (idx[1] == 1 && idx[2] == 0) {
              *w(r_gl, c_gl, f_gl) = res;
              // printf("w(cf)-store: (%d %d %d) <- %f (%d %d %d)\n", 
              // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
              // r_sm, c_sm, f_sm);
            // }
          }
        }
      }

      base += (C/2) * (F/2); // ROUND_UP_WARP((C/2) * (F/2)) * WARP_SIZE;
      if (!w.isNull() && c + (C/2) * 2 == nc_p - 1 && threadId >= base &&
          threadId < base + (R/2) * (F/2)) {
        r_sm = ((threadId - base) / (F/2)) * 2;
        c_sm = (C/2) * 2;
        f_sm = ((threadId - base) % (F/2)) * 2;
        r_gl = r / 2 + (threadId - base) / (F/2);
        c_gl = c / 2 + (C/2);
        f_gl = f / 2 + (threadId - base) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(rf)-store: (%d %d %d) <- %f (%d %d %d)\n",
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
        // printf("(%d %d %d) (%d %d %d) %f\n",
        //         r_sm, c_sm, f_sm, r_gl, c_gl, f_gl, dwork[get_idx(lddv1, lddv2,
        //         r_gl, c_gl, f_gl)]);
      }

      base += (R/2) * (F/2); // ROUND_UP_WARP((R/2) * (F/2)) * WARP_SIZE;
      // printf("%d %d\n", base,  threadId);
      if (!w.isNull() && f + (F/2) * 2 == nf_p - 1 && threadId >= base &&
          threadId < base + (R/2) * (C/2)) {
        r_sm = ((threadId - base) / (C/2)) * 2;
        c_sm = ((threadId - base) % (C/2)) * 2;
        f_sm = (F/2) * 2;
        r_gl = r / 2 + (threadId - base) / (C/2);
        c_gl = c / 2 + (threadId - base) % (C/2);
        f_gl = f / 2 + (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(rc)-store: (%d %d %d) <- %f (%d %d %d)\n", 
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
      }

      base += (R/2) * (C/2); // ROUND_UP_WARP((R/2) * (C/2)) * WARP_SIZE;
      // load extra edges
      if (!w.isNull() && c + (C/2) * 2 == nc_p - 1 && f + (F/2) * 2 == nf_p - 1 &&
          threadId >= base && threadId < base + (R/2)) {
        r_sm = (threadId - base) * 2;
        c_sm = (C/2) * 2;
        f_sm = (F/2) * 2;
        r_gl = r / 2 + threadId - base;
        c_gl = c / 2 + (C/2);
        f_gl = f / 2 + (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(r)-store: (%d %d %d) <- %f (%d %d %d)\n",
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
      }

      base += (R/2); // ROUND_UP_WARP((R/2)) * WARP_SIZE;
      // if (TYPE == 2) printf("%d %d, %d, %llu, %d\n",dw == NULL, f + (F/2) * 2, nf_p
      // - 1, threadId, (C/2));
      if (!w.isNull() && r + (R/2) * 2 == nr_p - 1 && f + (F/2) * 2 == nf_p - 1 &&
          threadId >= base && threadId < base + (C/2)) {
        r_sm = (R/2) * 2;
        c_sm = (threadId - base) * 2;
        f_sm = (F/2) * 2;
        r_gl = r / 2 + (R/2);
        c_gl = c / 2 + threadId - base;
        f_gl = f / 2 + (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(c)-store: (%d %d %d) <- %f (%d %d %d)\n", 
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
        // printf("store[%d %d %d]: %f\n", r_sm, c_sm, f_sm, v_sm[get_idx(ldsm1,
        // ldsm2, r_sm, c_sm, f_sm)]);
      }

      base += (C/2); // ROUND_UP_WARP((C/2)) * WARP_SIZE;
      if (!w.isNull() && r + (R/2) * 2 == nr_p - 1 && c + (C/2) * 2 == nc_p - 1 &&
          threadId >= base && threadId < base + (F/2)) {
        r_sm = (R/2) * 2;
        c_sm = (C/2) * 2;
        f_sm = (threadId - base) * 2;
        r_gl = r / 2 + (R/2);
        c_gl = c / 2 + (C/2);
        f_gl = f / 2 + threadId - base;
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(f)-store: (%d %d %d) <- %f (%d %d %d)\n",
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
      }
      base += (F/2); // ROUND_UP_WARP((F/2)) * WARP_SIZE;
      // // load extra vertex
      if (!w.isNull() && r + (R/2) * 2 == nr_p - 1 && c + (C/2) * 2 == nc_p - 1 &&
          f + (F/2) * 2 == nf_p - 1 && threadId >= base && threadId < base + 1) {
        r_sm = (R/2) * 2;
        c_sm = (C/2) * 2;
        f_sm = (F/2) * 2;
        r_gl = r / 2 + (R/2);
        c_gl = c / 2 + (C/2);
        f_gl = f / 2 + (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
            // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
            // r_sm, c_sm, f_sm);
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *w(r_gl, c_gl, f_gl);
                }
              }
            }
            *w(r_gl, c_gl, f_gl) = res;
            // printf("w(1)-store: (%d %d %d) <- %f (%d %d %d)\n", 
            // r_gl, c_gl, f_gl, *w(r_gl, c_gl, f_gl),
            // r_sm, c_sm, f_sm);
          }
        }
      }

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[store extra] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();

      // start = clock64();

      // printf("wf.isNull(): %d\n", wf.isNull());
      if (!wf.isNull() && threadId >= (R/2) * (C/2) * (F/2) && threadId < (R/2) * (C/2) * (F/2) * 2) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2)) / ((C/2) * (F/2))) * 2;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2)) % ((C/2) * (F/2))) / (F/2)) * 2;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2)) % ((C/2) * (F/2))) % (F/2)) * 2 + 1;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2)) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2)) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2)) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        // printf("TYPE: %d\n", TYPE);
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            // printf("skip: %d\n", skip);
            if (!skip) {
              // printf("INTERPOLATION: %d\n", INTERPOLATION);
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                           ratio_f_sm[f_sm - 1]);
                // printf("fw: lerp(%f, %f) -> %f\n", v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                //                                    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                //                                    res);
              }
              if (INTERPOLATION && CALC_COEFF) { // fused
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) { // calc_coeff only
                res -= *wf(r_gl, c_gl, f_gl);
              }
            }
            *wf(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) {
                if (in_next && f_gl < nf_c) {
                  ;
                } else {
                  res -= *wf(r_gl, c_gl, f_gl);
                }
              }
            }
            *wf(r_gl, c_gl, f_gl) = res;
          }
        }

        // if (nr == 70) printf("f-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
        // c_gl, f_gl, v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm, c_sm,
        // f_sm);
        // asm volatile("membar.cta;");
        // start = clock64() - start;
        // printf("[(F/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
        // blockIdx.y, blockIdx.x, start); start = clock64();
      }
      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[(F/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();

      // if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {

      if (!wc.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 2 && threadId < (R/2) * (C/2) * (F/2) * 3) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) / ((C/2) * (F/2))) * 2;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 2) % ((C/2) * (F/2))) / (F/2)) * 2 + 1;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 2) % ((C/2) * (F/2))) % (F/2)) * 2;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 2) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 2) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wc(r_gl, c_gl, f_gl);
              }
            }
            *wc(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
              if (CALC_COEFF) { // no need to test in_next
                res -= *wc(r_gl, c_gl, f_gl);
              }
            }
            *wc(r_gl, c_gl, f_gl) = res;
          }
          // if (nr == 70) printf("c-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
          // c_gl, f_gl, v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm,
          // c_sm, f_sm);
        }
      }

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[(C/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();

      // if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
      if (!wr.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 3 && threadId < (R/2) * (C/2) * (F/2) * 4) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 3) / ((C/2) * (F/2))) * 2 + 1;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 3) % ((C/2) * (F/2))) / (F/2)) * 2;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 3) % ((C/2) * (F/2))) % (F/2)) * 2;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 3) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 3) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 3) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wr(r_gl, c_gl, f_gl);
              }
            }
            *wr(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
              if (CALC_COEFF) { // no need to test if in_next
                res -= *wr(r_gl, c_gl, f_gl);
              }
            }
            *wr(r_gl, c_gl, f_gl) = res;
          }
        }
      }

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[(R/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();
      __syncthreads();
      if (!wcf.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 4 && threadId < (R/2) * (C/2) * (F/2) * 5) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 4) / ((C/2) * (F/2))) * 2;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 4) % ((C/2) * (F/2))) / (F/2)) * 2 + 1;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 4) % ((C/2) * (F/2))) % (F/2)) * 2 + 1;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 4) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 4) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 4) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
                T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
                T tmp = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
                res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wcf(r_gl, c_gl, f_gl);
              }
            }
            *wcf(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
              if (CALC_COEFF) { // not need to test if in_next
                res -= *wcf(r_gl, c_gl, f_gl);
              }
            }
            *wcf(r_gl, c_gl, f_gl) = res;
          }
        }
      }

      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[CF-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();

      if (!wrf.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 5 && threadId < (R/2) * (C/2) * (F/2) * 6) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 5) / ((C/2) * (F/2))) * 2 + 1;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 5) % ((C/2) * (F/2))) / (F/2)) * 2;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 5) % ((C/2) * (F/2))) % (F/2)) * 2 + 1;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 5) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 5) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 5) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
                T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
                res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wrf(r_gl, c_gl, f_gl);
              }
            }
            *wrf(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
              if (CALC_COEFF) { // no need to test if in_next
                res -= *wrf(r_gl, c_gl, f_gl);
              }
            }
            *wrf(r_gl, c_gl, f_gl) = res;
          }
        }
      }

      if (!wrc.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 6 && threadId < (R/2) * (C/2) * (F/2) * 7) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 6) / ((C/2) * (F/2))) * 2 + 1;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 6) % ((C/2) * (F/2))) / (F/2)) * 2 + 1;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 6) % ((C/2) * (F/2))) % (F/2)) * 2;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 6) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 6) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 6) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wrc(r_gl, c_gl, f_gl);
              }
            }
            *wrc(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              }
              if (CALC_COEFF) { // no need to test if in_next
                res -= *wrc(r_gl, c_gl, f_gl);
              }
            }
            *wrc(r_gl, c_gl, f_gl) = res;
          }
        }
      }

      if (!wrcf.isNull() && threadId >= (R/2) * (C/2) * (F/2) * 7 && threadId < (R/2) * (C/2) * (F/2) * 8) {
        r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 7) / ((C/2) * (F/2))) * 2 + 1;
        c_sm = (((threadId - (R/2) * (C/2) * (F/2) * 7) % ((C/2) * (F/2))) / (F/2)) * 2 + 1;
        f_sm = (((threadId - (R/2) * (C/2) * (F/2) * 7) % ((C/2) * (F/2))) % (F/2)) * 2 + 1;
        r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 7) / ((C/2) * (F/2));
        c_gl = c / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 7) % ((C/2) * (F/2))) / (F/2);
        f_gl = f / 2 + ((threadId - (R/2) * (C/2) * (F/2) * 7) % ((C/2) * (F/2))) % (F/2);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                T f1 = lerp(
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
                T f2 = lerp(
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
                T f3 = lerp(
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
                T f4 = lerp(
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);

                T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
                T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

                res = lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= *wrcf(r_gl, c_gl, f_gl);
              }
            }
            *wrcf(r_gl, c_gl, f_gl) = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
                res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              }
              if (CALC_COEFF) { // no need to test if in_next
                res -= *wrcf(r_gl, c_gl, f_gl);
              }
            }
            *wrcf(r_gl, c_gl, f_gl) = res;
          }
        }
      }
      // end = clock64();

      // asm volatile("membar.cta;");
      // if (threadId < 256 && blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x ==
      // 0) printf("threadId %d elapsed %lu\n", threadId, end-start);
      if (r + (R/2) * 2 == nr_p - 1) {
        // printf("test\n");
        if (threadId < (C/2) * (F/2)) {
          // printf("test1\n");
          if (!wf.isNull()) {
            // printf("test2\n");
            r_sm = (R/2) * 2;
            c_sm = (threadId / (F/2)) * 2;
            f_sm = (threadId % (F/2)) * 2 + 1;
            r_gl = r / 2 + (R/2);
            c_gl = c / 2 + threadId / (F/2);
            f_gl = f / 2 + threadId % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
                // printf("test3\n");
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                               ratio_f_sm[f_sm - 1]);
                    // printf("fw: lerp(%f, %f) -> %f\n", v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                    //                                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                    //                                res);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wf(r_gl, c_gl, f_gl);
                  }
                }
                // printf("dwf (%d %d %d): %f\n", r_gl, c_gl, f_gl, res);
                *wf(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl = 2 * f_gl + 1;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    ;
                  }
                  if (CALC_COEFF) { // need to test if in_next
                    if (in_next && f_gl < nf_c) {
                      ;
                    } // in_next
                    else {
                      res -= *wf(r_gl, c_gl, f_gl);
                    }
                  }
                }
                *wf(r_gl, c_gl, f_gl) = res;
                // printf("wf (%d %d %d): %f\n", r_gl, c_gl, f_gl, res);
              }
            }
          }

          if (!wc.isNull()) {
            r_sm = (R/2) * 2;
            c_sm = (threadId / (F/2)) * 2 + 1;
            f_sm = (threadId % (F/2)) * 2;
            r_gl = r / 2 + (R/2);
            c_gl = c / 2 + threadId / (F/2);
            f_gl = f / 2 + threadId % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                               ratio_c_sm[c_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wc(r_gl, c_gl, f_gl);
                  }
                }
                *wc(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl *= 2;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                               ratio_c_sm[c_sm - 1]);
                    // if (idx[1]==0 && idx[2]==0) {
                    //   printf("wc-lerp: %f %f (%f) -> %f\n", 
                    //   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                    //   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                    //   ratio_c_sm[c_sm - 1],
                    //   res);
                    // }
                  }
                  if (CALC_COEFF) { // no need to test if in_next
                    res -= *wc(r_gl, c_gl, f_gl);
                  }
                }
                *wc(r_gl, c_gl, f_gl) = res;

                // if (idx[1]==0 && idx[2]==0) {
                //   printf("wc-store: (%d %d %d) <- %f (%d %d %d)\n", 
                //   r_gl, c_gl, f_gl, *wc(r_gl, c_gl, f_gl),
                //   r_sm, c_sm, f_sm);
                // }
              }
            }
          }

         
          if (!wcf.isNull()) {
            r_sm = (R/2) * 2;
            c_sm = (threadId / (F/2)) * 2 + 1;
            f_sm = (threadId % (F/2)) * 2 + 1;
            r_gl = r / 2 + (R/2);
            c_gl = c / 2 + threadId / (F/2);
            f_gl = f / 2 + threadId % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    T f1 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
                    T f2 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
                    res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wcf(r_gl, c_gl, f_gl);
                  }
                }
                *wcf(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl = 2 * f_gl + 1;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                               ratio_c_sm[c_sm - 1]);
                  }
                  if (CALC_COEFF) {
                    res -= *wcf(r_gl, c_gl, f_gl);
                  }
                }
                *wcf(r_gl, c_gl, f_gl) = res;
              }
            }
          }
        }
      }

      if (c + (C/2) * 2 == nc_p - 1) {
        if (threadId >= (R/2) * (C/2) * (F/2) && threadId < (R/2) * (C/2) * (F/2) + (R/2) * (F/2)) {
          if (!wf.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2)) / (F/2)) * 2;
            c_sm = (C/2) * 2;
            f_sm = ((threadId - (R/2) * (C/2) * (F/2)) % (F/2)) * 2 + 1;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2)) / (F/2);
            c_gl = c / 2 + (C/2);
            f_gl = f / 2 + (threadId - (R/2) * (C/2) * (F/2)) % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                               ratio_f_sm[f_sm - 1]);
                    // printf("fw: lerp(%f, %f) -> %f\n", v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                    //                                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                    //                                res);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wf(r_gl, c_gl, f_gl);
                  }
                }
                *wf(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl = 2 * f_gl + 1;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    ;
                  }
                  if (CALC_COEFF) { // need to test if in_next
                    if (in_next && f_gl < nf_c) {
                      ;
                    } // in_next
                    else {
                      res -= *wf(r_gl, c_gl, f_gl);
                    }
                  }
                }
                *wf(r_gl, c_gl, f_gl) = res;
              }
            }
          }

          if (!wr.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2)) / (F/2)) * 2 + 1;
            c_sm = (C/2) * 2;
            f_sm = ((threadId - (R/2) * (C/2) * (F/2)) % (F/2)) * 2;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2)) / (F/2);
            c_gl = c / 2 + (C/2);
            f_gl = f / 2 + (threadId - (R/2) * (C/2) * (F/2)) % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                               ratio_r_sm[r_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wr(r_gl, c_gl, f_gl);
                  }
                }
                *wr(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl *= 2;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                               ratio_r_sm[r_sm - 1]);
                  }
                  if (CALC_COEFF) {
                    res -= *wr(r_gl, c_gl, f_gl);
                  }
                }
                *wr(r_gl, c_gl, f_gl) = res;
              }
            }
          }

          if (!wrf.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2)) / (F/2)) * 2 + 1;
            c_sm = (C/2) * 2;
            f_sm = ((threadId - (R/2) * (C/2) * (F/2)) % (F/2)) * 2 + 1;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2)) / (F/2);
            c_gl = c / 2 + (C/2);
            f_gl = f / 2 + (threadId - (R/2) * (C/2) * (F/2)) % (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    T f1 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
                    T f2 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
                    res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wrf(r_gl, c_gl, f_gl);
                  }
                }
                *wrf(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl = 2 * f_gl + 1;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                               ratio_r_sm[r_sm - 1]);
                  }
                  if (CALC_COEFF) { // no need to test if in_next
                    res -= *wrf(r_gl, c_gl, f_gl);
                  }
                }
                *wrf(r_gl, c_gl, f_gl) = res;
              }
            }
          }
        }
      }

      if (f + (F/2) * 2 == nf_p - 1) {
        if (threadId >= (R/2) * (C/2) * (F/2) * 2 && threadId < (R/2) * (C/2) * (F/2) * 2 + (R/2) * (C/2)) {
          if (!wc.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2)) * 2;
            c_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2)) * 2 + 1;
            f_sm = (F/2) * 2;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2);
            c_gl = c / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2);
            f_gl = f / 2 + (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                               ratio_c_sm[c_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wc(r_gl, c_gl, f_gl);
                  }
                }
                *wc(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl *= 2;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                               ratio_c_sm[c_sm - 1]);
                  }
                  if (CALC_COEFF) {
                    res -= *wc(r_gl, c_gl, f_gl);
                  }
                }
                *wc(r_gl, c_gl, f_gl) = res;
              }
            }
          }

          if (!wr.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2)) * 2 + 1;
            c_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2)) * 2;
            f_sm = (F/2) * 2;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2);
            c_gl = c / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2);
            f_gl = f / 2 + (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                               ratio_r_sm[r_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wr(r_gl, c_gl, f_gl);
                  }
                }
                *wr(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl *= 2;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                               ratio_r_sm[r_sm - 1]);
                  }
                  if (CALC_COEFF) {
                    res -= *wr(r_gl, c_gl, f_gl);
                  }
                }
                *wr(r_gl, c_gl, f_gl) = res;
              }
            }
          }

          if (!wrc.isNull()) {
            r_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2)) * 2 + 1;
            c_sm = ((threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2)) * 2 + 1;
            f_sm = (F/2) * 2;
            r_gl = r / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) / (C/2);
            c_gl = c / 2 + (threadId - (R/2) * (C/2) * (F/2) * 2) % (C/2);
            f_gl = f / 2 + (F/2);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
            if (TYPE == 1) {
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
                if (!skip) {
                  if (INTERPOLATION) {
                    T c1 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
                    T c2 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
                    res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
                  }
                  if (INTERPOLATION && CALC_COEFF) {
                    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                  }
                  if (!INTERPOLATION && CALC_COEFF) {
                    res -= *wrc(r_gl, c_gl, f_gl);
                  }
                }
                *wrc(r_gl, c_gl, f_gl) = res;
              }
            } else if (TYPE == 2) {
              f_gl *= 2;
              if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                  r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
                if (!skip) {
                  if (INTERPOLATION) {
                    T c1 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
                    T c2 = lerp(
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
                    res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
                  }
                  if (CALC_COEFF) {
                    res -= *wrc(r_gl, c_gl, f_gl);
                  }
                }
                *wrc(r_gl, c_gl, f_gl) = res;
              }
            }
          }
        }
      }

      if (!wr.isNull() && c + (C/2) * 2 == nc_p - 1 && f + (F/2) * 2 == nf_p - 1) {
        if (threadId >= (R/2) * (C/2) * (F/2) * 3 && threadId < (R/2) * (C/2) * (F/2) * 3 + (R/2)) {
          r_sm = (threadId - (R/2) * (C/2) * (F/2) * 3) * 2 + 1;
          c_sm = (C/2) * 2;
          f_sm = (F/2) * 2;
          r_gl = r / 2 + threadId - (R/2) * (C/2) * (F/2) * 3;
          c_gl = c / 2 + (C/2);
          f_gl = f / 2 + (F/2);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= *wr(r_gl, c_gl, f_gl);
                }
              }
              *wr(r_gl, c_gl, f_gl) = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= *wr(r_gl, c_gl, f_gl);
                }
              }
              *wr(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }

      if (!wc.isNull() && r + (R/2) * 2 == nr_p - 1 && f + (F/2) * 2 == nf_p - 1) {
        if (threadId >= (R/2) * (C/2) * (F/2) * 4 && threadId < (R/2) * (C/2) * (F/2) * 4 + (C/2)) {
          r_sm = (R/2) * 2;
          c_sm = (threadId - (R/2) * (C/2) * (F/2) * 4) * 2 + 1;
          f_sm = (F/2) * 2;
          r_gl = r / 2 + (R/2);
          c_gl = c / 2 + threadId - (R/2) * (C/2) * (F/2) * 4;
          f_gl = f / 2 + (F/2);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= *wc(r_gl, c_gl, f_gl);
                }
              }
              *wc(r_gl, c_gl, f_gl) = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= *wc(r_gl, c_gl, f_gl);
                }
              }
              *wc(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }

      // printf("test1\n");
      if (!wf.isNull() && r + (R/2) * 2 == nr_p - 1 && c + (C/2) * 2 == nc_p - 1) {
        // printf("test2\n");
        if (threadId >= (R/2) * (C/2) * (F/2) * 5 && threadId < (R/2) * (C/2) * (F/2) * 5 + (F/2)) {
          // printf("test3\n");
          r_sm = (R/2) * 2;
          c_sm = (C/2) * 2;
          f_sm = (threadId - (R/2) * (C/2) * (F/2) * 5) * 2 + 1;
          r_gl = r / 2 + (R/2);
          c_gl = c / 2 + (C/2);
          f_gl = f / 2 + threadId - (R/2) * (C/2) * (F/2) * 5;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
              // printf("test4\n");
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                             ratio_f_sm[f_sm - 1]);
                  // printf("fw: lerp(%f, %f) -> %f\n", v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                  //                                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                  //                                  res);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= *wf(r_gl, c_gl, f_gl);
                }
              }
              *wf(r_gl, c_gl, f_gl) = res;
              // printf("dwf(%d %d %d): %f\n", r_gl, c_gl, f_gl,
              // dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)]);
            }
          } else if (TYPE == 2) {
            f_gl = 2 * f_gl + 1;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  ;
                }
                if (CALC_COEFF) { // do need to test in_next
                  if (in_next && f_gl < nf_c) {
                    ;
                  } // in_next
                  else {
                    res -= *wf(r_gl, c_gl, f_gl);
                  }
                }
              }
              *wf(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }

    } // skip

    // if (r == 0 && c == 0 && f == 0 && threadId == 0) {
    //   printf("out config: %d %d %d (%d %d %d)\n", (R/2), (C/2), (F/2), r,c,f);
    //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
    //     for (int j = 0; j < (C/2) * 2 + 1; j++) {
    //       for (int k = 0; k < (F/2) * 2 + 1; k++) {
    //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }
  }


 
  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
    size += (D_GLOBAL * 4) * sizeof(SIZE);
    size += (D_GLOBAL * 1) * sizeof(DIM);
    return size;
  }


  private:

  SubArray<1, SIZE, DeviceType> shape, shape_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D_GLOBAL, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;
  // SIZE *ldvs;
  // SIZE *ldws;
  DIM unprocessed_n;
  SubArray<1, DIM, DeviceType> unprocessed_dims;
  LENGTH threadId;
  T * v_sm;
  T * ratio_f_sm;
  T * ratio_c_sm;
  T * ratio_r_sm;
  DIM curr_dim_r;
  DIM curr_dim_c;
  DIM curr_dim_f;
  

  

  SIZE nr, nc, nf;
  SIZE nr_c, nc_c, nf_c;
  SIZE r, c, f;
  SIZE rest_r, rest_c, rest_f;
  SIZE nr_p, nc_p, nf_p;
  SIZE rest_r_p, rest_c_p, rest_f_p;
  SIZE r_sm, c_sm, f_sm;
  SIZE r_sm_ex, c_sm_ex, f_sm_ex;
  SIZE r_gl, c_gl, f_gl;
  SIZE r_gl_ex, c_gl_ex, f_gl_ex;
  T res;
  bool in_next;

  SIZE ldsm1, ldsm2;
  

  SIZE * sm_size;
  SIZE * shape_sm;
  SIZE * shape_c_sm;
  SIZE * lvs_sm;
  SIZE * ldws_sm;

  DIM * sm_dim;
  DIM * unprocessed_dims_sm;

  SIZE idx[D_GLOBAL];

  int skip;
};

template <DIM D_GLOBAL, DIM D_LOCAL, typename T,
          bool INTERPOLATION, bool CALC_COEFF, int TYPE, typename DeviceType>
class GpkReo: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  GpkReo():AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDm_CONT
  Task<GpkReoFunctor<D_GLOBAL, D_LOCAL, T, R, C, F, INTERPOLATION, CALC_COEFF, TYPE, DeviceType>> 
  GenTask(SubArray<1, SIZE, DeviceType> shape, SubArray<1, SIZE, DeviceType> shape_c,
          DIM unprocessed_n, SubArray<1, DIM, DeviceType> unprocessed_dims,
          DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
          SubArray<1, T, DeviceType> ratio_r, 
          SubArray<1, T, DeviceType> ratio_c, 
          SubArray<1, T, DeviceType> ratio_f, 
          SubArray<D_GLOBAL, T, DeviceType> v,
          SubArray<D_GLOBAL, T, DeviceType> w,
          SubArray<D_GLOBAL, T, DeviceType> wf, SubArray<D_GLOBAL, T, DeviceType> wc, SubArray<D_GLOBAL, T, DeviceType> wr, 
          SubArray<D_GLOBAL, T, DeviceType> wcf, SubArray<D_GLOBAL, T, DeviceType> wrf, SubArray<D_GLOBAL, T, DeviceType> wrc,
          SubArray<D_GLOBAL, T, DeviceType> wrcf, int queue_idx) {

    using FunctorType = GpkReoFunctor<D_GLOBAL, D_LOCAL, T, R, C, F, INTERPOLATION, CALC_COEFF, TYPE, DeviceType>;
    FunctorType functor(shape, shape_c, unprocessed_n, unprocessed_dims,
                        curr_dim_r, curr_dim_c, curr_dim_f,
                        ratio_r, ratio_c, ratio_f,
                        v, w, wf, wc, wr, wcf, 
                        wrf, wrc, wrcf); 

    size_t base = (size_t)&(functor.ngridz);

    // printf("ngridz\t%llu\n", &(functor.ngridz));
    // printf("shape\t%llu\n", &(functor.shape));
    // printf("threadId\t%llu\n", &(functor.threadId));
    // printf("v_sm\t%llu\n", &(functor.v_sm));
                                                        
    SIZE nr = shape.dataHost()[curr_dim_r];
    SIZE nc = shape.dataHost()[curr_dim_c];
    SIZE nf = shape.dataHost()[curr_dim_f];
    if (D_LOCAL == 2) {
      nr = 1;
    }
    SIZE total_thread_z = std::max(nr - 1, (SIZE)1);
    SIZE total_thread_y = std::max(nc - 1, (SIZE)1);
    SIZE total_thread_x = std::max(nf - 1, (SIZE)1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;          
    size_t sm_size = functor.shared_memory_size();   
    // printf("sm_size: %llu\n", sm_size);
    // printf("RCF: %u %u %u\n", R, C, F);                         
    tbz = R;                                          
    tby = C;                                          
    tbx = F;                                          
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (DIM d = 0; d < D_GLOBAL; d++) {
      if (D_LOCAL == 3 && d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        gridx *= shape.dataHost()[d];
      }
      if (D_LOCAL == 2 && d != curr_dim_f && d != curr_dim_c) {
        gridx *= shape.dataHost()[d];
      }
    }

    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "GpkReo"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape, SubArray<1, SIZE, DeviceType> shape_c,
            DIM unprocessed_n, SubArray<1, DIM, DeviceType> unprocessed_dims,
            DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
            SubArray<1, T, DeviceType> ratio_r, 
            SubArray<1, T, DeviceType> ratio_c, 
            SubArray<1, T, DeviceType> ratio_f, 
            SubArray<D_GLOBAL, T, DeviceType> v,
            SubArray<D_GLOBAL, T, DeviceType> w, 
            SubArray<D_GLOBAL, T, DeviceType> wf, SubArray<D_GLOBAL, T, DeviceType> wc, SubArray<D_GLOBAL, T, DeviceType> wr, 
            SubArray<D_GLOBAL, T, DeviceType> wcf, SubArray<D_GLOBAL, T, DeviceType> wrf, SubArray<D_GLOBAL, T, DeviceType> wrc,
            SubArray<D_GLOBAL, T, DeviceType> wrcf, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(shape.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.auto_tuning_cc[arch][prec][range_l];
    #define GPK(CONFIG)                                    \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) { \
      const int R=GPK_CONFIG[D_LOCAL-1][CONFIG][0];\
      const int C=GPK_CONFIG[D_LOCAL-1][CONFIG][1];\
      const int F=GPK_CONFIG[D_LOCAL-1][CONFIG][2];\
      using FunctorType = GpkReoFunctor<D_GLOBAL, D_LOCAL, T, R, C, F, INTERPOLATION, CALC_COEFF, TYPE, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F>(\
                                shape, shape_c, unprocessed_n, unprocessed_dims,\
                                curr_dim_r, curr_dim_c, curr_dim_f,\
                                ratio_r, ratio_c, ratio_f,\
                                v, w, \
                                wf, wc, wr, \
                                wcf, wrf, wrc, \
                                wrcf, queue_idx); \
      DeviceAdapter<TaskType, DeviceType> adapter; \
      adapter.Execute(task);\
    }

    GPK(0)
    GPK(1)
    GPK(2)
    GPK(3)
    GPK(4)  
    GPK(5)
    GPK(6)
    #undef GPK
  }
};





}

#endif