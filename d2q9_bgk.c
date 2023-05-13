#include "d2q9_bgk.h"
#include "immintrin.h"
#include <omp.h>
#include <string.h>
#include <xmmintrin.h>

/* The main processes in one step */
int collision(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells, int *obstacles);
int streaming(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells);
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles);
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets);
int fuse(int start_col, int end_col, const t_param params, t_speed *cells,
         t_speed *tmp_cells, int *obstacles);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, int *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */

  fuse(0, params.nx, params, cells, tmp_cells, obstacles);

  boundary(params, tmp_cells, cells, inlets);

  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/

#define chunk_x 128
#define chunk_y 64
t_speed buffer[chunk_y][chunk_x];
float u[NSPEEDS];
float d_equ[NSPEEDS];
int fuse(int start_col, int end_col, const t_param params, t_speed *cells,
         t_speed *tmp_cells, int *obstacles) {
  static const float c_sq = 1.f / 3.f; /* square of speed of sound */
  static const float w0 = 4.f / 9.f;   /* weighting factor */
  static const float w1 = 1.f / 9.f;   /* weighting factor */
  static const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  __m256 _1 = _mm256_set1_ps(1.f);
  __m256 c = _mm256_set1_ps(c_sq);
  __m256 _2_c_c = _mm256_set1_ps(2.f * c_sq * c_sq);
  __m256 w = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  __m256 omega = _mm256_set1_ps(params.omega);

#pragma omp parallel private(buffer, u, d_equ)
  {
    int id = omp_get_thread_num();
    int col_per_thread = params.nx / omp_get_num_threads() + 1;
    int start_col = id * col_per_thread, end_col = (id + 1) * col_per_thread;
    int jj_start, jj_end, ii_start, ii_end;

    if (end_col > params.nx)
      end_col = params.nx;
    for (int j = 0; j < params.ny; j += chunk_y) {
      jj_start = j;
      jj_end = ((j + chunk_y) > params.ny) ? params.ny : (j + chunk_y);
      for (int i = start_col; i < end_col; i += chunk_x) {
        ii_start = i;
        ii_end = ((i + chunk_x) > end_col) ? end_col : (i + chunk_x);
        for (int jj = jj_start, jj_offset = 0; jj < jj_end; jj++, jj_offset++) {
          for (int ii = ii_start, ii_offset = 0; ii < ii_end;
               ii++, ii_offset++) {
            if (!obstacles[ii + jj * params.nx]) {
              /* compute local density total */
              float local_density = 0.f;

              for (int kk = 0; kk < NSPEEDS; kk++) {
                local_density += cells[ii + jj * params.nx].speeds[kk];
              }

              /* compute x velocity component */
              float u_x = (cells[ii + jj * params.nx].speeds[1] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[8] -
                           (cells[ii + jj * params.nx].speeds[3] +
                            cells[ii + jj * params.nx].speeds[6] +
                            cells[ii + jj * params.nx].speeds[7])) /
                          local_density;
              /* compute y velocity component */
              float u_y = (cells[ii + jj * params.nx].speeds[2] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[6] -
                           (cells[ii + jj * params.nx].speeds[4] +
                            cells[ii + jj * params.nx].speeds[7] +
                            cells[ii + jj * params.nx].speeds[8])) /
                          local_density;

              /* velocity squared */
              float u_sq = u_x * u_x + u_y * u_y;

              /* directional velocity components */

              u[0] = 0;          /* zero */
              u[1] = u_x;        /* east */
              u[2] = u_y;        /* north */
              u[3] = -u_x;       /* west */
              u[4] = -u_y;       /* south */
              u[5] = u_x + u_y;  /* north-east */
              u[6] = -u_x + u_y; /* north-west */
              u[7] = -u_x - u_y; /* south-west */
              u[8] = u_x - u_y;  /* south-east */

              /* equilibrium densities */

              /* zero velocity density: weight w0 */

              d_equ[0] =
                  w0 * local_density *
                  (1.f + u[0] / c_sq + (u[0] * u[0]) / (2.f * c_sq * c_sq) -
                   u_sq / (2.f * c_sq));
              __m256 x = _mm256_loadu_ps(u + 1);

              __m256 l_d = _mm256_set1_ps(local_density);

              __m256 u_2_c = _mm256_set1_ps(u_sq / (2.f * c_sq));
              __m256 x_2 = _mm256_mul_ps(x, x);
              x = _mm256_div_ps(x, c);
              x_2 = _mm256_div_ps(x_2, _2_c_c);
              __m256 res_1 = _mm256_add_ps(_1, x);
              __m256 res_2 = _mm256_sub_ps(x_2, u_2_c);
              __m256 res = _mm256_add_ps(res_1, res_2);
              res = _mm256_mul_ps(res, l_d);

              res = _mm256_mul_ps(res, w);
              _mm256_storeu_ps(d_equ + 1, res);
              /* relaxation step */
              buffer[jj_offset][ii_offset].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0] +
                  params.omega *
                      (d_equ[0] - cells[ii + jj * params.nx].speeds[0]);
              __m256 c_s =
                  _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
              res = _mm256_sub_ps(res, c_s);
              res = _mm256_mul_ps(res, omega);
              res = _mm256_add_ps(res, c_s);
              _mm256_storeu_ps(buffer[jj_offset][ii_offset].speeds + 1, res);
            } else {
              buffer[jj_offset][ii_offset].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0];
              buffer[jj_offset][ii_offset].speeds[1] =
                  cells[ii + jj * params.nx].speeds[3];
              buffer[jj_offset][ii_offset].speeds[2] =
                  cells[ii + jj * params.nx].speeds[4];
              buffer[jj_offset][ii_offset].speeds[3] =
                  cells[ii + jj * params.nx].speeds[1];
              buffer[jj_offset][ii_offset].speeds[4] =
                  cells[ii + jj * params.nx].speeds[2];
              buffer[jj_offset][ii_offset].speeds[5] =
                  cells[ii + jj * params.nx].speeds[7];
              buffer[jj_offset][ii_offset].speeds[6] =
                  cells[ii + jj * params.nx].speeds[8];
              buffer[jj_offset][ii_offset].speeds[7] =
                  cells[ii + jj * params.nx].speeds[5];
              buffer[jj_offset][ii_offset].speeds[8] =
                  cells[ii + jj * params.nx].speeds[6];
            }
            if (ii == 0 || jj == 0 || ii == params.nx - 1 ||
                jj == params.ny - 1)
              memcpy(&cells[ii + jj * params.nx], &buffer[jj_offset][ii_offset],
                     sizeof(buffer[jj_offset][ii_offset]));
          }
        }
        for (int jj = jj_start, jj_offset = 0; jj < jj_end; jj++, jj_offset++) {
          for (int ii = ii_start, ii_offset = 0; ii < ii_end;
               ii++, ii_offset++) {
            int y_n = ((jj + 1) >= params.ny) ? 0 : jj + 1;
            int x_e = ((ii + 1) >= params.nx) ? 0 : ii + 1;
            int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
            int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            tmp_cells[ii + jj * params.nx].speeds[0] =
                buffer[jj_offset][ii_offset]
                    .speeds[0]; /* central cell, no movement */
            tmp_cells[x_e + jj * params.nx].speeds[1] =
                buffer[jj_offset][ii_offset].speeds[1]; /* east */
            tmp_cells[ii + y_n * params.nx].speeds[2] =
                buffer[jj_offset][ii_offset].speeds[2]; /* north */
            tmp_cells[x_w + jj * params.nx].speeds[3] =
                buffer[jj_offset][ii_offset].speeds[3]; /* west */
            tmp_cells[ii + y_s * params.nx].speeds[4] =
                buffer[jj_offset][ii_offset].speeds[4]; /* south */
            tmp_cells[x_e + y_n * params.nx].speeds[5] =
                buffer[jj_offset][ii_offset].speeds[5]; /* north-east */
            tmp_cells[x_w + y_n * params.nx].speeds[6] =
                buffer[jj_offset][ii_offset].speeds[6]; /* north-west */
            tmp_cells[x_w + y_s * params.nx].speeds[7] =
                buffer[jj_offset][ii_offset].speeds[7]; /* south-west */
            tmp_cells[x_e + y_s * params.nx].speeds[8] =
                buffer[jj_offset][ii_offset].speeds[8]; /* south-east */
          }
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells, int *obstacles) {
  static const float c_sq = 1.f / 3.f; /* square of speed of sound */
  static const float w0 = 4.f / 9.f;   /* weighting factor */
  static const float w1 = 1.f / 9.f;   /* weighting factor */
  static const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  __m256 _1 = _mm256_set1_ps(1.f);
  __m256 c = _mm256_set1_ps(c_sq);
  __m256 _2_c_c = _mm256_set1_ps(2.f * c_sq * c_sq);
  __m256 w = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  __m256 omega = _mm256_set1_ps(params.omega);

#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = start_col; ii < end_col; ii++) {
      t_speed buffer;
      if (!obstacles[ii + jj * params.nx]) {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += cells[ii + jj * params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (cells[ii + jj * params.nx].speeds[1] +
                     cells[ii + jj * params.nx].speeds[5] +
                     cells[ii + jj * params.nx].speeds[8] -
                     (cells[ii + jj * params.nx].speeds[3] +
                      cells[ii + jj * params.nx].speeds[6] +
                      cells[ii + jj * params.nx].speeds[7])) /
                    local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj * params.nx].speeds[2] +
                     cells[ii + jj * params.nx].speeds[5] +
                     cells[ii + jj * params.nx].speeds[6] -
                     (cells[ii + jj * params.nx].speeds[4] +
                      cells[ii + jj * params.nx].speeds[7] +
                      cells[ii + jj * params.nx].speeds[8])) /
                    local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[0] = 0;          /* zero */
        u[1] = u_x;        /* east */
        u[2] = u_y;        /* north */
        u[3] = -u_x;       /* west */
        u[4] = -u_y;       /* south */
        u[5] = u_x + u_y;  /* north-east */
        u[6] = -u_x + u_y; /* north-west */
        u[7] = -u_x - u_y; /* south-west */
        u[8] = u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */

        d_equ[0] = w0 * local_density *
                   (1.f + u[0] / c_sq + (u[0] * u[0]) / (2.f * c_sq * c_sq) -
                    u_sq / (2.f * c_sq));
        __m256 x = _mm256_loadu_ps(u + 1);

        __m256 l_d = _mm256_set1_ps(local_density);

        __m256 u_2_c = _mm256_set1_ps(u_sq / (2.f * c_sq));
        __m256 x_2 = _mm256_mul_ps(x, x);
        x = _mm256_div_ps(x, c);
        x_2 = _mm256_div_ps(x_2, _2_c_c);
        __m256 res_1 = _mm256_add_ps(_1, x);
        __m256 res_2 = _mm256_sub_ps(x_2, u_2_c);
        __m256 res = _mm256_add_ps(res_1, res_2);
        res = _mm256_mul_ps(res, l_d);

        res = _mm256_mul_ps(res, w);
        _mm256_storeu_ps(d_equ + 1, res);
        /* axis speeds: weight w1 */
        // d_equ[1] = w1 * local_density *
        //            (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[2] = w1 * local_density *
        //            (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[3] = w1 * local_density *
        //            (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[4] = w1 * local_density *
        //            (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // /* diagonal speeds: weight w2 */
        // d_equ[5] = w2 * local_density *
        //            (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[6] = w2 * local_density *
        //            (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[7] = w2 * local_density *
        //            (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        // d_equ[8] = w2 * local_density *
        //            (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) -
        //             u_sq / (2.f * c_sq));
        /* relaxation step */
        buffer.speeds[0] =
            cells[ii + jj * params.nx].speeds[0] +
            params.omega * (d_equ[0] - cells[ii + jj * params.nx].speeds[0]);
        __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
        res = _mm256_sub_ps(res, c_s);
        res = _mm256_mul_ps(res, omega);
        res = _mm256_add_ps(res, c_s);
        _mm256_storeu_ps(buffer.speeds + 1, res);
        // for (int kk = 0; kk < NSPEEDS; kk++) {
        //   tmp_cells[ii + jj * params.nx].speeds[kk] =
        //       cells[ii + jj * params.nx].speeds[kk] +
        //       params.omega *
        //           (d_equ[kk] - cells[ii + jj * params.nx].speeds[kk]);
        // }
      } else {
        buffer.speeds[0] = cells[ii + jj * params.nx].speeds[0];
        buffer.speeds[1] = cells[ii + jj * params.nx].speeds[3];
        buffer.speeds[2] = cells[ii + jj * params.nx].speeds[4];
        buffer.speeds[3] = cells[ii + jj * params.nx].speeds[1];
        buffer.speeds[4] = cells[ii + jj * params.nx].speeds[2];
        buffer.speeds[5] = cells[ii + jj * params.nx].speeds[7];
        buffer.speeds[6] = cells[ii + jj * params.nx].speeds[8];
        buffer.speeds[7] = cells[ii + jj * params.nx].speeds[5];
        buffer.speeds[8] = cells[ii + jj * params.nx].speeds[6];
      }
      memcpy(&cells[ii + jj * params.nx], &buffer, sizeof(buffer));
    }
  }
}

/*
** Particles flow to the corresponding cell according to their speed
*direaction.
*/
int streaming(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells) {
  /* loop over _all_ cells */

#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = start_col; ii < end_col; ii++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = ((jj + 1) >= params.ny) ? 0 : jj + 1;
      int x_e = ((ii + 1) >= params.nx) ? 0 : ii + 1;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells[ii + jj * params.nx].speeds[0] =
          tmp_cells[ii + jj * params.nx]
              .speeds[0]; /* central cell, no movement */
      cells[x_e + jj * params.nx].speeds[1] =
          tmp_cells[ii + jj * params.nx].speeds[1]; /* east */
      cells[ii + y_n * params.nx].speeds[2] =
          tmp_cells[ii + jj * params.nx].speeds[2]; /* north */
      cells[x_w + jj * params.nx].speeds[3] =
          tmp_cells[ii + jj * params.nx].speeds[3]; /* west */
      cells[ii + y_s * params.nx].speeds[4] =
          tmp_cells[ii + jj * params.nx].speeds[4]; /* south */
      cells[x_e + y_n * params.nx].speeds[5] =
          tmp_cells[ii + jj * params.nx].speeds[5]; /* north-east */
      cells[x_w + y_n * params.nx].speeds[6] =
          tmp_cells[ii + jj * params.nx].speeds[6]; /* north-west */
      cells[x_w + y_s * params.nx].speeds[7] =
          tmp_cells[ii + jj * params.nx].speeds[7]; /* south-west */
      cells[x_e + y_s * params.nx].speeds[8] =
          tmp_cells[ii + jj * params.nx].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound
*plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0 / 3.0;
  const float cst2 = 1.0 / 6.0;
  const float cst3 = 1.0 / 2.0;

  int ii, jj;
  float local_density;

  // top wall (bounce)
  jj = params.ny - 1;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells[ii + jj * params.nx].speeds[4] =
        tmp_cells[ii + jj * params.nx].speeds[2];
    cells[ii + jj * params.nx].speeds[7] =
        tmp_cells[ii + jj * params.nx].speeds[5];
    cells[ii + jj * params.nx].speeds[8] =
        tmp_cells[ii + jj * params.nx].speeds[6];
  }

  // bottom wall (bounce)
  jj = 0;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells[ii + jj * params.nx].speeds[2] =
        tmp_cells[ii + jj * params.nx].speeds[4];
    cells[ii + jj * params.nx].speeds[5] =
        tmp_cells[ii + jj * params.nx].speeds[7];
    cells[ii + jj * params.nx].speeds[6] =
        tmp_cells[ii + jj * params.nx].speeds[8];
  }

  // left wall (inlet)
  ii = 0;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {
    local_density = (cells[ii + jj * params.nx].speeds[0] +
                     cells[ii + jj * params.nx].speeds[2] +
                     cells[ii + jj * params.nx].speeds[4] +
                     2.0 * cells[ii + jj * params.nx].speeds[3] +
                     2.0 * cells[ii + jj * params.nx].speeds[6] +
                     2.0 * cells[ii + jj * params.nx].speeds[7]) /
                    (1.0 - inlets[jj]);

    cells[ii + jj * params.nx].speeds[1] =
        cells[ii + jj * params.nx].speeds[3] +
        cst1 * local_density * inlets[jj];

    cells[ii + jj * params.nx].speeds[5] =
        cells[ii + jj * params.nx].speeds[7] -
        cst3 * (cells[ii + jj * params.nx].speeds[2] -
                cells[ii + jj * params.nx].speeds[4]) +
        cst2 * local_density * inlets[jj];

    cells[ii + jj * params.nx].speeds[8] =
        cells[ii + jj * params.nx].speeds[6] +
        cst3 * (cells[ii + jj * params.nx].speeds[2] -
                cells[ii + jj * params.nx].speeds[4]) +
        cst2 * local_density * inlets[jj];
  }

  // right wall (outlet)
  ii = params.nx - 1;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {

    for (int kk = 0; kk < NSPEEDS; kk++) {
      cells[ii + jj * params.nx].speeds[kk] =
          cells[ii - 1 + jj * params.nx].speeds[kk];
    }
  }

  return EXIT_SUCCESS;
}
