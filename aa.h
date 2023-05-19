#pragma once

#include "types.h"

int aa_odd(const t_param params, t_speed *cells, t_speed *tmp_cells,
           int *obstacles);

int aa_even(const t_param params, t_speed *cells, t_speed *tmp_cells,
            int *obstacles, float *inlets);

int aa_even_timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
                     float *inlets, int *obstacles);

int aa_odd_timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
                    float *inlets, int *obstacles);

int aa_boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
                float *inlets);