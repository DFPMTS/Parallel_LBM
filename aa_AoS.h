#pragma once

#include "types.h"

int AoS_aa_odd(const t_param params, t_speed_aos *cells, t_speed *tmp_cells,
               int *obstacles);

int AoS_aa_even(const t_param params, t_speed_aos *cells, t_speed *tmp_cells,
                int *obstacles, float *inlets);

int AoS_aa_even_timestep(const t_param params, t_speed_aos *cells,
                         t_speed *tmp_cells, float *inlets, int *obstacles);

int AoS_aa_odd_timestep(const t_param params, t_speed_aos *cells,
                        t_speed *tmp_cells, float *inlets, int *obstacles);

int AoS_aa_boundary(const t_param params, t_speed_aos *cells,
                    t_speed *tmp_cells, float *inlets);