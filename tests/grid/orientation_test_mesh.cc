// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


/*
 * Small test to analyse the equivalence of the normal component
 * on the element edges for the Raviart-Thomas elements.
 */



#include "../tests.h"

#define PRECISION 8

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/grid/grid_generator.h>

// STL
#include <map>

std::ofstream logfile("output");

/*
 * Check if the shape functions on meshes with cells that have faces that are
 * either not in standard orientation or rotated (8 cases in total) are permuted
 * in the correct way after mapping to physical cells. Aslo check the correct
 * sign change.
 *
 * We check for RaviartThomas order 0, 1, 2 (degree = order + 1) in all eight
 * cases in 2D and 3D.
 */

using namespace dealii;

template <int dim>
void
plot_all_info(const Triangulation<dim> &tria)
{
  for (const auto &cell : tria.active_cell_iterators())
    {
      CellId current_cell_id(cell->id());

      deallog << "CellId = " << current_cell_id << std::endl
              << "   {index -> face_orientation | face_flip | face_rotation}: "
              << std::endl;
      for (unsigned int face_index = 0;
           face_index < GeometryInfo<dim>::faces_per_cell;
           ++face_index)
        {
          deallog << "      {" << face_index << " -> "
                  << cell->face_orientation(face_index) << " | "
                  << cell->face_flip(face_index) << " | "
                  << cell->face_rotation(face_index) << " } " << std::endl;
        } // face_index

      deallog << "   line orientation: {  ";
      for (unsigned int line_index = 0;
           line_index < GeometryInfo<dim>::lines_per_cell;
           ++line_index)
        {
          //        	  auto line = cell->line(line_index);
          deallog << cell->line_orientation(line_index) << "  ";
        } // line_index
      deallog << "}" << std::endl << std::endl;
    } // cell
}


int
main(int /*argc*/, char ** /*argv*/)
{
  deallog << std::setprecision(PRECISION);
  deallog << std::fixed;
  deallog.attach(logfile);

  /*
   * 3D test only
   */
  const int          dim = 3;
  Triangulation<dim> tria_test;

  deallog << "Testing 3D mesh for orientation tests:" << std::endl;

  for (unsigned int config_switch = 0; config_switch < 8; ++config_switch)
    {
      tria_test.clear();

      bool face_orientation = (((config_switch / 4) % 2) == 1);
      bool face_flip        = (((config_switch / 2) % 2) == 1);
      bool face_rotation    = ((config_switch % 2) == 1);

      bool manipulate_first_cube = true;

      GridGenerator::orientation_test_mesh(tria_test,
                                           face_orientation,
                                           face_flip,
                                           face_rotation,
                                           manipulate_first_cube);

      plot_all_info(tria_test);

      deallog << "****************" << std::endl;
    } // config_switch++

  deallog << "Testing 3D orientation mesh done." << std::endl;

  return 0;
}
