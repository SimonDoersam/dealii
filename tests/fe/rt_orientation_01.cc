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

/*
 * This class inherits from the FE class since it is supposed to test
 * private/protected data members through an interface.
 */
template <int dim>
class RaviartThomas_PermutationSignTest : public FE_RaviartThomas<dim>
{
public:
  RaviartThomas_PermutationSignTest(const unsigned int _degree)
    : FE_RaviartThomas<dim>(_degree)
  {}

  void
  plot_all_info(const Triangulation<dim> &tria)
  {
    deallog << "*********************************************************"
            << std::endl
            << "Testing shape function permuation for   " << this->get_name()
            << "   elements of (polynomial) degree   " << this->degree
            << std::endl
            << std::endl;

    const unsigned int n_dofs_per_cell   = this->n_dofs_per_cell();
    const unsigned int n_dofs_per_face   = this->n_dofs_per_face();
    const unsigned int n_dofs_per_quad   = this->n_dofs_per_quad();
    const unsigned int n_dofs_per_line   = this->n_dofs_per_line();
    const unsigned int n_dofs_per_vertex = this->n_dofs_per_vertex();

    const unsigned int first_line_index = this->get_first_line_index();
    const unsigned int first_quad_index = this->get_first_quad_index();
    const unsigned int first_face_line_index =
      this->get_first_face_line_index();
    const unsigned int first_face_quad_index =
      this->get_first_face_quad_index();

    deallog << "Element Info:  " << std::endl
            << "   n_dofs_per_cell      : " << n_dofs_per_cell << std::endl
            << "   n_dofs_per_face      : " << n_dofs_per_face << std::endl
            << "   n_dofs_per_quad      : " << n_dofs_per_quad << std::endl
            << "   n_dofs_per_line      : " << n_dofs_per_line << std::endl
            << "   n_dofs_per_vertex    : " << n_dofs_per_vertex << std::endl
            << "   first_line_index     : " << first_line_index << std::endl
            << "   first_quad_index     : " << first_quad_index << std::endl
            << "   first_face_line_index: " << first_face_line_index
            << std::endl
            << "   first_face_quad_index: " << first_face_quad_index
            << std::endl
            << std::endl
            << std::endl;

    for (const auto &cell : tria.active_cell_iterators())
      {
        CellId current_cell_id(cell->id());

        deallog
          << "CellId = " << current_cell_id << std::endl
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

        /*
         * Now matching dofs and sign
         */

        for (unsigned int local_cell_dof_index = 0;
             local_cell_dof_index < this->n_dofs_per_cell();
             ++local_cell_dof_index)
          {
            std::pair<unsigned int, bool> matched_dof_and_sign =
              get_matched_dof(local_cell_dof_index, cell);

            if (((local_cell_dof_index - matched_dof_and_sign.first) != 0) ||
                matched_dof_and_sign.second)
              {
                deallog << "   " << local_cell_dof_index << " ---> "
                        << matched_dof_and_sign.first
                        << "   sign change = " << matched_dof_and_sign.second
                        << std::endl;
              }
          }
      } // cell
  }

private:
  std::pair<unsigned int, bool>
  get_matched_dof(const unsigned int cell_dof_index,
                  const typename Triangulation<dim>::cell_iterator &cell)
  {
    unsigned int matched_dof_index = cell_dof_index;
    bool         sign_flip         = false;

    const unsigned int n_face_dofs =
      GeometryInfo<dim>::faces_per_cell * this->n_dofs_per_face();

    /*
     * Assume that there are no vertex and no line dofs and that face dofs come
     * first
     */
    if (cell_dof_index < n_face_dofs)
      {
        /*
         * Find the face belonging to this dof_index. This is integer
         * division.
         */
        unsigned int face_index_from_dof_index =
          cell_dof_index / (this->n_dofs_per_quad());

        unsigned int local_quad_dof_index =
          cell_dof_index % this->n_dofs_per_quad();

        // Correct the dof_sign if necessary
        std::pair<unsigned int, bool> offset_and_sign =
          adjust_quad_dof_sign_and_index_for_face_orientation(
            local_quad_dof_index,
            face_index_from_dof_index,
            cell->face_orientation(face_index_from_dof_index),
            cell->face_flip(face_index_from_dof_index),
            cell->face_rotation(face_index_from_dof_index));

        return std::pair<unsigned int, bool>(cell_dof_index +
                                               offset_and_sign.first,
                                             offset_and_sign.second);
      }

    return std::pair<unsigned int, bool>(cell_dof_index, false);
  }


  std::pair<unsigned int, bool>
  adjust_quad_dof_sign_and_index_for_face_orientation(
    const unsigned int index,
    const unsigned int face,
    const bool         face_orientation,
    const bool         face_flip,
    const bool         face_rotation) const
  {
    // do nothing in 1D and 2D
    if (dim < 3)
      return std::pair<unsigned int, bool>(0, false);

    // The exception are discontinuous
    // elements for which there should be no
    // face dofs anyway (i.e. dofs_per_quad==0
    // in 3d), so we don't need the table, but
    // the function should also not have been
    // called
    AssertIndexRange(index, this->n_dofs_per_quad(face));
    Assert(this->adjust_quad_dof_sign_for_face_orientation_table
               [this->n_unique_quads() == 1 ? 0 : face]
                 .n_elements() ==
             (internal::ReferenceCell::get_cell(this->reference_cell())
                    .face_reference_cell(face) == ReferenceCell::Quadrilateral ?
                8 :
                6) *
               this->n_dofs_per_quad(face),
           ExcInternalError());

    bool sign_flip =
      this->adjust_quad_dof_sign_for_face_orientation_table
        [this->n_unique_quads() == 1 ? 0 : face](
          index, 4 * face_orientation + 2 * face_flip + face_rotation);

    unsigned int offset =
      this->adjust_quad_dof_index_for_face_orientation_table
        [this->n_unique_quads() == 1 ? 0 : face](
          index, 4 * face_orientation + 2 * face_flip + face_rotation);

    return std::pair<unsigned int, bool>(offset, sign_flip);
  }
};



int
main(int /*argc*/, char ** /*argv*/)
{
  deallog << std::setprecision(PRECISION);
  deallog << std::fixed;
  deallog.attach(logfile);

  /*
   * 3D test
   */
  const int          dim = 3;
  Triangulation<dim> tria_test;

  deallog << "Testing 3D case:" << std::endl;
  deallog << "(Moebius strip):" << std::endl;

  for (unsigned int degree = 0; degree < 3; ++degree)
    {
      RaviartThomas_PermutationSignTest<dim> fe(degree);

      for (unsigned int n_rotations = 0; n_rotations < 4; ++n_rotations)
        {
          tria_test.clear();

          // Mesh with roated faces
          GridGenerator::moebius(tria_test,
                                 /* n_cells */ 8,
                                 /* n_rotations by pi/2*/ n_rotations,
                                 /* R */ 2,
                                 /* r */ 0.5);

          /*
           * Plot all info about how dofs would change in each of the two
           * cells of the test triangulation.
           */
          fe.plot_all_info(tria_test);

          deallog << std::endl << std::endl;
        } // ++n_rotations
    }     // degree++

  deallog << "****************" << std::endl;
  deallog << "(hyper shell):" << std::endl;
  deallog << "****************" << std::endl;

  for (unsigned int degree = 0; degree < 3; ++degree)
    {
      RaviartThomas_PermutationSignTest<dim> fe(degree);

      tria_test.clear();

      // Mesh with non-standard faces
      GridGenerator::hyper_shell(tria_test,
                                 Point<dim>(),
                                 1,
                                 2,
                                 /* n_cells */ 6,
                                 /* colorize */ false);

      /*
       * Plot all info about how dofs would change in each of the two
       * cells of the test triangulation.
       */
      fe.plot_all_info(tria_test);

      deallog << std::endl << std::endl;
    } // degree++

  deallog << "****************" << std::endl;

  deallog << std::endl << "Testing 3D case done." << std::endl;

  return 0;
}
