#include <pidomus.h>
#include "streamer_interface.h"
#include <deal2lkit/imex_stepper.h>
#include <deal2lkit/ida_interface.h>

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);


  const int dim = 1;
  const int spacedim = 3;

  // for serial version using a direct solver use uncomment these two
  // lines
   StreamerModel<dim,spacedim,LADealII> problem;
   
   piDoMUS<dim,spacedim,LADealII> solver ("pidomus",problem);

  // for parallel version using an iterative solver uncomment these
  // two lines
//  HeatEquation<dim,spacedim,LATrilinos> problem;
//  piDoMUS<dim,spacedim,LATrilinos> solver ("pidomus",problem);

  IMEXStepper<typename LADealII::VectorType> imex{"Outer imex", MPI_COMM_WORLD};

  IDAInterface<typename LADealII::VectorType> ida("IDA Solver Parameters", MPI_COMM_WORLD);

  ParameterAcceptor::initialize("streamer.prm", "used_parameters.prm");


  solver.current_alpha = imex.get_alpha();
  imex.create_new_vector = solver.lambdas.create_new_vector;
  imex.residual = solver.lambdas.residual;
  imex.setup_jacobian = solver.lambdas.setup_jacobian;
  imex.solver_should_restart = solver.lambdas.solver_should_restart;
  imex.solve_jacobian_system = solver.lambdas.solve_jacobian_system;
  imex.output_step = solver.lambdas.output_step;
  imex.get_lumped_mass_matrix = solver.lambdas.get_lumped_mass_matrix;
  imex.jacobian_vmult = solver.lambdas.jacobian_vmult;

  ida.create_new_vector = solver.lambdas.create_new_vector;
  ida.residual = solver.lambdas.residual;
  ida.setup_jacobian = solver.lambdas.setup_jacobian;
  ida.solver_should_restart = solver.lambdas.solver_should_restart;
  ida.solve_jacobian_system = solver.lambdas.solve_jacobian_system;
  ida.output_step = solver.lambdas.output_step;
  ida.differential_components = solver.lambdas.differential_components;


//  solver.run ();

  solver.init();
  ida.solve_dae(solver.solution, solver.solution_dot);
  // imex.solve_dae(solver.solution, solver.solution_dot);

  

  return 0;
}
