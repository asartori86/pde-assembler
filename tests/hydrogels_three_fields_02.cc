#include "pidomus.h"
#include "interfaces/hydrogels_three_fields.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 3;
  const int spacedim = 3;

  HydroGelThreeFields<dim,spacedim,LADealII> gel;
  piDoMUS<dim,spacedim,LADealII> solver ("piDoMUS",gel);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/hydrogels_three_fields_02.prm", "used_parameters.prm");


  solver.run ();

  auto &sol = solver.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << sol[i] << std::endl;

  return 0;
}
