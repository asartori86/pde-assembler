#include "pidomus.h"
#include "interfaces/navier_stokes.h"
#include "tests.h"

/**
 * Test:     Navier Stokes interface.
 * Method:   Direct
 * Problem:  Non time depending Navier Stokes Equations
 * Exact solution:
 * \f[
 *    u=\big( 2*(x^2)*y, -2*x*(y^2) \big)
 *    \textrm{ and }p=xy;
 * \f]
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  NavierStokes<2,2,LADealII> energy(false);
  piDoMUS<2,2,LADealII> navier_stokes ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/navier_stokes_00.prm",
    "used_parameters.prm");

  navier_stokes.run ();

  auto& sol = navier_stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << std::fixed << std::setprecision(3) << sol[i] << std::endl ;
    }

  return 0;
}
