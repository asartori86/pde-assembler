#ifndef _pidoums_poisson_h_
#define _pidoums_poisson_h_

#include "pde_system_interface.h"

#include <deal2lkit/sacado_tools.h>


template <int dim, int spacedim, typename LAC=LADealII>
class StreamerModel : public PDESystemInterface<dim,spacedim, StreamerModel<dim,spacedim,LAC>, LAC>
{

public:
  virtual ~StreamerModel () {}
  StreamerModel ();

  mutable double sigma_x = 0;
  mutable double old_time = -1;

  // interface with the PDESystemInterface :)

  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values             |
            update_gradients          | /* this is the new entry */
            update_quadrature_points  |
            update_normal_vectors     |
            update_JxW_values);
  }


void declare_parameters(ParameterHandler& prm){

  PDESystemInterface<dim,spacedim,StreamerModel<dim,spacedim,LAC>, LAC >::declare_parameters(prm);
  this->add_parameter(prm, &rho, "rho linear", "0.0", Patterns::Double(0.0));
  this->add_parameter(prm, &k, "elasticity constant", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &eta, "viscosity", "1.0", Patterns::Double(0.0));


}

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;

 private:
double rho;
double k;
double eta;
};

template <int dim, int spacedim, typename LAC>
StreamerModel<dim,spacedim, LAC>::
StreamerModel():
  PDESystemInterface<dim,spacedim,StreamerModel<dim,spacedim,LAC>, LAC >("Streamer model",
      9,1,
      "FESystem[FE_Q(1)^3-FE_Q(1)^3-FE_Q(1)^3]",
      "u,u,u,v,v,v,ue,ue,ue","1,1,1")
{}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
StreamerModel<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{

  const FEValuesExtractors::Vector displ(0);
  const FEValuesExtractors::Vector vel(spacedim);
  const FEValuesExtractors::Vector uel(spacedim+spacedim);
  static const double L = 1e-3;
  ResidualType rt = 0; // dummy number to define the type of variables
  this->reinit (rt, cell, fe_cache);
  const double alpha = this->get_alpha();
  fe_cache.cache_local_solution_vector("explicit_solution_dot", this->get_locally_relevant_previous_explicit_solution(),alpha);
  auto &uts = fe_cache.get_values("solution_dot", "u", displ, rt);
  auto &graduts = fe_cache.get_gradients("solution_dot", "du", displ, rt);
  auto &gradus = fe_cache.get_gradients("solution", "du", displ, rt);

  auto &gradvs = fe_cache.get_gradients("solution", "dv", vel, rt);
  auto &vs     = fe_cache.get_values("solution", "v", vel, rt);
  auto &vts    = fe_cache.get_values("solution_dot", "v_dot", vel, rt);

  auto &gradues = fe_cache.get_gradients("solution", "due", uel, rt);
  auto &graduets    = fe_cache.get_gradients("solution_dot", "due_dot", uel, rt);
  auto &uets = fe_cache.get_values("explicit_solution_dot", "uet", uel, alpha);

  // auto &us = fe_cache.get_values("solution", "u", displ, rt);
  // auto &us_2 = fe_cache.get_values("previous_explicit_solution", "u", displ, alpha);
  // auto &us_1 = fe_cache.get_values("explicit_solution", "u", displ, alpha);

  const unsigned int n_q_points = uts.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

//  const double vv = 1e-2;
//  const Tensor<1,spacedim> vv;
//  vv[0] = 1e-2;
//  vv[1] = 0;
//  vv[2] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &ut = uts[q];
      auto &v = vs[q];
      auto &vt = vts[q];
      auto &gradu = gradus[q];
      auto &gradv = gradvs[q];

      auto &uet = uets[q];
      auto &gradue = gradues[q];
      auto &graduet = graduets[q];
      
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
	  // const double k = 1.;
          auto phi_u = fev[displ].value(i,q);
          auto phi_ue = fev[uel].value(i,q);
          auto grad_phi_u = fev[displ].gradient(i,q);
	  auto grad_phi_ue = fev[uel].gradient(i,q);
          auto phi_v = fev[vel].value(i,q);
          auto grad_phi_v = fev[vel].gradient(i,q);
          local_residuals[0][i] += (
				    k*scalar_product(gradue,grad_phi_ue) +
				    
				    // uet*phi_ue +
				    
				    rho*vt*phi_u +
				    1e5*(v-ut)*phi_v + 

				    eta*scalar_product(gradv,grad_phi_u)

				    - eta*scalar_product(graduet,grad_phi_u)
				    
				    +ut*phi_u

				    )*JxW[q];
        }

      (void)compute_only_system_terms;

    }

}


#endif
