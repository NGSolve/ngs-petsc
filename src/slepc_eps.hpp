#ifdef USE_SLEPC

#ifndef FILE_NGSPETSC_EPS_HPP
#define FILE_NGSPETSC_EPS_HPP

#include "petsc_interface.hpp"
#include "slepc_interface.hpp"

namespace ngs_petsc_interface
{

  /** EPS (EigenProblemSolver) from SLEPc, for linear Eigen-Problems of the form
      A x = lambda B x **/
  class SLEPcEPS
  {
  protected:
    EPS eps;
    shared_ptr<PETScBaseMatrix> A, B;
  public:

    SLEPcEPS (shared_ptr<PETScBaseMatrix> _A, shared_ptr<PETScBaseMatrix> _B, FlatArray<string> _opts, bool _finalize, string _name = "");
    
    ~SLEPcEPS ();

    void Finalize ();

    void Solve ();
    
    /** Get EigenVectors/Values **/

    size_t GetNConvergedEVs ();
    bool IsConverged ();
    EPSConvergedReason GetConvergedReason ();

    Complex GetEigenValue (size_t num);
    Array<Complex> GetEigenValues ();

    /** eigen-value, real part of Eingen-Vector , complex part of Eingen-Vector **/
    std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>> GetEigenPair (size_t num);
    Array<std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>>> GetEigenPairs (size_t num);

  }; // class SLEPcEPS
  
} // namespace ngs_petsc_interface

#endif // FILE_NGSPETSC_EPS_HPP
#endif // USE_SLEPC
