
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{

  N2P_Precond :: N2P_Precond (shared_ptr<PETScBaseMatrix> mat, shared_ptr<ngs::BaseMatrix> _ngs_pc)
  : PETScPreconditioner(mat, _ngs_pc)
  {
    shared_ptr<ngs::ParallelDofs> pardofs;
    if (auto pc = dynamic_pointer_cast<ngs::Preconditioner>(ngs_mat))
      { pardofs = pc->GetAMatrix().GetParallelDofs(); }
    else
      { pardofs = GetNGsMat()->GetParallelDofs(); }
    
    PCCreate(pardofs->GetCommunicator(), &petsc_pc);

    PCSetType(petsc_pc, PCSHELL);

    PCShellSetContext(petsc_pc, (void*)this);

    // TODO: this is complete garbage...
    PCSetOperators(GetPETScPC(), GetAMat()->GetPETScMat(), GetAMat()->GetPETScMat());

    PCShellSetApply(GetPETScPC(), this->ApplyPC);

    PCSetUp(GetPETScPC());
  }

  PetscErrorCode N2P_Precond :: ApplyPC (PETScPC pc, PETScVec x, PETScVec y)
  {
    void* ptr; PCShellGetContext(pc, &ptr);
    auto & n2p_pre = *( (N2P_Precond*) ptr);

    n2p_pre.GetRowMap()->PETSc2NGs(*n2p_pre.row_hvec, x);

    // cout << "RHS: " << endl << *n2p_pre.row_hvec << endl;
    
    n2p_pre.GetNGsMat()->Mult(*n2p_pre.row_hvec, *n2p_pre.col_hvec);

    // cout << "SOL: " << endl << *n2p_pre.col_hvec << endl;

    n2p_pre.GetRowMap()->NGs2PETSc(*n2p_pre.col_hvec, y);

    return PetscErrorCode(0);
  }


  void ExportPC (py::module & m)
  {
    py::class_<PETScPreconditioner, shared_ptr<PETScPreconditioner>, FlatPETScMatrix>
      (m, "PETScPreconditioner", "not much here...");

    py::class_<N2P_Precond, shared_ptr<N2P_Precond>, PETScPreconditioner>
      (m, "NGs2PETSc_PC", "NGSolve-Preconditioner, wrapped to PETSc")
      .def( py::init<>
	    ([] (shared_ptr<PETScBaseMatrix> mat, shared_ptr<ngs::BaseMatrix> pc)
	     {
	       return make_shared<N2P_Precond>(mat, pc);
	     }), py::arg("mat"), py::arg("pc"));
	    
    
  }


} // namespace ngs_petsc_interface
