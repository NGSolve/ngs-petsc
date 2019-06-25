
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{


  PETScBasePrecond :: PETScBasePrecond (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat,
					string _name, FlatArray<string> _petsc_options)
    : petsc_amat(_petsc_amat), petsc_pmat(_petsc_pmat), name(_name)
  {
    auto ngs_mat = petsc_amat->GetNGsMat();
    shared_ptr<ngs::ParallelDofs> pardofs;
    if (auto pc = dynamic_pointer_cast<ngs::Preconditioner>(ngs_mat))
      { pardofs = pc->GetAMatrix().GetParallelDofs(); }
    else
      { pardofs = ngs_mat->GetParallelDofs(); }

    MPI_Comm comm = (pardofs != nullptr) ? MPI_Comm(pardofs->GetCommunicator()) : PETSC_COMM_SELF;

    PCCreate(comm, &petsc_pc);

    name = (_name.size()) ? name : GetDefaultId();
    PCSetOptionsPrefix(petsc_pc, name.c_str());

    SetOptions(_petsc_options, name, NULL);
  }

  void PETScBasePrecond :: Finalize ()
  {
    if (petsc_amat != nullptr) {
      if (petsc_pmat != nullptr)
	{ PCSetOperators(petsc_pc, petsc_amat->GetPETScMat(), petsc_pmat->GetPETScMat()); }
      else
	{ PCSetOperators(petsc_pc, petsc_amat->GetPETScMat(), petsc_amat->GetPETScMat()); }
    }

    PCSetFromOptions(petsc_pc);

    PCSetUp(petsc_pc);
  }


  // PETScPrecond :: PETScPrecond (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat, Array<string> options, string _name)
  //   : PETScBasePrecond (_petsc_amat, _petsc_pmat, _name)
  // {
  //   SetOptions(options, name, NULL);
  // }


  NGs2PETScPrecond :: NGs2PETScPrecond (shared_ptr<PETScBaseMatrix> _mat, shared_ptr<ngs::BaseMatrix> _ngs_pc,
					string _name, FlatArray<string> _petsc_options, bool _finalize)
    : PETScBasePrecond(_mat, _mat, _name, _petsc_options),
      FlatPETScMatrix(_ngs_pc, nullptr, nullptr, _mat->GetRowMap(), _mat->GetColMap())
  {
    // shared_ptr<ngs::ParallelDofs> pardofs;
    // if (auto pc = dynamic_pointer_cast<ngs::Preconditioner>(ngs_mat))
    //   { pardofs = pc->GetAMatrix().GetParallelDofs(); }
    // else
    //   { pardofs = mat->GetNGsMat()->GetParallelDofs(); }
    
    // PCCreate(pardofs->GetCommunicator(), &GetPETScPC());

    PCSetType(GetPETScPC(), PCSHELL);

    PCShellSetContext(GetPETScPC(), (void*)this);

    PCShellSetApply(GetPETScPC(), this->ApplyPC);

    if (_finalize)
      { Finalize(); }
  }

  PetscErrorCode NGs2PETScPrecond :: ApplyPC (PETScPC pc, PETScVec x, PETScVec y)
  {
    void* ptr; PCShellGetContext(pc, &ptr);
    auto & n2p_pre = *( (NGs2PETScPrecond*) ptr);

    n2p_pre.GetAMat()->GetRowMap()->PETSc2NGs(*n2p_pre.row_hvec, x);

    n2p_pre.GetNGsMat()->Mult(*n2p_pre.row_hvec, *n2p_pre.col_hvec);

    n2p_pre.GetAMat()->GetColMap()->NGs2PETSc(*n2p_pre.col_hvec, y);

    return PetscErrorCode(0);
  }


  PETScCompositePC :: PETScCompositePC (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat,
					string _name, FlatArray<string> _petsc_options)
    : PETScBasePrecond (_petsc_amat, _petsc_pmat, _name, _petsc_options)
  {
    PCSetType(GetPETScPC(), PCCOMPOSITE);
  }


  void PETScCompositePC :: AddPC (shared_ptr<NGs2PETScPrecond> component)
  {
    keep_alive.Append(component); // keep it alive

    PCCompositeAddPC(GetPETScPC(), PCSHELL);
    
    PetscInt num; PCCompositeGetNumberPC(GetPETScPC(), &num);
    PETScPC cpc; PCCompositeGetPC(GetPETScPC(), num-1, &cpc);
    PCShellSetContext(cpc, (void*)component.get());
    PCShellSetApply(cpc, component->ApplyPC);
  }


  FSField :: FSField (shared_ptr<PETScBasePrecond> _pc, string _name)
    : pc(_pc), name(_name)
  { ; }


  FSFieldRange :: FSFieldRange (shared_ptr<PETScBaseMatrix> _mat, size_t _first, size_t _next, string _name)
    : FSField(nullptr,  _name)
  {
    SetUpIS(_mat, _first, _next);
  }


  FSFieldRange :: FSFieldRange (shared_ptr<PETScBasePrecond> _pc, size_t _first, size_t _next, string _name)
    : FSField(_pc, _name)
  {
    SetUpIS(_pc->GetAMat(), _first, _next);
  }


  void FSFieldRange :: SetUpIS (shared_ptr<PETScBaseMatrix> mat, size_t _first, size_t _next)
  {
    auto row_map = mat->GetRowMap();
    auto dof_map = row_map->GetDOFMap();
    Array<PetscInt> inds(_next - _first);
    size_t cnt = 0;
    for (auto k : Range(_first, _next)) {
      auto gnum = dof_map[k];
      if (gnum != -1)
	{ inds[cnt++] = gnum; }
    }
    inds.SetSize(cnt);
    auto pardofs = row_map->GetParallelDofs();
    MPI_Comm comm = (pardofs != nullptr) ? MPI_Comm(pardofs->GetCommunicator()) : PETSC_COMM_SELF;
    ISCreateGeneral (comm, cnt, &inds[0], PETSC_COPY_VALUES, &is);
  }


  PETScFieldSplitPC :: PETScFieldSplitPC (shared_ptr<PETScBaseMatrix> _amat,
					  string _name, FlatArray<string> _petsc_options)
    : PETScBasePrecond (_amat, _amat, _name, _petsc_options)
  {
    PCSetType(GetPETScPC(), PCFIELDSPLIT);
  }


  void PETScFieldSplitPC :: AddField (shared_ptr<FSField> field)
  {
    fields.Append(field);
  }


  void PETScFieldSplitPC :: Finalize ()
  {
    PCSetOperators(GetPETScPC(), GetAMat()->GetPETScMat(), GetPMat()->GetPETScMat());

    Array<string> set_no_pc; // fields which already have a PC set
    for (auto k : Range(fields.Size())) {
      auto field = fields[k];
      auto name = field->GetName();
      if (auto f_pc = field->GetPC()) {
	string o = string("fieldsplit_") + name + "_pc_type none";
	set_no_pc.Append( o );
      }
      PCFieldSplitSetIS(GetPETScPC(), name.size() ? name.c_str() : NULL, field->GetIS());
    }

    SetOptions(set_no_pc, name, NULL); // no idea if I need the name prefix here or not??

    PCSetFromOptions(GetPETScPC());

    PCSetUp(GetPETScPC());

    // Now set the PCs we already had before
    KSP* ksps; PetscInt n;
    PCFieldSplitGetSubKSP(GetPETScPC(), &n, &ksps);
    for (auto k : Range(fields.Size())) {
      auto field = fields[k];
      auto f_pc = field->GetPC();
      if (f_pc == nullptr)
	{ continue; }
      KSP ksp = ksps[k];
      KSPSetPC(ksp, f_pc->GetPETScPC());
    }
    
    PetscFree(ksps);
  }


  void ExportPC (py::module & m)
  {
    extern Array<string> Dict2SA (py::dict & petsc_options);

    py::class_<PETScBasePrecond, shared_ptr<PETScBasePrecond>>
      (m, "PETScPrecond", "not much here...");

    py::class_<NGs2PETScPrecond, shared_ptr<NGs2PETScPrecond>, PETScBasePrecond>
      (m, "NGs2PETScPrecond", "NGSolve-Preconditioner, wrapped to PETSc")
      .def( py::init<>
	    ([] (shared_ptr<PETScBaseMatrix> mat, shared_ptr<ngs::BaseMatrix> pc, string name)
	     {
	       return make_shared<NGs2PETScPrecond>(mat, pc);
	     }), py::arg("mat"), py::arg("pc"), py::arg("name") = string(""));
	    
    py::class_<PETScFieldSplitPC, shared_ptr<PETScFieldSplitPC>, PETScBasePrecond>
      (m, "FieldSplitPrecond", "Fieldsplit Preconditioner from PETSc")
      .def (py::init<>
	    ([](shared_ptr<PETScBaseMatrix> amat, string name, py::dict petsc_options)
	     {
	       auto opt_array = Dict2SA(petsc_options);
	       return make_shared<PETScFieldSplitPC>(amat, name, opt_array);
	     }), py::arg("mat"), py::arg("name"), py::arg("petsc_options") = py::dict())
      .def("AddField", [](shared_ptr<PETScFieldSplitPC> & pc, size_t lower, size_t upper, string name, py::dict petsc_options)
	   {
	     auto opt_array = Dict2SA(petsc_options);
	     pc->AddField(make_shared<FSFieldRange>(pc->GetAMat(), lower, upper, name));
	   }, py::arg("lower"), py::arg("upper"), py::arg("name") = "", py::arg("petsc_options") = py::dict())
      .def("Finalize", [](shared_ptr<PETScFieldSplitPC> & pc)
	   { pc->Finalize(); } );

  }


} // namespace ngs_petsc_interface
