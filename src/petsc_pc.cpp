
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{


  PETScBasePrecond :: PETScBasePrecond (MPI_Comm comm, string _name, FlatArray<string> _petsc_options)
  {
    PCCreate(comm, &petsc_pc);

    name = (_name.size()) ? name : GetDefaultId();
    PCSetOptionsPrefix(petsc_pc, name.c_str());

    SetOptions(_petsc_options, name, NULL);
 }


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

    petsc_rhs = GetAMat()->GetRowMap()->CreatePETScVector();
    petsc_sol = GetAMat()->GetColMap()->CreatePETScVector();
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


  PETSc2NGsPrecond :: PETSc2NGsPrecond (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags, const string _aname)
    : PETScBasePrecond(_bfa->GetFESpace()->IsParallel() ? MPI_Comm(_bfa->GetFESpace()->GetParallelDofs()->GetCommunicator()) : PETSC_COMM_SELF, _aname),
      ngs::Preconditioner (_bfa, _aflags, _aname), bfa(_bfa)
  {
    auto & petsc_options = flags.GetStringListFlag("petsc_pc_petsc_options");
    SetOptions(petsc_options, PETScBasePrecond::GetName(), NULL);
  }


  PETSc2NGsPrecond :: PETSc2NGsPrecond (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname)
    : PETScBasePrecond(MPI_COMM_NULL, ""), ngs::Preconditioner( &_apde, _aflags, _aname)
  { throw Exception("Not implemented! (Who still uses PDE files?)"); }


  void PETSc2NGsPrecond :: InitLevel (shared_ptr<ngs::BitArray> freedofs)
  {
    subset = freedofs;
  }


  void PETSc2NGsPrecond :: FinalizeLevel (const ngs::BaseMatrix * mat)
  {
    if (petsc_amat == nullptr) {
      if (mat == nullptr)
	{ throw Exception("PETSc2NGsPrecond has no matrix!"); }
      petsc_amat = make_shared<PETScMatrix>(shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter), subset, subset);
    }
    if (petsc_pmat == nullptr)
      { petsc_pmat = petsc_amat; }

    petsc_rhs = GetAMat()->GetRowMap()->CreatePETScVector();
    petsc_sol = GetAMat()->GetColMap()->CreatePETScVector();

    Finalize();
  }


  void PETSc2NGsPrecond :: Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const
  {
    static ngs::Timer tm("PETSc2NGsPrecond::Mult");
    static ngs::Timer ts("PETSc2NGsPrecond::PCApply");
    ngs::RegionTimer rts(tm);

    GetAMat()->GetRowMap()->NGs2PETSc(const_cast<ngs::BaseVector&>(x), petsc_rhs);

    {
      ngs::RegionTimer rts(ts);
      PCApply (GetPETScPC(), petsc_rhs, petsc_sol);
    }

    GetAMat()->GetColMap()->PETSc2NGs(y, petsc_sol);

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
    : PETSc2NGsPrecond (_petsc_amat, _petsc_pmat, _name, _petsc_options)
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


  PETScHypreAuxiliarySpacePC :: PETScHypreAuxiliarySpacePC (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags, const string _aname)
    : PETSc2NGsPrecond(_bfa, _aflags, _aname)
  {
    ;
  }

  PETScHypreAuxiliarySpacePC :: PETScHypreAuxiliarySpacePC (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname)
    : PETSc2NGsPrecond(_apde, _aflags, _aname)
  {
    ;
  }

  PETScHypreAuxiliarySpacePC :: PETScHypreAuxiliarySpacePC (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat,
							    string _name, FlatArray<string> _petsc_options)
    : PETSc2NGsPrecond(_petsc_amat, _petsc_pmat, _name, _petsc_options)
  {
    ;
  }


  void PETScHypreAuxiliarySpacePC :: SetConstantVectors (shared_ptr<ngs::BaseVector> _ozz, shared_ptr<ngs::BaseVector> _zoz, shared_ptr<ngs::BaseVector> _zzo)
  {
    ozz = _ozz; zoz = _zoz; zzo = _zzo;
    if (ozz != nullptr)
      { dimension = (zzo == nullptr) ? 2 : 3; }
  }


  void PETScHypreAuxiliarySpacePC :: FinalizeLevel (const ngs::BaseMatrix * mat)
  {
    // PC, dim (used for AMS), HD-embed, HD-embed components, HC-embed, HC-embed components
    if (HD_embed || HC_embed) {
      PCHYPRESetInterpolations(GetPETScPC(), dimension,
			       HD_embed ? HD_embed->GetPETScMat() : NULL, NULL,
			       HC_embed ? HC_embed->GetPETScMat() : NULL, NULL);
    }
    PETSc2NGsPrecond :: FinalizeLevel();
  }


  PETScHypreAMS :: PETScHypreAMS (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags, const string _aname)
    : PETScHypreAuxiliarySpacePC (_bfa, _aflags, _aname)
  {
    if (bfa == nullptr)
      { throw Exception("AMS Preconditioner with nullptr BLF, not sure how that happens..."); }

    if (dynamic_pointer_cast<ngs::HCurlHighOrderFESpace>(bfa->GetFESpace()) == nullptr)
      { throw Exception(string("AMS does not work for space") + typeid(_bfa->GetFESpace()).name() + string("!!")); }

    dimension = bfa->GetMeshAccess()->GetDimension();

    PCSetType(GetPETScPC(), PCHYPRE);
    PCHYPRESetType(GetPETScPC(), "ams");
  }


  PETScHypreAMS :: PETScHypreAMS (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname)
    : PETScHypreAuxiliarySpacePC (_apde, _aflags, _aname)
  {
    throw Exception("PETScHypreAMS PDE-constructor not implemented. (Who still uses PDE files ... ?");
  }


  PETScHypreAMS :: PETScHypreAMS (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat,
				  string _name, FlatArray<string> _petsc_options)
    : PETScHypreAuxiliarySpacePC (_petsc_amat, _petsc_pmat, _name, _petsc_options)
  {
    PCSetType(GetPETScPC(), PCHYPRE);
    PCHYPRESetType(GetPETScPC(), "ams");
  }


  void PETScHypreAMS :: InitLevel (shared_ptr<ngs::BitArray> freedofs)
  {
    PETSc2NGsPrecond :: InitLevel (freedofs);

    if ( bfa == nullptr )
      { throw Exception("PETScHypreAMS::InitLevel called, but we have no bilinearform. How did we get here??"); }

    if (grad_mat == nullptr) {
      auto feshc = dynamic_pointer_cast<ngs::HCurlHighOrderFESpace>(bfa->GetFESpace());
      auto fesh1 = feshc->CreateGradientSpace();
      shared_ptr<ngs::BaseMatrix> grad = feshc->CreateGradient(*fesh1);

      shared_ptr<ngs::BitArray> h1_subset;
      if (freedofs != bfa->GetFESpace()->GetFreeDofs()) { // we are being used as a coarse grid solver I think
	// adjust freedofs for the scalar h1-space
	h1_subset = make_shared<BitArray>(fesh1->GetNDof());
	h1_subset->Clear();
	auto spmat = dynamic_pointer_cast<ngs::BaseSparseMatrix>(grad);
	for (auto k : Range(spmat->Height())) {
	  if (subset->Test(k)) {
	    for (auto j : spmat->GetRowIndices(k))
	      { h1_subset->Set(j); }
	  }
	}
      }
      else
	{ h1_subset = fesh1->GetFreeDofs(); }

      if (feshc->IsParallel())
	{ grad = make_shared<ngs::ParallelMatrix>(grad, fesh1->GetParallelDofs(), feshc->GetParallelDofs(), ngs::PARALLEL_OP::C2C); }

      auto petsc_grad_mat = make_shared<PETScMatrix> (grad, h1_subset, subset);
    }
    
  }


  void PETScHypreAMS :: FinalizeLevel (const ngs::BaseMatrix * mat)
  {
    if (petsc_amat == nullptr) { // probably build via RegisterPreconditioner - convert matrix
      if (mat == nullptr)
	{ throw Exception("PETScHypreAMS::FinalizeLevel, have no matrix!"); }
      shared_ptr<NGs2PETScVecMap> vec_map;
      if (grad_mat != nullptr) // re-use vec-map if possible
	{ vec_map = grad_mat->GetColMap(); }
      petsc_pmat = petsc_amat = make_shared<PETScMatrix> (shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter),
							  subset, subset, vec_map, vec_map);
    }

    if (grad_mat == nullptr) {
      // should we try to construct this from bfa->GetFESpace? This sould have happened in InitLevel already ...
      throw Exception("The AMS Preconditioner needs the discrete H1-to-HCurl gradient Matrix!");
    }

    PCHYPRESetDiscreteGradient(GetPETScPC(), grad_mat->GetPETScMat());

    if (ozz != nullptr) {
      auto hc_map = GetAMat()->GetRowMap();
      PETScVec pozz = hc_map->CreatePETScVector(), pzoz = hc_map->CreatePETScVector(), pzzo = (zzo != nullptr) ? hc_map->CreatePETScVector() : nullptr;
      hc_map->NGs2PETSc(*ozz, pozz);
      hc_map->NGs2PETSc(*zoz, pzoz);
      if (zzo != nullptr)
	{ hc_map->NGs2PETSc(*zzo, pzzo); }
      PCHYPRESetEdgeConstantVectors(GetPETScPC(), pozz, pzoz, (zzo != nullptr) ? pzzo : NULL);
    }

    PETScHypreAuxiliarySpacePC :: FinalizeLevel (mat);
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
    : PETSc2NGsPrecond (_amat, _amat, _name, _petsc_options)
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


  ngs::RegisterPreconditioner<PETSc2NGsPrecond> registerPETSc2NGsPrecond("petsc_pc");

  ngs::RegisterPreconditioner<PETScHypreAMS> registerPETScHypreAMS("petsc_pc_hypre_ams");



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
	     }), py::arg("mat"), py::arg("pc"), py::arg("name") = string(""))
      .def("Finalize", [](shared_ptr<NGs2PETScPrecond> & pc)
	   { pc->Finalize(); } );
	    
    py::class_<PETSc2NGsPrecond, shared_ptr<PETSc2NGsPrecond>, PETScBasePrecond, ngs::BaseMatrix>
      (m, "PETSc2NGsPrecond", "A Preconditioner built in PETsc")
      .def (py::init<>
	    ([](shared_ptr<PETScBaseMatrix> amat, string name, py::dict petsc_options)
	     {
	       auto opt_array = Dict2SA(petsc_options);
	       return make_shared<PETSc2NGsPrecond>(amat, amat, name, opt_array);
	     }), py::arg("mat"), py::arg("name"), py::arg("petsc_options") = py::dict());

    py::class_<PETScHypreAuxiliarySpacePC, shared_ptr<PETScHypreAuxiliarySpacePC>, PETSc2NGsPrecond>
      (m, "HypreAuxiliarySpacePrecond", "ADS/AMS from hypre package")
      .def ("SetGradientMatrix"      , [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<PETScMatrix> mat)
	    { pc->SetGradientMatrix(mat); })
      .def ("SetHCurlEmbeddingMatrix", [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<PETScMatrix> mat)
	    { pc->SetHCurlEmbeddingMatrix(mat); })
      .def ("SetHDivEmbeddingMatrix" , [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<PETScMatrix> mat)
	    { pc->SetHDivEmbeddingMatrix(mat); })
      .def ("SetVectorLaplaceMatrix" , [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<PETScMatrix> mat)
	    { pc->SetVectorLaplaceMatrix(mat); })
      .def ("SetScalarLaplaceMatrix" , [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<PETScMatrix> mat)
	    { pc->SetScalarLaplaceMatrix(mat); })
      .def ("SetConstantVectors"     , [](shared_ptr<PETScHypreAuxiliarySpacePC> & pc, shared_ptr<ngs::BaseVector> ozz,
					  shared_ptr<ngs::BaseVector> zoz, shared_ptr<ngs::BaseVector> zzo)
	    { pc->SetConstantVectors(ozz, zoz, zzo); },
	    py::arg("ozz"), py::arg("zoz"), py::arg("zzo") = nullptr);

    py::class_<PETScHypreAMS, shared_ptr<PETScHypreAMS>, PETScHypreAuxiliarySpacePC>
      (m, "HypreAMSPrecond", "Auxiliary Maxwell Space Preconditioner from hypre")
      .def (py::init<>
	    ([](shared_ptr<PETScBaseMatrix> amat, string name, py::dict petsc_options,
		shared_ptr<PETScMatrix> grad_mat)
	     {
	       auto opt_array = Dict2SA(petsc_options);

	       auto pc = make_shared<PETScHypreAMS>(amat, amat, name, opt_array);

	       if (grad_mat != nullptr)
		 { pc->SetGradientMatrix(grad_mat); }

	       return pc;
	     }), py::arg("mat"), py::arg("name") = "", py::arg("petsc_options") = py::dict(),
	    py::arg("grad_mat") = nullptr);

    py::class_<PETScFieldSplitPC, shared_ptr<PETScFieldSplitPC>, PETSc2NGsPrecond>
      (m, "FieldSplitPrecond", "Fieldsplit Preconditioner from PETSc")
      .def (py::init<>
	    ([](shared_ptr<PETScBaseMatrix> amat, string name, py::dict petsc_options)
	     {
	       auto opt_array = Dict2SA(petsc_options);
	       return make_shared<PETScFieldSplitPC>(amat, name, opt_array);
	     }), py::arg("mat"), py::arg("name") = "", py::arg("petsc_options") = py::dict())
      .def("AddField", [](shared_ptr<PETScFieldSplitPC> & pc, size_t lower, size_t upper, string name, py::dict petsc_options)
	   {
	     auto opt_array = Dict2SA(petsc_options);
	     pc->AddField(make_shared<FSFieldRange>(pc->GetAMat(), lower, upper, name));
	   }, py::arg("lower"), py::arg("upper"), py::arg("name") = "", py::arg("petsc_options") = py::dict());
  }


} // namespace ngs_petsc_interface
