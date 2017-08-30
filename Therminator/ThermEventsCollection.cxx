///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#include "ThermEventsCollection.h"

#ifdef __ROOT__
ClassImp(ThermEventsCollection)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermEventsCollection::ThermEventsCollection(TString aEventsDirectory) :
  fNFiles(0),
  fNEvents(0),
  fEventsDirectory(aEventsDirectory),
  fFileNameCollection(0),
  fEventsCollection(0),

  fMixEvents(false),
  fNEventsToMix(5),
  fMixingEventsCollection(0),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fBuildUniqueParents(false),

  fUniqueLamParents(0),
  fUniqueALamParents(0),
  fUniquecLamParents(0),

  fUniqueK0Parents(0),

  fUniqueKchPParents(0),
  fUniqueKchMParents(0),
  fUniquecKchParents(0),

  fUniqueProtParents(0),
  fUniqueAProtParents(0),
  fUniquecProtParents(0),

  //LamKchP
  fSigToLamKchPTransform(0),
  fXiCToLamKchPTransform(0),
  fXi0ToLamKchPTransform(0),
  fOmegaToLamKchPTransform(0),
  fSigStPToLamKchPTransform(0),
  fSigStMToLamKchPTransform(0),
  fSigSt0ToLamKchPTransform(0),
  fLamKSt0ToLamKchPTransform(0),
  fSigKSt0ToLamKchPTransform(0),
  fXiCKSt0ToLamKchPTransform(0),
  fXi0KSt0ToLamKchPTransform(0),
  //-----
  fPairFractionsLamKchP(0),
  fParentsMatrixLamKchP(0),

  fPrimaryPairInfoLamKchP(0),
  fOtherPairInfoLamKchP(0),

  //ALamKchP
  fASigToALamKchPTransform(0),
  fAXiCToALamKchPTransform(0),
  fAXi0ToALamKchPTransform(0),
  fAOmegaToALamKchPTransform(0),
  fASigStMToALamKchPTransform(0),
  fASigStPToALamKchPTransform(0),
  fASigSt0ToALamKchPTransform(0),
  fALamKSt0ToALamKchPTransform(0),
  fASigKSt0ToALamKchPTransform(0),
  fAXiCKSt0ToALamKchPTransform(0),
  fAXi0KSt0ToALamKchPTransform(0),
  //-----
  fPairFractionsALamKchP(0),
  fParentsMatrixALamKchP(0),

  fPrimaryPairInfoALamKchP(0),
  fOtherPairInfoALamKchP(0),

  //LamKchM
  fSigToLamKchMTransform(0),
  fXiCToLamKchMTransform(0),
  fXi0ToLamKchMTransform(0),
  fOmegaToLamKchMTransform(0),
  fSigStPToLamKchMTransform(0),
  fSigStMToLamKchMTransform(0),
  fSigSt0ToLamKchMTransform(0),
  fLamAKSt0ToLamKchMTransform(0),
  fSigAKSt0ToLamKchMTransform(0),
  fXiCAKSt0ToLamKchMTransform(0),
  fXi0AKSt0ToLamKchMTransform(0),
  //-----
  fPairFractionsLamKchM(0),
  fParentsMatrixLamKchM(0),

  fPrimaryPairInfoLamKchM(0),
  fOtherPairInfoLamKchM(0),

  //ALamKchM
  fASigToALamKchMTransform(0),
  fAXiCToALamKchMTransform(0),
  fAXi0ToALamKchMTransform(0),
  fAOmegaToALamKchMTransform(0),
  fASigStMToALamKchMTransform(0),
  fASigStPToALamKchMTransform(0),
  fASigSt0ToALamKchMTransform(0),
  fALamAKSt0ToALamKchMTransform(0),
  fASigAKSt0ToALamKchMTransform(0),
  fAXiCAKSt0ToALamKchMTransform(0),
  fAXi0AKSt0ToALamKchMTransform(0),
  //-----
  fPairFractionsALamKchM(0),
  fParentsMatrixALamKchM(0),

  fPrimaryPairInfoALamKchM(0),
  fOtherPairInfoALamKchM(0),

  //LamK0
  fSigToLamK0Transform(0),
  fXiCToLamK0Transform(0),
  fXi0ToLamK0Transform(0),
  fOmegaToLamK0Transform(0),
  fSigStPToLamK0Transform(0),
  fSigStMToLamK0Transform(0),
  fSigSt0ToLamK0Transform(0),
  fLamKSt0ToLamK0Transform(0),
  fSigKSt0ToLamK0Transform(0),
  fXiCKSt0ToLamK0Transform(0),
  fXi0KSt0ToLamK0Transform(0),
  //-----
  fPairFractionsLamK0(0),
  fParentsMatrixLamK0(0),

  fPrimaryPairInfoLamK0(0),
  fOtherPairInfoLamK0(0),

  //ALamK0
  fASigToALamK0Transform(0),
  fAXiCToALamK0Transform(0),
  fAXi0ToALamK0Transform(0),
  fAOmegaToALamK0Transform(0),
  fASigStMToALamK0Transform(0),
  fASigStPToALamK0Transform(0),
  fASigSt0ToALamK0Transform(0),
  fALamKSt0ToALamK0Transform(0),
  fASigKSt0ToALamK0Transform(0),
  fAXiCKSt0ToALamK0Transform(0),
  fAXi0KSt0ToALamK0Transform(0),
  //-----
  fPairFractionsALamK0(0),
  fParentsMatrixALamK0(0),

  fPrimaryPairInfoALamK0(0),
  fOtherPairInfoALamK0(0),

  fSigToLamLamTransform(0),

  fProtonParents(0),
  fAProtonParents(0),

  fProtonRadii(0),
  fAProtonRadii(0),

  f2dProtonRadii(0),
  f2dAProtonRadii(0),

  fLamRadii(0),
  fALamRadii(0),

  f2dLamRadii(0),
  f2dALamRadii(0)

{
  //LamKchP
  fSigToLamKchPTransform = new TH2D("fSigToLamKchPTransform","fSigToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCToLamKchPTransform = new TH2D("fXiCToLamKchPTransform","fXiCToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0ToLamKchPTransform = new TH2D("fXi0ToLamKchPTransform","fXi0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fOmegaToLamKchPTransform = new TH2D("fOmegaToLamKchPTransform","fOmegaToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStPToLamKchPTransform = new TH2D("fSigStPToLamKchPTransform","fSigStPToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStMToLamKchPTransform = new TH2D("fSigStMToLamKchPTransform","fSigStMToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigSt0ToLamKchPTransform = new TH2D("fSigSt0ToLamKchPTransform","fSigSt0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fLamKSt0ToLamKchPTransform = new TH2D("fLamKSt0ToLamKchPTransform","fLamKSt0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigKSt0ToLamKchPTransform = new TH2D("fSigKSt0ToLamKchPTransform","fSigKSt0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCKSt0ToLamKchPTransform = new TH2D("fXiCKSt0ToLamKchPTransform","fXiCKSt0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0KSt0ToLamKchPTransform = new TH2D("fXi0KSt0ToLamKchPTransform","fXi0KSt0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsLamKchP = new TH1D("fPairFractionsLamKchP", "fPairFractionsLamKchP", 12, 0, 12);
  fParentsMatrixLamKchP = new TH2D("fParentsMatrixLamKchP", "fParentsMatrixLamKchP", 100, 0, 100, 135, 0, 135);

  //ALamKchP
  fASigToALamKchPTransform = new TH2D("fASigToALamKchPTransform","fASigToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCToALamKchPTransform = new TH2D("fAXiCToALamKchPTransform","fAXiCToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0ToALamKchPTransform = new TH2D("fAXi0ToALamKchPTransform","fAXi0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAOmegaToALamKchPTransform = new TH2D("fAOmegaToALamKchPTransform","fAOmegaToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStMToALamKchPTransform = new TH2D("fASigStMToALamKchPTransform","fASigStMToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStPToALamKchPTransform = new TH2D("fASigStPToALamKchPTransform","fASigStPToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigSt0ToALamKchPTransform = new TH2D("fASigSt0ToALamKchPTransform","fASigSt0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fALamKSt0ToALamKchPTransform = new TH2D("fALamKSt0ToALamKchPTransform","fALamKSt0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigKSt0ToALamKchPTransform = new TH2D("fASigKSt0ToALamKchPTransform","fASigKSt0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCKSt0ToALamKchPTransform = new TH2D("fAXiCKSt0ToALamKchPTransform","fAXiCKSt0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0KSt0ToALamKchPTransform = new TH2D("fAXi0KSt0ToALamKchPTransform","fAXi0KSt0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsALamKchP = new TH1D("fPairFractionsALamKchP", "fPairFractionsALamKchP", 12, 0, 12);
  fParentsMatrixALamKchP = new TH2D("fParentsMatrixALamKchP", "fParentsMatrixALamKchP", 100, 0, 100, 135, 0, 135);

  //LamKchM
  fSigToLamKchMTransform = new TH2D("fSigToLamKchMTransform","fSigToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCToLamKchMTransform = new TH2D("fXiCToLamKchMTransform","fXiCToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0ToLamKchMTransform = new TH2D("fXi0ToLamKchMTransform","fXi0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fOmegaToLamKchMTransform = new TH2D("fOmegaToLamKchMTransform","fOmegaToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStPToLamKchMTransform = new TH2D("fSigStPToLamKchMTransform","fSigStPToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStMToLamKchMTransform = new TH2D("fSigStMToLamKchMTransform","fSigStMToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigSt0ToLamKchMTransform = new TH2D("fSigSt0ToLamKchMTransform","fSigSt0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fLamAKSt0ToLamKchMTransform = new TH2D("fLamAKSt0ToLamKchMTransform","fLamAKSt0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigAKSt0ToLamKchMTransform = new TH2D("fSigAKSt0ToLamKchMTransform","fSigAKSt0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCAKSt0ToLamKchMTransform = new TH2D("fXiCAKSt0ToLamKchMTransform","fXiCAKSt0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0AKSt0ToLamKchMTransform = new TH2D("fXi0AKSt0ToLamKchMTransform","fXi0AKSt0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsLamKchM = new TH1D("fPairFractionsLamKchM", "fPairFractionsLamKchM", 12, 0, 12);
  fParentsMatrixLamKchM = new TH2D("fParentsMatrixLamKchM", "fParentsMatrixLamKchM", 100, 0, 100, 135, 0, 135);

  //ALamKchM
  fASigToALamKchMTransform = new TH2D("fASigToALamKchMTransform","fASigToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCToALamKchMTransform = new TH2D("fAXiCToALamKchMTransform","fAXiCToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0ToALamKchMTransform = new TH2D("fAXi0ToALamKchMTransform","fAXi0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAOmegaToALamKchMTransform = new TH2D("fAOmegaToALamKchMTransform","fAOmegaToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStMToALamKchMTransform = new TH2D("fASigStMToALamKchMTransform","fASigStMToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStPToALamKchMTransform = new TH2D("fASigStPToALamKchMTransform","fASigStPToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigSt0ToALamKchMTransform = new TH2D("fASigSt0ToALamKchMTransform","fASigSt0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fALamAKSt0ToALamKchMTransform = new TH2D("fALamAKSt0ToALamKchMTransform","fALamAKSt0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigAKSt0ToALamKchMTransform = new TH2D("fASigAKSt0ToALamKchMTransform","fASigAKSt0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCAKSt0ToALamKchMTransform = new TH2D("fAXiCAKSt0ToALamKchMTransform","fAXiCAKSt0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0AKSt0ToALamKchMTransform = new TH2D("fAXi0AKSt0ToALamKchMTransform","fAXi0AKSt0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsALamKchM = new TH1D("fPairFractionsALamKchM", "fPairFractionsALamKchM", 12, 0, 12);
  fParentsMatrixALamKchM = new TH2D("fParentsMatrixALamKchM", "fParentsMatrixALamKchM", 100, 0, 100, 135, 0, 135);

  //LamK0
  fSigToLamK0Transform = new TH2D("fSigToLamK0Transform","fSigToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCToLamK0Transform = new TH2D("fXiCToLamK0Transform","fXiCToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0ToLamK0Transform = new TH2D("fXi0ToLamK0Transform","fXi0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fOmegaToLamK0Transform = new TH2D("fOmegaToLamK0Transform","fOmegaToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStPToLamK0Transform = new TH2D("fSigStPToLamK0Transform","fSigStPToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigStMToLamK0Transform = new TH2D("fSigStMToLamK0Transform","fSigStMToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigSt0ToLamK0Transform = new TH2D("fSigSt0ToLamK0Transform","fSigSt0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fLamKSt0ToLamK0Transform = new TH2D("fLamKSt0ToLamK0Transform","fLamKSt0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fSigKSt0ToLamK0Transform = new TH2D("fSigKSt0ToLamK0Transform","fSigKSt0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCKSt0ToLamK0Transform = new TH2D("fXiCKSt0ToLamK0Transform","fXiCKSt0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0KSt0ToLamK0Transform = new TH2D("fXi0KSt0ToLamK0Transform","fXi0KSt0ToLamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsLamK0 = new TH1D("fPairFractionsLamK0", "fPairFractionsLamK0", 12, 0, 12);
  fParentsMatrixLamK0 = new TH2D("fParentsMatrixLamK0", "fParentsMatrixLamK0", 100, 0, 100, 135, 0, 135);

  //ALamK0
  fASigToALamK0Transform = new TH2D("fASigToALamK0Transform","fASigToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCToALamK0Transform = new TH2D("fAXiCToALamK0Transform","fAXiCToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0ToALamK0Transform = new TH2D("fAXi0ToALamK0Transform","fAXi0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAOmegaToALamK0Transform = new TH2D("fAOmegaToALamK0Transform","fAOmegaToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStMToALamK0Transform = new TH2D("fASigStMToALamK0Transform","fASigStMToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigStPToALamK0Transform = new TH2D("fASigStPToALamK0Transform","fASigStPToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigSt0ToALamK0Transform = new TH2D("fASigSt0ToALamK0Transform","fASigSt0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fALamKSt0ToALamK0Transform = new TH2D("fALamKSt0ToALamK0Transform","fALamKSt0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fASigKSt0ToALamK0Transform = new TH2D("fASigKSt0ToALamK0Transform","fASigKSt0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCKSt0ToALamK0Transform = new TH2D("fAXiCKSt0ToALamK0Transform","fAXiCKSt0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0KSt0ToALamK0Transform = new TH2D("fAXi0KSt0ToALamK0Transform","fAXi0KSt0ToALamK0Transform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  //-----
  fPairFractionsALamK0 = new TH1D("fPairFractionsALamK0", "fPairFractionsALamK0", 12, 0, 12);
  fParentsMatrixALamK0 = new TH2D("fParentsMatrixALamK0", "fParentsMatrixALamK0", 100, 0, 100, 135, 0, 135);

  //LamLam
  fSigToLamLamTransform = new TH2D("fSigToLamLamTransform","fSigToLamLamTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);

  //Protons
  fProtonParents = new TH1D("fProtonParents", "fProtonParents", 200, 0, 200);
  fAProtonParents = new TH1D("fAProtonParents", "fAProtonParents", 200, 0, 200);

  fProtonRadii = new TH1D("fProtonRadii", "fProtonRadii", 200, 0, 200);
  fAProtonRadii = new TH1D("fAProtonRadii", "fAProtonRadii", 200, 0, 200);

  f2dProtonRadii = new TH2D("f2dProtonRadii", "f2dProtonRadii", 200, 0, 200, 200, 0, 200);
  f2dAProtonRadii = new TH2D("f2dAProtonRadii", "f2dAProtonRadii", 200, 0, 200, 200, 0, 200);

  fLamRadii = new TH1D("fLamRadii", "fLamRadii", 200, 0, 200);
  fALamRadii = new TH1D("fALamRadii", "fALamRadii", 200, 0, 200);

  f2dLamRadii = new TH2D("f2dLamRadii", "f2dLamRadii", 200, 0, 200, 200, 0, 200);
  f2dALamRadii = new TH2D("f2dALamRadii", "f2dALamRadii", 200, 0, 200, 200, 0, 200);
}


//________________________________________________________________________________________________________________
ThermEventsCollection::~ThermEventsCollection()
{
  cout << "ThermEventsCollection object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
int ThermEventsCollection::ReturnEventIndex(unsigned int aEventID)
{
  int tEventIndex = -1;
  for(unsigned int i=0; i<fEventsCollection.size(); i++)
  {
    if(fEventsCollection[i].GetEventID() == aEventID)
    {
      tEventIndex = i;
      break;
    }
  }

  return tEventIndex;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(unsigned int i = 0; i < aRow.size(); i++)
  {
    if( i < aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == aRow.size()-1) aOutput << aRow[i] << endl;
    else 
    {
      cout << "SOMETHING IS WRONG!!!!!\n";
      assert(0);
    }
  }
}

//________________________________________________________________________________________________________________
vector<double> ThermEventsCollection::PackageV0ParticleForWriting(ThermV0Particle &aV0)
{
  vector<double> tReturnVector;
    tReturnVector.resize(53);
    // 27(ThermParticle) + 26(ThermV0Particle) = 53 total

  //------ThermParticle
  tReturnVector[0] = aV0.IsPrimordial();
  tReturnVector[1] = aV0.IsParticleOfInterest();

  tReturnVector[2] = aV0.GetMass();

  tReturnVector[3] = aV0.GetT();
  tReturnVector[4] = aV0.GetX();
  tReturnVector[5] = aV0.GetY();
  tReturnVector[6] = aV0.GetZ();

  tReturnVector[7] = aV0.GetE();
  tReturnVector[8] = aV0.GetPx();
  tReturnVector[9] = aV0.GetPy();
  tReturnVector[10] = aV0.GetPz();

  tReturnVector[11] = aV0.GetDecayed();
  tReturnVector[12] = aV0.GetPID();
  tReturnVector[13] = aV0.GetFatherPID();
  tReturnVector[14] = aV0.GetRootPID();
  tReturnVector[15] = aV0.GetEID();
  tReturnVector[16] = aV0.GetFatherEID();
  tReturnVector[17] = aV0.GetEventID();

  tReturnVector[18] = aV0.GetFatherMass();

  tReturnVector[19] = aV0.GetFatherT();
  tReturnVector[20] = aV0.GetFatherX();
  tReturnVector[21] = aV0.GetFatherY();
  tReturnVector[22] = aV0.GetFatherZ();

  tReturnVector[23] = aV0.GetFatherE();
  tReturnVector[24] = aV0.GetFatherPx();
  tReturnVector[25] = aV0.GetFatherPy();
  tReturnVector[26] = aV0.GetFatherPz();

  //------ThermV0Particle
  tReturnVector[27] = aV0.Daughter1Found();
  tReturnVector[28] = aV0.Daughter2Found();
  tReturnVector[29] = aV0.BothDaughtersFound();

  tReturnVector[30] = aV0.GoodV0();

  tReturnVector[31] = aV0.GetDaughter1PID();
  tReturnVector[32] = aV0.GetDaughter2PID();

  tReturnVector[33] = aV0.GetDaughter1EID();
  tReturnVector[34] = aV0.GetDaughter2EID();

  tReturnVector[35] = aV0.GetDaughter1Mass();

  tReturnVector[36] = aV0.GetDaughter1T();
  tReturnVector[37] = aV0.GetDaughter1X();
  tReturnVector[38] = aV0.GetDaughter1Y();
  tReturnVector[39] = aV0.GetDaughter1Z();

  tReturnVector[40] = aV0.GetDaughter1E();
  tReturnVector[41] = aV0.GetDaughter1Px();
  tReturnVector[42] = aV0.GetDaughter1Py();
  tReturnVector[43] = aV0.GetDaughter1Pz();

  tReturnVector[44] = aV0.GetDaughter2Mass();

  tReturnVector[45] = aV0.GetDaughter2T();
  tReturnVector[46] = aV0.GetDaughter2X();
  tReturnVector[47] = aV0.GetDaughter2Y();
  tReturnVector[48] = aV0.GetDaughter2Z();

  tReturnVector[49] = aV0.GetDaughter2E();
  tReturnVector[50] = aV0.GetDaughter2Px();
  tReturnVector[51] = aV0.GetDaughter2Py();
  tReturnVector[52] = aV0.GetDaughter2Pz();

  return tReturnVector;
}

//________________________________________________________________________________________________________________
vector<double> ThermEventsCollection::PackageParticleForWriting(ThermParticle &aParticle)
{
  vector<double> tReturnVector;
    tReturnVector.resize(27);
    // 27(ThermParticle)

  //------ThermParticle
  tReturnVector[0] = aParticle.IsPrimordial();
  tReturnVector[1] = aParticle.IsParticleOfInterest();

  tReturnVector[2] = aParticle.GetMass();

  tReturnVector[3] = aParticle.GetT();
  tReturnVector[4] = aParticle.GetX();
  tReturnVector[5] = aParticle.GetY();
  tReturnVector[6] = aParticle.GetZ();

  tReturnVector[7] = aParticle.GetE();
  tReturnVector[8] = aParticle.GetPx();
  tReturnVector[9] = aParticle.GetPy();
  tReturnVector[10] = aParticle.GetPz();

  tReturnVector[11] = aParticle.GetDecayed();
  tReturnVector[12] = aParticle.GetPID();
  tReturnVector[13] = aParticle.GetFatherPID();
  tReturnVector[14] = aParticle.GetRootPID();
  tReturnVector[15] = aParticle.GetEID();
  tReturnVector[16] = aParticle.GetFatherEID();
  tReturnVector[17] = aParticle.GetEventID();

  tReturnVector[18] = aParticle.GetFatherMass();

  tReturnVector[19] = aParticle.GetFatherT();
  tReturnVector[20] = aParticle.GetFatherX();
  tReturnVector[21] = aParticle.GetFatherY();
  tReturnVector[22] = aParticle.GetFatherZ();

  tReturnVector[23] = aParticle.GetFatherE();
  tReturnVector[24] = aParticle.GetFatherPx();
  tReturnVector[25] = aParticle.GetFatherPy();
  tReturnVector[26] = aParticle.GetFatherPz();

  return tReturnVector;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteThermEventV0s(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent)
{
  vector<ThermV0Particle> tV0ParticleVec = aThermEvent.GetV0ParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent.GetEventID() << " " << aParticleType << " " << tV0ParticleVec.size() << endl;
  for(unsigned int i=0; i<tV0ParticleVec.size(); i++)
  {
    tTempVec = PackageV0ParticleForWriting(tV0ParticleVec[i]);
    WriteRow(aOutput,tTempVec);
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteThermEventParticles(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent)
{
  vector<ThermParticle> tParticleVec = aThermEvent.GetParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent.GetEventID() << " " << aParticleType << " " << tParticleVec.size() << endl;
  for(unsigned int i=0; i<tParticleVec.size(); i++)
  {
    tTempVec = PackageParticleForWriting(tParticleVec[i]);
    WriteRow(aOutput,tTempVec);
  }
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteAllEventsParticlesOfType(TString aOutputName, ParticlePDGType aParticleType)
{
  ofstream tFileOut(aOutputName);

  for(unsigned int i=0; i<fEventsCollection.size(); i++)
  {
    if(aParticleType == kPDGLam || aParticleType == kPDGALam || aParticleType == kPDGK0) WriteThermEventV0s(tFileOut,aParticleType,fEventsCollection[i]);
    else if(aParticleType == kPDGKchP || aParticleType == kPDGKchM) WriteThermEventParticles(tFileOut,aParticleType,fEventsCollection[i]);
    else assert(0);
  }

  tFileOut.close();
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteAllEvents(TString aOutputNameBase)
{
  TString tNameLam = aOutputNameBase + TString("Lambda.txt");
  TString tNameALam = aOutputNameBase + TString("AntiLambda.txt");
  TString tNameK0 = aOutputNameBase + TString("K0.txt");
  TString tNameKchP = aOutputNameBase + TString("KchP.txt");
  TString tNameKchM = aOutputNameBase + TString("KchM.txt");

  WriteAllEventsParticlesOfType(tNameLam,kPDGLam);
    cout << "Done writing file: " << tNameLam << endl;

  WriteAllEventsParticlesOfType(tNameALam,kPDGALam);
    cout << "Done writing file: " << tNameALam << endl;

  WriteAllEventsParticlesOfType(tNameK0,kPDGK0);
    cout << "Done writing file: " << tNameK0 << endl;

  WriteAllEventsParticlesOfType(tNameKchP,kPDGKchP);
    cout << "Done writing file: " << tNameKchP << endl;

  WriteAllEventsParticlesOfType(tNameKchM,kPDGKchM);
    cout << "Done writing file: " << tNameKchM << endl;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractV0ParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType)
{
  assert(aPDGType == kPDGLam || aPDGType == kPDGALam || aPDGType == kPDGK0);

  ifstream tFileIn(aFileName);

  vector<ThermV0Particle> tTempV0Collection;
  vector<double> tTempParticle1dVec;

  unsigned int tEventID = 0;
  int tEventIndex;
  unsigned int tEntrySize = 53;  //size of V0 particle vector

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString))
  {
    tTempParticle1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempParticle1dVec.push_back(dbl);
    }

    if(tTempParticle1dVec.size() == 3)  //event header
    {           
      tCount++;
      
      if(tCount==1) tEventID = tTempParticle1dVec[0];
      else
      {
        tEventIndex = ReturnEventIndex(tEventID);
        if(tEventIndex == -1)  //Event does not already exist in fEventsCollection
        {
          ThermEvent tThermEvent;
          tThermEvent.SetEventID(tEventID);
          tThermEvent.SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex].SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
        }
        tTempV0Collection.clear();
      }
    tEventID = tTempParticle1dVec[0];
    }
    else if(tTempParticle1dVec.size() == tEntrySize)  //a V0 particle
    {
      tTempV0Collection.emplace_back(tTempParticle1dVec);
    }
    else
    {
      cout << "ERROR: Incorrect row size in ExtractV0ParticleCollectionsFromTxtFile" << endl;
      assert(0);
    }
  }
  tEventIndex = ReturnEventIndex(tEventID);
  if(tEventIndex == -1)
  {
    ThermEvent tThermEvent;
    tThermEvent.SetEventID(tEventID);
    tThermEvent.SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex].SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
  }
  tTempV0Collection.clear();
  tFileIn.close();
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType)
{
  assert(aPDGType == kPDGKchP || aPDGType == kPDGKchM);

  ifstream tFileIn(aFileName);

  vector<ThermParticle> tTempCollection;
  vector<double> tTempParticle1dVec;

  unsigned int tEventID = 0;
  int tEventIndex;
  unsigned int tEntrySize = 27;  //size of particle vector

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString))
  {
    tTempParticle1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempParticle1dVec.push_back(dbl);
    }

    if(tTempParticle1dVec.size() == 3)  //event header
    {           
      tCount++;
      
      if(tCount==1) tEventID = tTempParticle1dVec[0];
      else
      {
        tEventIndex = ReturnEventIndex(tEventID);
        if(tEventIndex == -1)  //Event does not already exist in fEventsCollection
        {
          ThermEvent tThermEvent;
          tThermEvent.SetEventID(tEventID);
          tThermEvent.SetParticleCollection(tEventID,aPDGType,tTempCollection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex].SetParticleCollection(tEventID,aPDGType,tTempCollection);
        }
        tTempCollection.clear();
      }
    tEventID = tTempParticle1dVec[0];
    }
    else if(tTempParticle1dVec.size() == tEntrySize)  //a particle
    {
      tTempCollection.emplace_back(tTempParticle1dVec);
    }
    else
    {
      cout << "ERROR: Incorrect row size in ExtractParticleCollectionsFromTxtFile" << endl;
      assert(0);
    }
  }
  tEventIndex = ReturnEventIndex(tEventID);
  if(tEventIndex == -1)
  {
    ThermEvent tThermEvent;
    tThermEvent.SetEventID(tEventID);
    tThermEvent.SetParticleCollection(tEventID,aPDGType,tTempCollection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex].SetParticleCollection(tEventID,aPDGType,tTempCollection);
  }
  tTempCollection.clear();
  tFileIn.close();
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractEventsFromAllTxtFiles(TString aFileLocationBase)
{
  TString tNameLam = aFileLocationBase + TString("Lambda.txt");
  TString tNameALam = aFileLocationBase + TString("AntiLambda.txt");
  TString tNameK0 = aFileLocationBase + TString("K0.txt");
  TString tNameKchP = aFileLocationBase + TString("KchP.txt");
  TString tNameKchM = aFileLocationBase + TString("KchM.txt");

  ExtractV0ParticleCollectionsFromTxtFile(tNameLam,kPDGLam);
    cout << "Done reading file: " << tNameLam << endl;

  ExtractV0ParticleCollectionsFromTxtFile(tNameALam,kPDGALam);
    cout << "Done reading file: " << tNameALam << endl;

  ExtractV0ParticleCollectionsFromTxtFile(tNameK0,kPDGK0);
    cout << "Done reading file: " << tNameK0 << endl;

  ExtractParticleCollectionsFromTxtFile(tNameKchP,kPDGKchP);
    cout << "Done reading file: " << tNameKchP << endl;

  ExtractParticleCollectionsFromTxtFile(tNameKchM,kPDGKchM);
    cout << "Done reading file: " << tNameKchM << endl;
}



//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractEventsFromRootFile(TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TTree *tTree = (TTree*)tFile->Get("particles");

  ParticleCoor *tParticleEntry = new ParticleCoor();
  TBranch *tParticleBranch = tTree->GetBranch("particle");
  tParticleBranch->SetAddress(tParticleEntry);

  int tNEvents = 1;
  unsigned int tEventID;
  ThermEvent tThermEvent;

  for(int i=0; i<tParticleBranch->GetEntries(); i++)
  {
    tParticleBranch->GetEntry(i);

    if(i==0) tEventID = tParticleEntry->eventid;

    if(tParticleEntry->eventid != tEventID)
    {
      tThermEvent.MatchDaughtersWithFathers();
      tThermEvent.FindAllFathers();
      tThermEvent.SetEventID(tEventID);
      fEventsCollection.push_back(tThermEvent);
      tThermEvent.ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    tThermEvent.PushBackThermParticle(tParticleEntry);
    if(tThermEvent.IsDaughterOfInterest(tParticleEntry)) tThermEvent.PushBackThermDaughterOfInterest(tParticleEntry);
    if(tThermEvent.IsParticleOfInterest(tParticleEntry)) tThermEvent.PushBackThermParticleOfInterest(tParticleEntry);
  }
  tThermEvent.MatchDaughtersWithFathers();
  tThermEvent.FindAllFathers();
  tThermEvent.SetEventID(tEventID);
  fEventsCollection.push_back(tThermEvent);

  fNEvents += tNEvents;

cout << "aFileLocation = " << aFileLocation << endl;
cout << "fEventsCollection.size() = " << fEventsCollection.size() << endl;
cout << "fNEvents = " << fNEvents << endl;


  tFile->Close();
  delete tFile;

  delete tParticleEntry;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractFromAllRootFiles(const char *aDirName, bool bBuildUniqueParents)
{
  fBuildUniqueParents = bBuildUniqueParents;

  fFileNameCollection.clear();
  TString tCompleteFilePath;

  TSystemDirectory tDir(aDirName,aDirName);
  TList* tFiles = tDir.GetListOfFiles();

  const char* tBeginningText = "event";
  const char* tEndingText = ".root";

  fNFiles = 0;

  if(tFiles)
  {
    TSystemFile* tFile;
    TString tName;
    TIter tIterNext(tFiles);

    while((tFile=(TSystemFile*)tIterNext()))
    {
      tName = tFile->GetName();
      if(!tFile->IsDirectory() && tName.BeginsWith(tBeginningText) && tName.EndsWith(tEndingText))
      {
        fNFiles++;
        fFileNameCollection.push_back(tName);
        tCompleteFilePath = TString(aDirName) + tName;
        ExtractEventsFromRootFile(tCompleteFilePath);

        BuildAllTransformMatrices();
        BuildAllPairFractionHistograms();
        fEventsCollection.clear();
        fEventsCollection.shrink_to_fit();

        cout << "fNFiles = " << fNFiles << endl << endl;
      }
    }
  }
  cout << "Total number of files = " << fNFiles << endl;
  cout << "Total number of events = " << fNEvents << endl << endl;
}


//________________________________________________________________________________________________________________
double ThermEventsCollection::GetFatherKStar(ThermParticle &aParticle1, ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  if(aUseParticleFather1)
  {
    px1 = aParticle1.GetFatherPx();
    py1 = aParticle1.GetFatherPy();
    pz1 = aParticle1.GetFatherPz();
    mass1 = aParticle1.GetFatherMass();
    E1 = aParticle1.GetFatherE();
  }
  else
  {
    px1 = aParticle1.GetPx();
    py1 = aParticle1.GetPy();
    pz1 = aParticle1.GetPz();
    mass1 = aParticle1.GetMass();
    E1 = aParticle1.GetE();
  }

  if(aUseParticleFather2)
  {
    px2 = aParticle2.GetFatherPx();
    py2 = aParticle2.GetFatherPy();
    pz2 = aParticle2.GetFatherPz();
    mass2 = aParticle2.GetFatherMass();
    E2 = aParticle2.GetFatherE();
  }
  else
  {
    px2 = aParticle2.GetPx();
    py2 = aParticle2.GetPy();
    pz2 = aParticle2.GetPz();
    mass2 = aParticle2.GetMass();
    E2 = aParticle2.GetE();
  }

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEventsCollection::GetKStar(ThermParticle &aParticle1, ThermParticle &aParticle2)
{
  return GetFatherKStar(aParticle1, aParticle2, false, false);
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;
  double tKStar, tFatherKStar;

  bool bUseParticleFather=true;
  bool bUseV0Father = true;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if((tV0.GetFatherPID() == aV0FatherType || aV0FatherType == kPDGNull) && tV0.GoodV0())
    {
      if(aV0FatherType == kPDGNull) bUseV0Father = false;  //because, by setting aV0FatherType==kPDGNull, I am saying I don't care where the V0 comes from
                                                           //In which case, I am also saying to not use the V0Father
      if(tV0.GetPID() == aV0FatherType)  //here, I want only primary V0s, which, of course, cannot have a father
      {
        assert(tV0.IsPrimordial());  //the V0 should only be primordial if that's what we're looking for
        bUseV0Father = false;
      }
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        if(tParticle.GetFatherPID() == aParticleFatherType || aParticleFatherType == kPDGNull)
        {
          if(aParticleFatherType == kPDGNull) bUseParticleFather=false; //similar explanation as above for if(aV0FatherType == kPDGNull) bUseV0Father = false;
          if(tParticle.GetPID() == aParticleFatherType)  //similar explanation as above for if(tV0.GetPID() == aV0FatherType)
          {
            assert(tParticle.IsPrimordial());
            bUseParticleFather = false;
          }

          tKStar = GetKStar(tParticle,tV0);
          tFatherKStar = GetFatherKStar(tParticle,tV0,bUseParticleFather,bUseV0Father);

          assert(tV0.DoubleCheckV0Attributes());
          aMatrix->Fill(tKStar,tFatherKStar);
        }
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  ThermV0Particle tV01, tV02;
  double tKStar, tFatherKStar;

  bool bUseV01Father = true;
  bool bUseV02Father = true;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if((tV01.GetFatherPID() == aV01FatherType || aV01FatherType == kPDGNull) && tV01.GoodV0())
    {
      if(aV01FatherType == kPDGNull) bUseV01Father = false;  //because, by setting aV01FatherType==kPDGNull, I am saying I don't care where V01 comes from
                                                           //In which case, I am also saying to not use the V0Father
      if(tV01.GetPID() == aV01FatherType)  //here, I want only primary V0s, which, of course, cannot have a father
      {
        assert(tV01.IsPrimordial());  //the V0 should only be primordial if that's what we're looking for
        bUseV01Father = false;
      }
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if((tV02.GetFatherPID() == aV02FatherType || aV02FatherType == kPDGNull) && tV02.GoodV0() 
           && !(tV02.GetEID()==tV01.GetEID() && tV02.GetEventID()==tV02.GetEventID()) ) //For instance, if I am doing LamLam w/o mixing events, I do not want to pair a Lam with itself
        {
          if(aV02FatherType == kPDGNull) bUseV02Father=false; //similar explanation as above for if(aV01FatherType == kPDGNull) bUseV01Father = false;
          if(tV02.GetPID() == aV02FatherType)  //similar explanation as above for if(tV01.GetPID() == aV01FatherType)
          {
            assert(tV02.IsPrimordial());
            bUseV02Father = false;
          }

          tKStar = GetKStar(tV01,tV02);
          tFatherKStar = GetFatherKStar(tV01,tV02,bUseV01Father,bUseV02Father);

          assert(tV01.DoubleCheckV0Attributes() && tV02.DoubleCheckV0Attributes());  
          aMatrix->Fill(tKStar,tFatherKStar);
        }
      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV0Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV0Type);
    if(!fMixEvents)
    {
      fMixingEventsCollection.clear();
      aParticleCollection = fEventsCollection[iEv].GetParticleCollection(aParticleType);
      FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
    }
    else
    {
      for(unsigned int iMixEv=0; iMixEv < fMixingEventsCollection.size(); iMixEv++)
      {
        aParticleCollection = fMixingEventsCollection[iMixEv].GetParticleCollection(aParticleType);
        FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
      }
    }

    if(fMixEvents)
    {
      assert(fMixingEventsCollection.size() <= fNEventsToMix);
      if(fMixingEventsCollection.size() == fNEventsToMix)
      {
        //delete fMixingEventsCollection.back();
        fMixingEventsCollection.pop_back();
      }

      fMixingEventsCollection.insert(fMixingEventsCollection.begin(), fEventsCollection[iEv]);
    }
  }

}



//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildTransformMatrixV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV01Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV01Type);
    if(!fMixEvents)
    {
      fMixingEventsCollection.clear();
      aV02Collection = fEventsCollection[iEv].GetV0ParticleCollection(aV02Type);
      FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
    }
    else
    {
      for(unsigned int iMixEv=0; iMixEv < fMixingEventsCollection.size(); iMixEv++)
      {
        aV02Collection = fMixingEventsCollection[iMixEv].GetV0ParticleCollection(aV02Type);
        FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
      }
    }

    if(fMixEvents)
    {
      assert(fMixingEventsCollection.size() <= fNEventsToMix);
      if(fMixingEventsCollection.size() == fNEventsToMix)
      {
        //delete fMixingEventsCollection.back();
        fMixingEventsCollection.pop_back();
      }

      fMixingEventsCollection.insert(fMixingEventsCollection.begin(), fEventsCollection[iEv]);
    }
  }
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildAllTransformMatrices()
{
  //LamKchP
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGSigma, fSigToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGXiC, fXiCToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGXi0, fXi0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGOmega, fOmegaToLamKchPTransform);

  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGSigStP, fSigStPToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGSigStM, fSigStMToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGNull, kPDGSigSt0, fSigSt0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGKSt0, kPDGLam, fLamKSt0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGKSt0, kPDGSigma, fSigKSt0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGKSt0, kPDGXiC, fXiCKSt0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGKSt0, kPDGXi0, fXi0KSt0ToLamKchPTransform);

  //ALamKchP
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGASigma, fASigToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGAXiC, fAXiCToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGAXi0, fAXi0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGAOmega, fAOmegaToALamKchPTransform);

  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGASigStM, fASigStMToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGASigStP, fASigStPToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGNull, kPDGASigSt0, fASigSt0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGKSt0, kPDGALam, fALamKSt0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGKSt0, kPDGASigma, fASigKSt0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGKSt0, kPDGAXiC, fAXiCKSt0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGKSt0, kPDGAXi0, fAXi0KSt0ToALamKchPTransform);

  //LamKchM
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGSigma, fSigToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGXiC, fXiCToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGXi0, fXi0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGOmega, fOmegaToLamKchMTransform);

  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGSigStP, fSigStPToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGSigStM, fSigStMToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGNull, kPDGSigSt0, fSigSt0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGAKSt0, kPDGLam, fLamAKSt0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGAKSt0, kPDGSigma, fSigAKSt0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGAKSt0, kPDGXiC, fXiCAKSt0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGAKSt0, kPDGXi0, fXi0AKSt0ToLamKchMTransform);

  //ALamKchM
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGASigma, fASigToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGAXiC, fAXiCToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGAXi0, fAXi0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGAOmega, fAOmegaToALamKchMTransform);

  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGASigStM, fASigStMToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGASigStP, fASigStPToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGNull, kPDGASigSt0, fASigSt0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAKSt0, kPDGALam, fALamAKSt0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAKSt0, kPDGASigma, fASigAKSt0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAKSt0, kPDGAXiC, fAXiCAKSt0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAKSt0, kPDGAXi0, fAXi0AKSt0ToALamKchMTransform);

  //LamK0
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGSigma, kPDGNull, fSigToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGXiC, kPDGNull, fXiCToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGXi0, kPDGNull, fXi0ToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGOmega, kPDGNull, fOmegaToLamK0Transform);

  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGSigStP, kPDGNull, fSigStPToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGSigStM, kPDGNull, fSigStMToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGSigSt0, kPDGNull, fSigSt0ToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGLam, kPDGKSt0, fLamKSt0ToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGSigma, kPDGKSt0, fSigKSt0ToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGXiC, kPDGKSt0, fXiCKSt0ToLamK0Transform);
  BuildTransformMatrixV0V0(kPDGLam, kPDGK0, kPDGXi0, kPDGKSt0, fXi0KSt0ToLamK0Transform);

  //ALamK0
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGASigma, kPDGNull, fASigToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGAXiC, kPDGNull, fAXiCToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGAXi0, kPDGNull, fAXi0ToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGAOmega, kPDGNull, fAOmegaToALamK0Transform);

  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGASigStM, kPDGNull, fASigStMToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGASigStP, kPDGNull, fASigStPToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGASigSt0, kPDGNull, fASigSt0ToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGALam, kPDGKSt0, fALamKSt0ToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGASigma, kPDGKSt0, fASigKSt0ToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGAXiC, kPDGKSt0, fAXiCKSt0ToALamK0Transform);
  BuildTransformMatrixV0V0(kPDGALam, kPDGK0, kPDGAXi0, kPDGKSt0, fAXi0KSt0ToALamK0Transform);

  //LamLam
  BuildTransformMatrixV0V0(kPDGLam, kPDGLam, kPDGSigma, kPDGNull, fSigToLamLamTransform);
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::SaveAllTransformMatrices(TString aSaveFileLocation)
{
  TFile *tFile = new TFile(aSaveFileLocation, "RECREATE");
  assert(tFile->IsOpen());


  //LamKchP
  fSigToLamKchPTransform->Write();
  fXiCToLamKchPTransform->Write();
  fXi0ToLamKchPTransform->Write();
  fOmegaToLamKchPTransform->Write();
  fSigStPToLamKchPTransform->Write();
  fSigStMToLamKchPTransform->Write();
  fSigSt0ToLamKchPTransform->Write();
  fLamKSt0ToLamKchPTransform->Write();
  fSigKSt0ToLamKchPTransform->Write();
  fXiCKSt0ToLamKchPTransform->Write();
  fXi0KSt0ToLamKchPTransform->Write();

  //ALamKchP
  fASigToALamKchPTransform->Write();
  fAXiCToALamKchPTransform->Write();
  fAXi0ToALamKchPTransform->Write();
  fAOmegaToALamKchPTransform->Write();
  fASigStMToALamKchPTransform->Write();
  fASigStPToALamKchPTransform->Write();
  fASigSt0ToALamKchPTransform->Write();
  fALamKSt0ToALamKchPTransform->Write();
  fASigKSt0ToALamKchPTransform->Write();
  fAXiCKSt0ToALamKchPTransform->Write();
  fAXi0KSt0ToALamKchPTransform->Write();

  //LamKchM
  fSigToLamKchMTransform->Write();
  fXiCToLamKchMTransform->Write();
  fXi0ToLamKchMTransform->Write();
  fOmegaToLamKchMTransform->Write();
  fSigStPToLamKchMTransform->Write();
  fSigStMToLamKchMTransform->Write();
  fSigSt0ToLamKchMTransform->Write();
  fLamAKSt0ToLamKchMTransform->Write();
  fSigAKSt0ToLamKchMTransform->Write();
  fXiCAKSt0ToLamKchMTransform->Write();
  fXi0AKSt0ToLamKchMTransform->Write();

  //ALamKchM
  fASigToALamKchMTransform->Write();
  fAXiCToALamKchMTransform->Write();
  fAXi0ToALamKchMTransform->Write();
  fAOmegaToALamKchMTransform->Write();
  fASigStMToALamKchMTransform->Write();
  fASigStPToALamKchMTransform->Write();
  fASigSt0ToALamKchMTransform->Write();
  fALamAKSt0ToALamKchMTransform->Write();
  fASigAKSt0ToALamKchMTransform->Write();
  fAXiCAKSt0ToALamKchMTransform->Write();
  fAXi0AKSt0ToALamKchMTransform->Write();

  //LamK0
  fSigToLamK0Transform->Write();
  fXiCToLamK0Transform->Write();
  fXi0ToLamK0Transform->Write();
  fOmegaToLamK0Transform->Write();
  fSigStPToLamK0Transform->Write();
  fSigStMToLamK0Transform->Write();
  fSigSt0ToLamK0Transform->Write();
  fLamKSt0ToLamK0Transform->Write();
  fSigKSt0ToLamK0Transform->Write();
  fXiCKSt0ToLamK0Transform->Write();
  fXi0KSt0ToLamK0Transform->Write();

  //ALamK0
  fASigToALamK0Transform->Write();
  fAXiCToALamK0Transform->Write();
  fAXi0ToALamK0Transform->Write();
  fAOmegaToALamK0Transform->Write();
  fASigStMToALamK0Transform->Write();
  fASigStPToALamK0Transform->Write();
  fASigSt0ToALamK0Transform->Write();
  fALamKSt0ToALamK0Transform->Write();
  fASigKSt0ToALamK0Transform->Write();
  fAXiCKSt0ToALamK0Transform->Write();
  fAXi0KSt0ToALamK0Transform->Write();

  //LamLam to check with Jai
  fSigToLamLamTransform->Write();

  tFile->Close();
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermEventsCollection::ExtractFromAllRootFiles
  int tBinV0Father=-1, tBinTrackFather=-1;
  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++) if(aV0FatherType==cAllLambdaFathers[i]) tBinV0Father=i;
  for(unsigned int i=0; i<cAllKchFathers.size(); i++) if(aTrackFatherType==cAllKchFathers[i]) tBinTrackFather=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBinV0Father==-1) cout << "FAILURE IMMINENT: aV0FatherType = " << aV0FatherType << endl;
    if(tBinTrackFather==-1) cout << "FAILURE IMMINENT: aTrackFatherType = " << aTrackFatherType << endl;
    assert(tBinV0Father>-1);
    assert(tBinTrackFather>-1);
  }
  aMatrix->Fill(tBinV0Father,tBinTrackFather);
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02FatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermEventsCollection::ExtractFromAllRootFiles
  int tBinV01Father=-1, tBinV02Father=-1;
  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++) if(aV01FatherType==cAllLambdaFathers[i]) tBinV01Father=i;
  for(unsigned int i=0; i<cAllK0ShortFathers.size(); i++) if(aV02FatherType==cAllK0ShortFathers[i]) tBinV02Father=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBinV01Father==-1) cout << "FAILURE IMMINENT: aV01FatherType = " << aV01FatherType << endl;
    if(tBinV02Father==-1) cout << "FAILURE IMMINENT: aV02FatherType = " << aV02FatherType << endl;
    assert(tBinV01Father>-1);
    assert(tBinV02Father>-1);
  }
  aMatrix->Fill(tBinV01Father,tBinV02Father);
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillProtonParents(TH1* aHist, int aFatherType)
{
  int tBin=-1;
  for(unsigned int i=0; i<cAllProtonFathers.size(); i++) if(aFatherType==cAllProtonFathers[i]) tBin=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBin==-1) cout << "FAILURE IMMINENT: aFatherType = " << aFatherType << endl;
    assert(tBin>-1);
  }
  aHist->Fill(tBin);
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillProtonRadii(TH2* a2dHist, ThermParticle &aParticle)
{
  int tBin=-1;
  int tFatherType = aParticle.GetFatherPID();
  for(unsigned int i=0; i<cAllProtonFathers.size(); i++) if(tFatherType==cAllProtonFathers[i]) tBin=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBin==-1) cout << "FAILURE IMMINENT: tFatherType = " << tFatherType << endl;
    assert(tBin>-1);
  }

  if(aParticle.IsPrimordial()) a2dHist->Fill(tBin, 0.);
  else a2dHist->Fill(tBin, GetLabDecayLength(GetParticleDecayLength(tFatherType), aParticle.GetFatherMass(), aParticle.GetFatherE()));
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillLambdaRadii(TH2* a2dHist, ThermV0Particle &aParticle)
{
  int tBin=-1;
  int tFatherType = aParticle.GetFatherPID();
  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++) if(tFatherType==cAllLambdaFathers[i]) tBin=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBin==-1) cout << "FAILURE IMMINENT: tFatherType = " << tFatherType << endl;
    assert(tBin>-1);
  }

  if(aParticle.IsPrimordial()) a2dHist->Fill(tBin, 0.);
  else a2dHist->Fill(tBin, GetLabDecayLength(GetParticleDecayLength(tFatherType), aParticle.GetFatherMass(), aParticle.GetFatherE()));
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::FillPrimaryAndOtherPairInfo(int aType1, int aType2, int aParentType1, int aParentType2, double aMaxPrimaryDecayLength)
{
  ParticlePDGType tParticleType1 = static_cast<ParticlePDGType>(aType1);
  ParticlePDGType tParticleType2 = static_cast<ParticlePDGType>(aType2);

  vector<vector<PidInfo> > *aPrimaryPairInfo;
  vector<vector<PidInfo> > *aOtherPairInfo;

  if(tParticleType1==kPDGLam && tParticleType2==kPDGKchP) {aPrimaryPairInfo=&fPrimaryPairInfoLamKchP; aOtherPairInfo=&fOtherPairInfoLamKchP;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGKchM) {aPrimaryPairInfo=&fPrimaryPairInfoALamKchM; aOtherPairInfo=&fOtherPairInfoALamKchM;}

  else if(tParticleType1==kPDGLam && tParticleType2==kPDGKchM) {aPrimaryPairInfo=&fPrimaryPairInfoLamKchM; aOtherPairInfo=&fOtherPairInfoLamKchM;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGKchP) {aPrimaryPairInfo=&fPrimaryPairInfoALamKchP; aOtherPairInfo=&fOtherPairInfoALamKchP;}

  else if(tParticleType1==kPDGLam && tParticleType2==kPDGK0) {aPrimaryPairInfo=&fPrimaryPairInfoLamK0; aOtherPairInfo=&fOtherPairInfoLamK0;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGK0) {aPrimaryPairInfo=&fPrimaryPairInfoALamK0; aOtherPairInfo=&fOtherPairInfoALamK0;}

  else assert(0);

  //--------------------------------------------------------------------

  bool bPairAlreadyIncluded = false;

  if(IncludeAsPrimary(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    for(unsigned int i=0; i<aPrimaryPairInfo->size(); i++)
    {
      if((*aPrimaryPairInfo)[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         (*aPrimaryPairInfo)[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) aPrimaryPairInfo->push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }

  //--------------------
  if(IncludeInOthers(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    bPairAlreadyIncluded = false;
    for(unsigned int i=0; i<aOtherPairInfo->size(); i++)
    {
      if((*aOtherPairInfo)[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         (*aOtherPairInfo)[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) aOtherPairInfo->push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::PrintPrimaryAndOtherPairInfo(int aType1, int aType2)
{
  ParticlePDGType tParticleType1 = static_cast<ParticlePDGType>(aType1);
  ParticlePDGType tParticleType2 = static_cast<ParticlePDGType>(aType2);

  vector<vector<PidInfo> > *aPrimaryPairInfo;
  vector<vector<PidInfo> > *aOtherPairInfo;

  if(tParticleType1==kPDGLam && tParticleType2==kPDGKchP) {aPrimaryPairInfo=&fPrimaryPairInfoLamKchP; aOtherPairInfo=&fOtherPairInfoLamKchP;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGKchM) {aPrimaryPairInfo=&fPrimaryPairInfoALamKchM; aOtherPairInfo=&fOtherPairInfoALamKchM;}

  else if(tParticleType1==kPDGLam && tParticleType2==kPDGKchM) {aPrimaryPairInfo=&fPrimaryPairInfoLamKchM; aOtherPairInfo=&fOtherPairInfoLamKchM;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGKchP) {aPrimaryPairInfo=&fPrimaryPairInfoALamKchP; aOtherPairInfo=&fOtherPairInfoALamKchP;}

  else if(tParticleType1==kPDGLam && tParticleType2==kPDGK0) {aPrimaryPairInfo=&fPrimaryPairInfoLamK0; aOtherPairInfo=&fOtherPairInfoLamK0;}
  else if(tParticleType1==kPDGALam && tParticleType2==kPDGK0) {aPrimaryPairInfo=&fPrimaryPairInfoALamK0; aOtherPairInfo=&fOtherPairInfoALamK0;}

  else assert(0);

  //--------------------------------------------------------------------

  cout << "----------------------------------------- " << GetPDGRootName(tParticleType1) << " and " << GetPDGRootName(tParticleType2) << " -----------------------------------------" << endl;

  cout << "---------- aPrimaryPairInfo ----------" << endl;
  cout << "\aPrimaryPairInfo.size() = " << aPrimaryPairInfo->size() << endl;
  for(unsigned int i=0; i<aPrimaryPairInfo->size(); i++)
  {
    cout << "PID1, PID2   = " << (*aPrimaryPairInfo)[i][0].pdgType << ", " << (*aPrimaryPairInfo)[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << (*aPrimaryPairInfo)[i][0].name << ", " << (*aPrimaryPairInfo)[i][1].name << endl << endl;
  }

  cout << "---------- aOtherPairInfo ----------" << endl;
  cout << "\aOtherPairInfo.size() = " << aOtherPairInfo->size() << endl;
  for(unsigned int i=0; i<aOtherPairInfo->size(); i++)
  {
    cout << "PID1, PID2   = " << (*aOtherPairInfo)[i][0].pdgType << ", " << (*aOtherPairInfo)[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << (*aOtherPairInfo)[i][0].name << ", " << (*aOtherPairInfo)[i][1].name << endl << endl;
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::PrintAllPrimaryAndOtherPairInfo()
{
  PrintPrimaryAndOtherPairInfo(kPDGLam, kPDGKchP);
  PrintPrimaryAndOtherPairInfo(kPDGALam, kPDGKchM);

  PrintPrimaryAndOtherPairInfo(kPDGLam, kPDGKchM);
  PrintPrimaryAndOtherPairInfo(kPDGALam, kPDGKchP);

  PrintPrimaryAndOtherPairInfo(kPDGLam, kPDGK0);
  PrintPrimaryAndOtherPairInfo(kPDGALam, kPDGK0);
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength, double tWeight)
{
  double tBin = -1.;
  if((aV0FatherType == kPDGLam || aV0FatherType == kPDGALam) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 0.;
  else if((aV0FatherType==kPDGSigma || aV0FatherType==kPDGASigma) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 1.;
  else if((aV0FatherType==kPDGXi0 || aV0FatherType==kPDGAXi0) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 2.;
  else if((aV0FatherType==kPDGXiC || aV0FatherType==kPDGAXiC) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 3.;
  else if((aV0FatherType==kPDGSigStP || aV0FatherType==kPDGASigStM) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 4.;
  else if((aV0FatherType==kPDGSigStM || aV0FatherType==kPDGASigStP) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 5.;
  else if((aV0FatherType==kPDGSigSt0 || aV0FatherType==kPDGASigSt0) && (aTrackFatherType == kPDGKchP || aTrackFatherType == kPDGKchM)) tBin = 6.;

  else if((aV0FatherType == kPDGLam || aV0FatherType == kPDGALam) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 7.;
  else if((aV0FatherType==kPDGSigma || aV0FatherType==kPDGASigma) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 8.;
  else if((aV0FatherType==kPDGXi0 || aV0FatherType==kPDGAXi0) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 9.;
  else if((aV0FatherType==kPDGXiC || aV0FatherType==kPDGAXiC) && (aTrackFatherType == kPDGKSt0 || aTrackFatherType == kPDGAKSt0)) tBin = 10.;
  else if(IncludeAsPrimary(aV0FatherType, aTrackFatherType, aMaxPrimaryDecayLength)) tBin=0.;
  else {assert(IncludeInOthers(aV0FatherType, aTrackFatherType, aMaxPrimaryDecayLength)); tBin = 11.;}

  if(tBin > -1)
  {
    tBin += 0.1;
    aHistogram->Fill(tBin, tWeight);
  }

}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength, double tWeight)
{
  double tBin = -1.;
  if((aV01FatherType == kPDGLam || aV01FatherType == kPDGALam) && (aV02FatherType == kPDGK0)) tBin = 0.;
  else if((aV01FatherType==kPDGSigma || aV01FatherType==kPDGASigma) && (aV02FatherType == kPDGK0)) tBin = 1.;
  else if((aV01FatherType==kPDGXi0 || aV01FatherType==kPDGAXi0) && (aV02FatherType == kPDGK0)) tBin = 2.;
  else if((aV01FatherType==kPDGXiC || aV01FatherType==kPDGAXiC) && (aV02FatherType == kPDGK0)) tBin = 3.;
  else if((aV01FatherType==kPDGSigStP || aV01FatherType==kPDGASigStM) && (aV02FatherType == kPDGK0)) tBin = 4.;
  else if((aV01FatherType==kPDGSigStM || aV01FatherType==kPDGASigStP) && (aV02FatherType == kPDGK0)) tBin = 5.;
  else if((aV01FatherType==kPDGSigSt0 || aV01FatherType==kPDGASigSt0) && (aV02FatherType == kPDGK0)) tBin = 6.;

  else if((aV01FatherType == kPDGLam || aV01FatherType == kPDGALam) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 7.;
  else if((aV01FatherType==kPDGSigma || aV01FatherType==kPDGASigma) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 8.;
  else if((aV01FatherType==kPDGXi0 || aV01FatherType==kPDGAXi0) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 9.;
  else if((aV01FatherType==kPDGXiC || aV01FatherType==kPDGAXiC) && (aV02FatherType == kPDGKSt0 || aV02FatherType == kPDGAKSt0)) tBin = 10.;
  else if(IncludeAsPrimary(aV01FatherType, aV02FatherType, aMaxPrimaryDecayLength)) tBin=0.;
  else {assert(IncludeInOthers(aV01FatherType,aV02FatherType, aMaxPrimaryDecayLength)); tBin = 11.;}

  if(tBin > -1)
  {
    tBin += 0.1;
    aHistogram->Fill(tBin, tWeight);
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildPairFractionHistogramsParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, TH1* aHistogram, TH2* aMatrix, double aMaxPrimaryDecayLength)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV0Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV0Type);
    aParticleCollection = fEventsCollection[iEv].GetParticleCollection(aParticleType);

    ThermV0Particle tV0;

    for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
    {
      tV0 = aV0Collection[iV0];
      int tV0FatherType = tV0.GetFatherPID();

      if(tV0.GoodV0())
      {
        ThermParticle tParticle;
        for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
        {
          tParticle = aParticleCollection[iPar];
          int tParticleFatherType = tParticle.GetFatherPID();

          MapAndFillPairFractionHistogramParticleV0(aHistogram, tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
          FillPrimaryAndOtherPairInfo(aV0Type, aParticleType, tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
          if(fBuildUniqueParents)
          {
            BuildUniqueParents(aParticleType, tParticleFatherType);
            BuildUniqueParents(aV0Type, tV0FatherType);
          }
          MapAndFillParentsMatrixParticleV0(aMatrix, tV0FatherType, tParticleFatherType);
        }
      }
    }

  }

}

//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildPairFractionHistogramsV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, TH1* aHistogram, TH2* aMatrix, double aMaxPrimaryDecayLength)
{
  vector<ThermV0Particle> aV01Collection, aV02Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV01Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV01Type);
    aV02Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV02Type);

    ThermV0Particle tV01;

    for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
    {
      tV01 = aV01Collection[iV01];
      int tV01FatherType = tV01.GetFatherPID();

      if(tV01.GoodV0())
      {
        ThermV0Particle tV02;
        for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
        {
          tV02 = aV02Collection[iV02];
          int tV02FatherType = tV02.GetFatherPID();

          if(tV02.GoodV0())
          {
            MapAndFillPairFractionHistogramV0V0(aHistogram, tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
            FillPrimaryAndOtherPairInfo(aV01Type, aV02Type, tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
            if(fBuildUniqueParents)
            {
              BuildUniqueParents(aV01Type, tV01FatherType);
              BuildUniqueParents(aV02Type, tV02FatherType);
            }
            MapAndFillParentsMatrixV0V0(aMatrix, tV01FatherType, tV02FatherType);
          }
        }
      }
    }

  }

}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildProtonParents()
{
  vector<ThermParticle> aProtCollection, aAProtCollection;
  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aProtCollection = fEventsCollection[iEv].GetParticleCollection(kPDGProt);
    aAProtCollection = fEventsCollection[iEv].GetParticleCollection(kPDGAntiProt);

    ThermParticle tParticle;
    int tParticleFatherType;
    for(unsigned int iPar=0; iPar<aProtCollection.size(); iPar++)
    {
      tParticle = aProtCollection[iPar];
      tParticleFatherType = tParticle.GetFatherPID();
      if(fBuildUniqueParents) BuildUniqueParents(kPDGProt, tParticleFatherType);
      MapAndFillProtonParents(fProtonParents, tParticleFatherType);
      MapAndFillProtonRadii(f2dProtonRadii, tParticle);
      if(tParticle.IsPrimordial()) fProtonRadii->Fill(0.);
      else fProtonRadii->Fill(GetLabDecayLength(GetParticleDecayLength(tParticleFatherType), tParticle.GetFatherMass(), tParticle.GetFatherE()));
    }
    for(unsigned int iPar=0; iPar<aAProtCollection.size(); iPar++)
    {
      tParticle = aAProtCollection[iPar];
      tParticleFatherType = tParticle.GetFatherPID();
      if(fBuildUniqueParents) BuildUniqueParents(kPDGAntiProt, tParticleFatherType);
      MapAndFillProtonParents(fAProtonParents, tParticleFatherType);
      MapAndFillProtonRadii(f2dAProtonRadii, tParticle);
      if(tParticle.IsPrimordial()) fAProtonRadii->Fill(0.);
      else fAProtonRadii->Fill(GetLabDecayLength(GetParticleDecayLength(tParticleFatherType), tParticle.GetFatherMass(), tParticle.GetFatherE()));
    }
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildLambdaParents()
{
  vector<ThermV0Particle> aLamCollection, aALamCollection;
  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aLamCollection = fEventsCollection[iEv].GetV0ParticleCollection(kPDGLam);
    aALamCollection = fEventsCollection[iEv].GetV0ParticleCollection(kPDGALam);

    ThermV0Particle tParticle;
    int tParticleFatherType;
    for(unsigned int iPar=0; iPar<aLamCollection.size(); iPar++)
    {
      tParticle = aLamCollection[iPar];
      tParticleFatherType = tParticle.GetFatherPID();
      MapAndFillLambdaRadii(f2dLamRadii, tParticle);
      if(tParticle.IsPrimordial()) fLamRadii->Fill(0.);
      else fLamRadii->Fill(GetLabDecayLength(GetParticleDecayLength(tParticleFatherType), tParticle.GetFatherMass(), tParticle.GetFatherE()));
    }
    for(unsigned int iPar=0; iPar<aALamCollection.size(); iPar++)
    {
      tParticle = aALamCollection[iPar];
      tParticleFatherType = tParticle.GetFatherPID();
      MapAndFillLambdaRadii(f2dALamRadii, tParticle);
      if(tParticle.IsPrimordial()) fALamRadii->Fill(0.);
      else fALamRadii->Fill(GetLabDecayLength(GetParticleDecayLength(tParticleFatherType), tParticle.GetFatherMass(), tParticle.GetFatherE()));
    }
  }
}



//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildAllPairFractionHistograms(double aMaxPrimaryDecayLength)
{
  BuildPairFractionHistogramsParticleV0(kPDGKchP, kPDGLam, fPairFractionsLamKchP, fParentsMatrixLamKchP, aMaxPrimaryDecayLength);
  BuildPairFractionHistogramsParticleV0(kPDGKchM, kPDGALam, fPairFractionsALamKchM, fParentsMatrixALamKchM, aMaxPrimaryDecayLength);

  BuildPairFractionHistogramsParticleV0(kPDGKchM, kPDGLam, fPairFractionsLamKchM, fParentsMatrixLamKchM, aMaxPrimaryDecayLength);
  BuildPairFractionHistogramsParticleV0(kPDGKchP, kPDGALam, fPairFractionsALamKchP, fParentsMatrixALamKchP, aMaxPrimaryDecayLength);

  BuildPairFractionHistogramsV0V0(kPDGLam, kPDGK0, fPairFractionsLamK0, fParentsMatrixLamK0, aMaxPrimaryDecayLength);
  BuildPairFractionHistogramsV0V0(kPDGALam, kPDGK0, fPairFractionsALamK0, fParentsMatrixALamK0, aMaxPrimaryDecayLength);

  BuildProtonParents();
  BuildLambdaParents();

}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildUniqueParents(int aParticleType, int aFatherType)
{
  ParticlePDGType tParticleType = static_cast<ParticlePDGType>(aParticleType);
  vector<int> *tCollectionToFill;
  switch(tParticleType) {
  case kPDGLam:
    tCollectionToFill = &fUniqueLamParents;
    break;

  case kPDGALam:
    tCollectionToFill = &fUniqueALamParents;
    break;

  case kPDGK0:
    tCollectionToFill = &fUniqueK0Parents;
    break;

  case kPDGKchP:
    tCollectionToFill = &fUniqueKchPParents;
    break;

  case kPDGKchM:
    tCollectionToFill = &fUniqueKchMParents;
    break;

  case kPDGProt:
    tCollectionToFill = &fUniqueProtParents;
    break;

  case kPDGAntiProt:
    tCollectionToFill = &fUniqueAProtParents;
    break;

  default:
    cout << "ERROR: ThermEventsCollection::BuildUniqueParents: aParticleType = " << aParticleType << " is not appropriate" << endl << endl;
    assert(0);
  }

  bool bParentAlreadyIncluded = false;
  for(unsigned int i=0; i<tCollectionToFill->size(); i++)
  {
    if((*tCollectionToFill)[i] == aFatherType) bParentAlreadyIncluded = true;
  }
  if(!bParentAlreadyIncluded) tCollectionToFill->push_back(aFatherType);
}

//________________________________________________________________________________________________________________
vector<int> ThermEventsCollection::UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2)
{
  vector<int> tReturnVector = aVec1;
  bool bAlreadyIncluded = false;
  for(unsigned int i=0; i<aVec2.size(); i++)
  {
    bAlreadyIncluded = false;
    for(unsigned int j=0; j<tReturnVector.size(); j++)
    {
      if(tReturnVector[j] == aVec2[i]) bAlreadyIncluded = true;
    }
    if(!bAlreadyIncluded) tReturnVector.push_back(aVec2[i]);
  }
  return tReturnVector;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::PrintUniqueParents()
{
  //-------Lam and ALam-----------

  std::sort(fUniqueLamParents.begin(), fUniqueLamParents.end());
  cout << "fUniqueLamParents.size() = " << fUniqueLamParents.size() << endl;
  for(unsigned int i=0; i<fUniqueLamParents.size()-1; i++) cout << fUniqueLamParents[i] << ", ";
  cout << fUniqueLamParents[fUniqueLamParents.size()-1] << endl;
  cout << endl << endl << endl;

  std::sort(fUniqueALamParents.begin(), fUniqueALamParents.end());
  cout << "fUniqueALamParents.size() = " << fUniqueALamParents.size() << endl;
  for(unsigned int i=0; i<fUniqueALamParents.size()-1; i++) cout << fUniqueALamParents[i] << ", ";
  cout << fUniqueALamParents[fUniqueALamParents.size()-1] << endl;
  cout << endl << endl << endl;

  fUniquecLamParents = UniqueCombineVectors(fUniqueLamParents, fUniqueALamParents);
  std::sort(fUniquecLamParents.begin(), fUniquecLamParents.end());
  cout << "fUniquecLamParents.size() = " << fUniquecLamParents.size() << endl;
  for(unsigned int i=0; i<fUniquecLamParents.size()-1; i++) cout << fUniquecLamParents[i] << ", ";
  cout << fUniquecLamParents[fUniquecLamParents.size()-1] << endl;
  cout << endl << endl << endl;

  //-----------K0-------------
  std::sort(fUniqueK0Parents.begin(), fUniqueK0Parents.end());
  cout << "fUniqueK0Parents.size() = " << fUniqueK0Parents.size() << endl;
  for(unsigned int i=0; i<fUniqueK0Parents.size()-1; i++) cout << fUniqueK0Parents[i] << ", ";
  cout << fUniqueK0Parents[fUniqueK0Parents.size()-1] << endl;
  cout << endl << endl << endl;

  //----------KchP and KchM-------
  std::sort(fUniqueKchPParents.begin(), fUniqueKchPParents.end());
  cout << "fUniqueKchPParents.size() = " << fUniqueKchPParents.size() << endl;
  for(unsigned int i=0; i<fUniqueKchPParents.size()-1; i++) cout << fUniqueKchPParents[i] << ", ";
  cout << fUniqueKchPParents[fUniqueKchPParents.size()-1] << endl;
  cout << endl << endl << endl;

  std::sort(fUniqueKchMParents.begin(), fUniqueKchMParents.end());
  cout << "fUniqueKchMParents.size() = " << fUniqueKchMParents.size() << endl;
  for(unsigned int i=0; i<fUniqueKchMParents.size()-1; i++) cout << fUniqueKchMParents[i] << ", ";
  cout << fUniqueKchMParents[fUniqueKchMParents.size()-1] << endl;
  cout << endl << endl << endl;

  fUniquecKchParents = UniqueCombineVectors(fUniqueKchPParents, fUniqueKchMParents);
  std::sort(fUniquecKchParents.begin(), fUniquecKchParents.end());
  cout << "fUniquecKchParents.size() = " << fUniquecKchParents.size() << endl;
  for(unsigned int i=0; i<fUniquecKchParents.size()-1; i++) cout << fUniquecKchParents[i] << ", ";
  cout << fUniquecKchParents[fUniquecKchParents.size()-1] << endl;
  cout << endl << endl << endl;

  //-----Proton and AntiProton-----
  std::sort(fUniqueProtParents.begin(), fUniqueProtParents.end());
  cout << "fUniqueProtParents.size() = " << fUniqueProtParents.size() << endl;
  for(unsigned int i=0; i<fUniqueProtParents.size()-1; i++) cout << fUniqueProtParents[i] << ", ";
  cout << fUniqueProtParents[fUniqueProtParents.size()-1] << endl;
  cout << endl << endl << endl;

  std::sort(fUniqueAProtParents.begin(), fUniqueAProtParents.end());
  cout << "fUniqueAProtParents.size() = " << fUniqueAProtParents.size() << endl;
  for(unsigned int i=0; i<fUniqueAProtParents.size()-1; i++) cout << fUniqueAProtParents[i] << ", ";
  cout << fUniqueAProtParents[fUniqueAProtParents.size()-1] << endl;
  cout << endl << endl << endl;

  fUniquecProtParents = UniqueCombineVectors(fUniqueProtParents, fUniqueAProtParents);
  std::sort(fUniquecProtParents.begin(), fUniquecProtParents.end());
  cout << "fUniquecProtParents.size() = " << fUniquecProtParents.size() << endl;
  for(unsigned int i=0; i<fUniquecProtParents.size()-1; i++) cout << fUniquecProtParents[i] << ", ";
  cout << fUniquecProtParents[fUniquecProtParents.size()-1] << endl;
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::SaveAllPairFractionHistograms(TString aSaveFileLocation)
{
  TFile *tFile = new TFile(aSaveFileLocation, "RECREATE");
  assert(tFile->IsOpen());

  fPairFractionsLamKchP->Write();
  fParentsMatrixLamKchP->Write();

  fPairFractionsALamKchM->Write();
  fParentsMatrixALamKchM->Write();

  fPairFractionsLamKchM->Write();
  fParentsMatrixLamKchM->Write();

  fPairFractionsALamKchP->Write();
  fParentsMatrixALamKchP->Write();

  fPairFractionsLamK0->Write();
  fParentsMatrixLamK0->Write();

  fPairFractionsALamK0->Write();
  fParentsMatrixALamK0->Write();

  fProtonParents->Write();
  fAProtonParents->Write();

  fProtonRadii->Write();
  fAProtonRadii->Write();

  f2dProtonRadii->Write();
  f2dAProtonRadii->Write();

  fLamRadii->Write();
  fALamRadii->Write();

  f2dLamRadii->Write();
  f2dALamRadii->Write();

  tFile->Close();
}

//________________________________________________________________________________________________________________
TCanvas* ThermEventsCollection::DrawAllPairFractionHistograms()
{
  TCanvas* tReturnCan = new TCanvas("tReturnCan","tReturnCan");
  tReturnCan->Divide(2,3);
  
  tReturnCan->cd(1);
  fPairFractionsLamKchP->Draw();

  tReturnCan->cd(2);
  fPairFractionsALamKchM->Draw();

  tReturnCan->cd(3);
  fPairFractionsLamKchM->Draw();

  tReturnCan->cd(4);
  fPairFractionsALamKchP->Draw();

  tReturnCan->cd(5);
  fPairFractionsLamK0->Draw();

  tReturnCan->cd(6);
  fPairFractionsALamK0->Draw();

  return tReturnCan;
}


//________________________________________________________________________________________________________________
double ThermEventsCollection::GetProperDecayLength(double aMeanDecayLength)
{
  double tLambda = 1.0/aMeanDecayLength;
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::exponential_distribution<double> tExpDistribution(tLambda);
  double tDecayLength = tExpDistribution(tGenerator);
//TODO
//cout << "aMeanDecayLength = " << aMeanDecayLength << endl;
//cout << "tDecayLength = " << tDecayLength << endl << endl;
  return tDecayLength;
}


//________________________________________________________________________________________________________________
double ThermEventsCollection::GetLabDecayLength(double aMeanDecayLength, double aMass, double aE)
{
  double tGamma = aE/aMass;
  return tGamma*GetProperDecayLength(aMeanDecayLength);
}

