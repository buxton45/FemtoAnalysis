/* ThermPairAnalysis.cxx */

#include "ThermPairAnalysis.h"

#ifdef __ROOT__
ClassImp(ThermPairAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermPairAnalysis::ThermPairAnalysis(AnalysisType aAnType) :
  fAnalysisType(aAnType),
  fPartType1(kPDGNull),
  fPartType2(kPDGNull),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fBuildUniqueParents(false),

  fUniqueParents1(0),
  fUniqueParents2(0),

  fBuildPairFractions(true),
  fBuildTransformMatrices(true),
  fBuildCorrelationFunctions(true),
  fBuildMixedEventNumerators(false),

  fTransformStorageMapping(0),
  fTransformInfo(),
  fTransformMatrices(nullptr),

  fPairFractions(nullptr),
  fParentsMatrix(nullptr),

  fPrimaryPairInfo(0),
  fOtherPairInfo(0),

  fPairSource3d(nullptr),
  fNum3d(nullptr),
  fDen3d(nullptr),

  fPairSourceFull(nullptr),
  fNumFull(nullptr),
  fDenFull(nullptr),
  fCfFull(nullptr),

  fPairSourcePrimaryOnly(nullptr),
  fNumPrimaryOnly(nullptr),
  fDenPrimaryOnly(nullptr),
  fCfPrimaryOnly(nullptr),

  fPairSourcePrimaryAndShortDecays(nullptr),
  fNumPrimaryAndShortDecays(nullptr),
  fDenPrimaryAndShortDecays(nullptr),
  fCfPrimaryAndShortDecays(nullptr),

  fPairSourceWithoutSigmaSt(nullptr),
  fNumWithoutSigmaSt(nullptr),
  fDenWithoutSigmaSt(nullptr),
  fCfWithoutSigmaSt(nullptr),

  fPairSourceSigmaStOnly(nullptr),
  fNumSigmaStOnly(nullptr),
  fDenSigmaStOnly(nullptr),
  fCfSigmaStOnly(nullptr),

  fPairSourceSecondaryOnly(nullptr),
  fNumSecondaryOnly(nullptr),
  fDenSecondaryOnly(nullptr),
  fCfSecondaryOnly(nullptr)

{
  SetPartTypes();
  InitiateTransformMatrices();

  TString tPairFractionsName = TString::Format("PairFractions%s", cAnalysisBaseTags[aAnType]);
  fPairFractions = new TH1D(tPairFractionsName, tPairFractionsName, 12, 0, 12);
  fPairFractions->Sumw2();

  TString tParentsMatrixName = TString::Format("ParentsMatrix%s", cAnalysisBaseTags[aAnType]);
  fParentsMatrix = new TH2D(tParentsMatrixName, tParentsMatrixName, 100, 0, 100, 135, 0, 135);
  fParentsMatrix->Sumw2();

  unsigned int tPidInfoSize = cPidInfo.size();
  fPairSource3d = new TH3D(TString::Format("PairSource3d%s", cAnalysisBaseTags[aAnType]), 
                           TString::Format("PairSource3d%s", cAnalysisBaseTags[aAnType]),
                           tPidInfoSize, 0, tPidInfoSize, 
                           tPidInfoSize, 0, tPidInfoSize,
                           100, 0., 1.);
  fNum3d = new TH3D(TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]), 
                           TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]),
                           tPidInfoSize, 0, tPidInfoSize, 
                           tPidInfoSize, 0, tPidInfoSize,
                           100, 0., 1.);
  fDen3d = new TH3D(TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]), 
                           TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]),
                           tPidInfoSize, 0, tPidInfoSize, 
                           tPidInfoSize, 0, tPidInfoSize,
                           100, 0., 1.);
  fPairSource3d->Sumw2();
  fNum3d->Sumw2();
  fDen3d->Sumw2();


  fPairSourceFull = new TH1D(TString::Format("PairSourceFull%s", cAnalysisBaseTags[aAnType]), 
                             TString::Format("PairSourceFull%s", cAnalysisBaseTags[aAnType]), 
                             1000, 0, 1000);
  fNumFull = new TH1D(TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]),
                      TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]), 
                      100, 0., 1.);
  fDenFull = new TH1D(TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]),
                      TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]), 
                      100, 0., 1.);
  fPairSourceFull->Sumw2();
  fNumFull->Sumw2();
  fDenFull->Sumw2();


  fPairSourcePrimaryOnly = new TH1D(TString::Format("PairSourcePrimaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("PairSourcePrimaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                    1000, 0, 1000);
  fNumPrimaryOnly = new TH1D(TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]),
                             TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]), 
                             100, 0., 1.);
  fDenPrimaryOnly = new TH1D(TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]),
                             TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]), 
                             100, 0., 1.);
  fPairSourcePrimaryOnly->Sumw2();
  fNumPrimaryOnly->Sumw2();
  fDenPrimaryOnly->Sumw2();

  fPairSourcePrimaryAndShortDecays = new TH1D(TString::Format("PairSourcePrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                              TString::Format("PairSourcePrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                              1000, 0, 1000);
  fNumPrimaryAndShortDecays = new TH1D(TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]),
                                       TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                       100, 0., 1.);
  fDenPrimaryAndShortDecays = new TH1D(TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]),
                                       TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                       100, 0., 1.);
  fPairSourcePrimaryAndShortDecays->Sumw2();
  fNumPrimaryAndShortDecays->Sumw2();
  fDenPrimaryAndShortDecays->Sumw2();


  fPairSourceWithoutSigmaSt = new TH1D(TString::Format("PairSourceWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                       TString::Format("PairSourceWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                       1000, 0, 1000);
  fNumWithoutSigmaSt = new TH1D(TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]),
                                TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                100, 0., 1.);
  fDenWithoutSigmaSt = new TH1D(TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]),
                                TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                100, 0., 1.);
  fPairSourceWithoutSigmaSt->Sumw2();
  fNumWithoutSigmaSt->Sumw2();
  fDenWithoutSigmaSt->Sumw2();

  fPairSourceSigmaStOnly = new TH1D(TString::Format("PairSourceSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                       TString::Format("PairSourceSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                       1000, 0, 1000);
  fNumSigmaStOnly = new TH1D(TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]),
                                TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                100, 0., 1.);
  fDenSigmaStOnly = new TH1D(TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]),
                                TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                100, 0., 1.);
  fPairSourceSigmaStOnly->Sumw2();
  fNumSigmaStOnly->Sumw2();
  fDenSigmaStOnly->Sumw2();

  fPairSourceSecondaryOnly = new TH1D(TString::Format("PairSourceSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                      TString::Format("PairSourceSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                      1000, 0, 1000);
  fNumSecondaryOnly = new TH1D(TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]),
                               TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                               100, 0., 1.);
  fDenSecondaryOnly = new TH1D(TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]),
                               TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                               100, 0., 1.);
  fPairSourceSecondaryOnly->Sumw2();
  fNumSecondaryOnly->Sumw2();
  fDenSecondaryOnly->Sumw2();

}



//________________________________________________________________________________________________________________
ThermPairAnalysis::~ThermPairAnalysis()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetPartTypes()
{
  switch(fAnalysisType) {
  //LamKchP-------------------------------
  case kLamKchP:
    fPartType1 = kPDGLam;
    fPartType2 = kPDGKchP;
    break;

  //ALamKchM-------------------------------
  case kALamKchM:
    fPartType1 = kPDGALam;
    fPartType2 = kPDGKchM;
    break;
  //-------------

  //LamKchM-------------------------------
  case kLamKchM:
    fPartType1 = kPDGLam;
    fPartType2 = kPDGKchM;
    break;
  //-------------

  //ALamKchP-------------------------------
  case kALamKchP:
    fPartType1 = kPDGALam;
    fPartType2 = kPDGKchP;
    break;
  //-------------

  //LamK0-------------------------------
  case kLamK0:
    fPartType1 = kPDGLam;
    fPartType2 = kPDGK0;
    break;
  //-------------

  //ALamK0-------------------------------
  case kALamK0:
    fPartType1 = kPDGALam;
    fPartType2 = kPDGK0;
    break;
  //-------------

  default:
    cout << "ERROR: ThermPairAnalysis::SetPartTypes:  fAnalysisType = " << fAnalysisType << " is not appropriate" << endl << endl;
    assert(0);
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::InitiateTransformMatrices()
{
  cout << "_______________________________________________________________________________________" << endl;
  cout << "InitiateTransformMatrices called for analysis of type " << cAnalysisBaseTags[fAnalysisType] << endl;

  switch(fAnalysisType) {
  case kLamKchP:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0KchP, kResXiCKchP, kResXi0KchP, kResOmegaKchP, kResSigStPKchP, 
                                                    kResSigStMKchP, kResSigSt0KchP, kResLamKSt0, kResSig0KSt0, kResXiCKSt0, kResXi0KSt0};

    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGLam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGSigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchP, kPDGXi0, kPDGKSt0);

    break;

  case kALamKchM:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0KchM, kResAXiCKchM, kResAXi0KchM, kResAOmegaKchM, kResASigStMKchM, 
                                                    kResASigStPKchM, kResASigSt0KchM, kResALamAKSt0, kResASig0AKSt0, kResAXiCAKSt0, kResAXi0AKSt0};

    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGALam, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGASigma, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXiC, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchM, kPDGAXi0, kPDGAKSt0);

    break;

  case kLamKchM:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0KchM, kResXiCKchM, kResXi0KchM, kResOmegaKchM, kResSigStPKchM, 
                                                    kResSigStMKchM, kResSigSt0KchM, kResLamAKSt0, kResSig0AKSt0, kResXiCAKSt0, kResXi0AKSt0};

    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGLam, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGSigma, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXiC, kPDGAKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGKchM, kPDGXi0, kPDGAKSt0);

    break;

  case kALamKchP:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0KchP, kResAXiCKchP, kResAXi0KchP, kResAOmegaKchP, kResASigStMKchP, 
                                                    kResASigStPKchP, kResASigSt0KchP, kResALamKSt0, kResASig0KSt0, kResAXiCKSt0, kResAXi0KSt0};

    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGALam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGASigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGKchP, kPDGAXi0, kPDGKSt0);

    break;

  case kLamK0:
    fTransformStorageMapping = vector<AnalysisType>{kResSig0K0, kResXiCK0, kResXi0K0, kResOmegaK0, kResSigStPK0, 
                                                    kResSigStMK0, kResSigSt0K0, kResLamKSt0ToLamK0, kResSig0KSt0ToLamK0, kResXiCKSt0ToLamK0, kResXi0KSt0ToLamK0};

    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGLam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGSigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGLam, kPDGK0, kPDGXi0, kPDGKSt0);

    break;

  case kALamK0:
    fTransformStorageMapping = vector<AnalysisType>{kResASig0K0, kResAXiCK0, kResAXi0K0, kResAOmegaK0, kResASigStMK0, 
                                                    kResASigStPK0, kResASigSt0K0, kResALamKSt0ToALamK0, kResASig0KSt0ToALamK0, kResAXiCKSt0ToALamK0, kResAXi0KSt0ToALamK0};

    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigma, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXiC, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXi0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAOmega, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigStM, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigStP, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigSt0, kPDGNull);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGALam, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGASigma, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXiC, kPDGKSt0);
    fTransformInfo.emplace_back(kPDGALam, kPDGK0, kPDGAXi0, kPDGKSt0);

    break;

  default:
    cout << "Error in ThermPairAnalysis::InitiateTransformMatrices, invalide fAnalysisType = " << fAnalysisType << " selected." << endl;
    assert(0);
  }

  fTransformMatrices = new TObjArray();
  fTransformMatrices->SetName(TString::Format("TransformMatrices_%s", cAnalysisBaseTags[fAnalysisType]));
  TString tTempTitle;
  for(unsigned int i=0; i<fTransformStorageMapping.size(); i++)
  {
    if(fTransformStorageMapping[i] == kResLamKSt0ToLamK0 || fTransformStorageMapping[i] == kResALamKSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResSig0KSt0ToLamK0 || fTransformStorageMapping[i] == kResASig0KSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResXi0KSt0ToLamK0 || fTransformStorageMapping[i] == kResAXi0KSt0ToALamK0 || 
       fTransformStorageMapping[i] == kResXiCKSt0ToLamK0 || fTransformStorageMapping[i] == kResAXiCKSt0ToALamK0)
    {
      tTempTitle = TString::Format("%sTransform", cAnalysisBaseTags[fTransformStorageMapping[i]]);
    }
    else tTempTitle = TString::Format("%sTo%sTransform", cAnalysisBaseTags[fTransformStorageMapping[i]], cAnalysisBaseTags[fAnalysisType]);
    fTransformMatrices->Add(new TH2D(tTempTitle, tTempTitle, fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax));

    cout << "Added transform with name " << tTempTitle << endl;
  }
  cout << "_______________________________________________________________________________________" << endl;
}



//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetFatherKStar(ThermParticle &aParticle1, ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2)
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
double ThermPairAnalysis::GetKStar(ThermParticle &aParticle1, ThermParticle &aParticle2)
{
  return GetFatherKStar(aParticle1, aParticle2, false, false);
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
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
void ThermPairAnalysis::FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
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
void ThermPairAnalysis::BuildTransformMatrixParticleV0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;


  aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  if(!fMixEvents)  //no mixing
  {
    aParticleCollection = aEvent.GetParticleCollection(tParticleType);
    FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
  }
  else
  {
    for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
    {
      aParticleCollection = aMixingEventsCollection[iMixEv].GetParticleCollection(tParticleType);
      FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aParticleFatherType,aV0FatherType,aMatrix);
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTransformMatrixV0V0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;


  aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  if(!fMixEvents)  //no mixing
  {
    aV02Collection = aEvent.GetV0ParticleCollection(tV02Type);
    FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
  }
  else
  {
    for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
    {
      aV02Collection = aMixingEventsCollection[iMixEv].GetV0ParticleCollection(tV02Type);
      FillTransformMatrixV0V0(aV01Collection,aV02Collection,aV01FatherType,aV02FatherType,aMatrix);
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildAllTransformMatrices(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection)
{
  bool bIsV0V0 = false;
  if(fTransformInfo[0].particleType2==kPDGK0 || fTransformInfo[0].particleType2==kPDGLam || fTransformInfo[0].particleType2==kPDGALam) bIsV0V0=true;
  for(unsigned int i=0; i<fTransformInfo.size(); i++)
  {
    if(bIsV0V0) BuildTransformMatrixV0V0(aEvent, aMixingEventsCollection, fTransformInfo[i].parentType1, fTransformInfo[i].parentType2, (TH2D*)fTransformMatrices->At(i));
    else BuildTransformMatrixParticleV0(aEvent, aMixingEventsCollection, fTransformInfo[i].parentType2, fTransformInfo[i].parentType1, (TH2D*)fTransformMatrices->At(i));
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SaveAllTransformMatrices(TFile *aFile)
{
  assert(aFile->IsOpen());
  for(int i=0; i<fTransformMatrices->GetEntries(); i++) ((TH2D*)fTransformMatrices->At(i))->Write();
}




//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermPairAnalysis::ExtractFromAllRootFiles
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
void ThermPairAnalysis::MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02FatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermPairAnalysis::ExtractFromAllRootFiles
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
void ThermPairAnalysis::FillUniqueParents(vector<int> &aUniqueParents, int aFatherType)
{
  bool bParentAlreadyIncluded = false;
  for(unsigned int i=0; i<aUniqueParents.size(); i++)
  {
    if(aUniqueParents[i] == aFatherType) bParentAlreadyIncluded = true;
  }
  if(!bParentAlreadyIncluded) aUniqueParents.push_back(aFatherType);
}

//________________________________________________________________________________________________________________
vector<int> ThermPairAnalysis::UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2)
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
void ThermPairAnalysis::PrintUniqueParents()
{
  std::sort(fUniqueParents1.begin(), fUniqueParents1.end());
  cout << "Unique parents of " << fTransformInfo[0].particleType1 << endl;
  cout << "\tN(parents) = " << fUniqueParents1.size() << endl;
  for(unsigned int i=0; i<fUniqueParents1.size()-1; i++) cout << fUniqueParents1[i] << ", ";
  cout << fUniqueParents1[fUniqueParents1.size()-1] << endl;
  cout << endl << endl << endl;

  std::sort(fUniqueParents2.begin(), fUniqueParents2.end());
  cout << "Unique parents of " << fTransformInfo[0].particleType2 << endl;
  cout << "\tN(parents) = " << fUniqueParents2.size() << endl;
  for(unsigned int i=0; i<fUniqueParents2.size()-1; i++) cout << fUniqueParents2[i] << ", ";
  cout << fUniqueParents2[fUniqueParents2.size()-1] << endl;
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillPrimaryAndOtherPairInfo(int aParentType1, int aParentType2, double aMaxPrimaryDecayLength)
{
  bool bPairAlreadyIncluded = false;

  if(IncludeAsPrimary(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    for(unsigned int i=0; i<fPrimaryPairInfo.size(); i++)
    {
      if(fPrimaryPairInfo[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         fPrimaryPairInfo[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) fPrimaryPairInfo.push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }

  //--------------------
  if(IncludeInOthers(aParentType1, aParentType2, aMaxPrimaryDecayLength))
  {
    bPairAlreadyIncluded = false;
    for(unsigned int i=0; i<fOtherPairInfo.size(); i++)
    {
      if(fOtherPairInfo[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         fOtherPairInfo[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) fOtherPairInfo.push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::PrintPrimaryAndOtherPairInfo()
{
  cout << "----------------------------------------- " << cAnalysisBaseTags[fAnalysisType] << " -----------------------------------------" << endl;

  cout << "---------- fPrimaryPairInfo ----------" << endl;
  cout << "\tfPrimaryPairInfo.size() = " << fPrimaryPairInfo.size() << endl;
  for(unsigned int i=0; i<fPrimaryPairInfo.size(); i++)
  {
    cout << "PID1, PID2   = " << fPrimaryPairInfo[i][0].pdgType << ", " << fPrimaryPairInfo[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << fPrimaryPairInfo[i][0].name << ", " << fPrimaryPairInfo[i][1].name << endl << endl;
  }

  cout << "---------- fOtherPairInfo ----------" << endl;
  cout << "\fOtherPairInfo.size() = " << fOtherPairInfo.size() << endl;
  for(unsigned int i=0; i<fOtherPairInfo.size(); i++)
  {
    cout << "PID1, PID2   = " << fOtherPairInfo[i][0].pdgType << ", " << fOtherPairInfo[i][1].pdgType << endl;
    cout << "Name1, Name2 = " << fOtherPairInfo[i][0].name << ", " << fOtherPairInfo[i][1].name << endl << endl;
  }
  
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength, double tWeight)
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
void ThermPairAnalysis::MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength, double tWeight)
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
void ThermPairAnalysis::BuildPairFractionHistogramsParticleV0(ThermEvent &aEvent, double aMaxPrimaryDecayLength)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  aParticleCollection = aEvent.GetParticleCollection(tParticleType);

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

        MapAndFillPairFractionHistogramParticleV0(fPairFractions, tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
        FillPrimaryAndOtherPairInfo(tV0FatherType, tParticleFatherType, aMaxPrimaryDecayLength);
        if(fBuildUniqueParents)
        {
          FillUniqueParents(fUniqueParents2, tParticleFatherType);
          FillUniqueParents(fUniqueParents1, tV0FatherType);
        }
        MapAndFillParentsMatrixParticleV0(fParentsMatrix, tV0FatherType, tParticleFatherType);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildPairFractionHistogramsV0V0(ThermEvent &aEvent, double aMaxPrimaryDecayLength)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  vector<ThermV0Particle> aV01Collection, aV02Collection;


  aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  aV02Collection =  aEvent.GetV0ParticleCollection(tV02Type);

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
          MapAndFillPairFractionHistogramV0V0(fPairFractions, tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
          FillPrimaryAndOtherPairInfo(tV01FatherType, tV02FatherType, aMaxPrimaryDecayLength);
          if(fBuildUniqueParents)
          {
            FillUniqueParents(fUniqueParents1, tV01FatherType);
            FillUniqueParents(fUniqueParents2, tV02FatherType);
          }
          MapAndFillParentsMatrixV0V0(fParentsMatrix, tV01FatherType, tV02FatherType);
        }
      }
    }
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::SavePairFractionsAndParentsMatrix(TFile *aFile)
{
  assert(aFile->IsOpen());
  
  fPairFractions->Write();
  fParentsMatrix->Write();
}



//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcKStar(ThermParticle &tPart1, ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  const double p_inv = (p1 + p2).Mag2(),
               q_inv = (p1 - p2).Mag2(),
           mass_diff = p1.Mag2() - p2.Mag2();

  const double tQ = ::pow(mass_diff, 2) / p_inv - q_inv;
  return ::sqrt(tQ) / 2.0;
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetKStar3Vec(ThermParticle &tPart1, ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  TLorentzVector x1 = tPart1.GetFourPosition();
  TLorentzVector x2 = tPart2.GetFourPosition();

  //---------------------------------

  TLorentzVector P = p1 + p2;

  // Calculate pair variables

  const double tPx = P.X(),
               tPy = P.Y(),
               tPz = P.Z();

  double tE1 = p1.E();
  double tE2 = p2.E();

  double tE  = tE1 + tE2;
  double tPt = tPx*tPx + tPy*tPy;
  double tMt = tE*tE - tPz*tPz;//mCVK;
  double tM  = (tMt - tPt > 0.0) ? sqrt(tMt - tPt) : 0.0;

  if (tMt == 0 || tE == 0 || tM == 0 || tPt == 0 ) {
    assert(0);
  }

  tMt = sqrt(tMt);
  tPt = sqrt(tPt);

  double pX = p1.X();
  double pY = p1.Y();
  double pZ = p1.Z();

  // Boost to LCMS
  double tBeta = tPz/tE;
  double tGamma = tE/tMt;
  double tKStarLong = tGamma * (pZ - tBeta * tE1);
  double tE1L = tGamma * (tE1  - tBeta * pZ);

  // Rotate in transverse plane
  double tKStarOut  = ( pX*tPx + pY*tPy)/tPt;
  double tKStarSide = (-pX*tPy + pY*tPx)/tPt;

  // Boost to pair cms
  tKStarOut = tMt/tM * (tKStarOut - tPt/tMt * tE1L);

  return TVector3(tKStarOut, tKStarSide, tKStarLong);
}


//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetRStar3Vec(ThermParticle &tPart1, ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  TLorentzVector x1 = tPart1.GetFourPosition();
  TLorentzVector x2 = tPart2.GetFourPosition();

  //---------------------------------

  TLorentzVector P = p1 + p2;

  // Calculate pair variables

  const double tPx = P.X(),
               tPy = P.Y(),
               tPz = P.Z();

  double tE1 = p1.E();
  double tE2 = p2.E();

  double tE  = tE1 + tE2;
  double tPt = tPx*tPx + tPy*tPy;
  double tMt = tE*tE - tPz*tPz;//mCVK;
  double tM  = (tMt - tPt > 0.0) ? sqrt(tMt - tPt) : 0.0;

  if (tMt == 0 || tE == 0 || tM == 0 || tPt == 0 ) {
    assert(0);
  }

  tMt = sqrt(tMt);
  tPt = sqrt(tPt);

  double pX = p1.X();
  double pY = p1.Y();
  double pZ = p1.Z();

  // Boost to LCMS
  double tBeta = tPz/tE;
  double tGamma = tE/tMt;
  double tKStarLong = tGamma * (pZ - tBeta * tE1);
  double tE1L = tGamma * (tE1  - tBeta * pZ);

  // Rotate in transverse plane
  double tKStarOut  = ( pX*tPx + pY*tPy)/tPt;
  double tKStarSide = (-pX*tPy + pY*tPx)/tPt;

  // Boost to pair cms
  tKStarOut = tMt/tM * (tKStarOut - tPt/tMt * tE1L);

  // separation distance
  TLorentzVector D = x1 - x2;

  double tDX = D.X();
  double tDY = D.Y();
  double tRLong = D.Z();
  double tDTime = D.T();

  double tROut = (tDX*tPx + tDY*tPy)/tPt;
  double tRSide = (-tDX*tPy + tDY*tPx)/tPt;

  double tRStarSide = tRSide;

  double tRStarLong = tGamma*(tRLong - tBeta* tDTime);
  double tDTimePairLCMS = tGamma*(tDTime - tBeta* tRLong);

  tBeta = tPt/tMt;
  tGamma = tMt/tM;

  double tRStarOut = tGamma*(tROut - tBeta* tDTimePairLCMS);

  return TVector3(tRStarOut, tRStarSide, tRStarLong);
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcRStar(ThermParticle &tPart1, ThermParticle &tPart2)
{
/*
  double tRStar = 0.;

  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  TLorentzVector x1 = tPart1.GetFourPosition();
  TLorentzVector x2 = tPart2.GetFourPosition();

  //---------------------------------

  TLorentzVector P = p1 + p2;

  // Calculate pair variables

  const double tPx = P.X(),
               tPy = P.Y(),
               tPz = P.Z();

  double tE1 = p1.E();
  double tE2 = p2.E();

  double tE  = tE1 + tE2;
  double tPt = tPx*tPx + tPy*tPy;
  double tMt = tE*tE - tPz*tPz;//mCVK;
  double tM  = (tMt - tPt > 0.0) ? sqrt(tMt - tPt) : 0.0;

  if (tMt == 0 || tE == 0 || tM == 0 || tPt == 0 ) {
    assert(0);
    return 0.0;
  }

  tMt = sqrt(tMt);
  tPt = sqrt(tPt);

  double pX = p1.X();
  double pY = p1.Y();
  double pZ = p1.Z();

  // Boost to LCMS
  double tBeta = tPz/tE;
  double tGamma = tE/tMt;
  double tKStarLong = tGamma * (pZ - tBeta * tE1);
  double tE1L = tGamma * (tE1  - tBeta * pZ);

  // Rotate in transverse plane
  double tKStarOut  = ( pX*tPx + pY*tPy)/tPt;
  double tKStarSide = (-pX*tPy + pY*tPx)/tPt;

  // Boost to pair cms
  tKStarOut = tMt/tM * (tKStarOut - tPt/tMt * tE1L);

  // separation distance
  TLorentzVector D = x1 - x2;

  double tDX = D.X();
  double tDY = D.Y();
  double tRLong = D.Z();
  double tDTime = D.T();

  double tROut = (tDX*tPx + tDY*tPy)/tPt;
  double tRSide = (-tDX*tPy + tDY*tPx)/tPt;

  double tRStarSide = tRSide;

  double tRStarLong = tGamma*(tRLong - tBeta* tDTime);
  double tDTimePairLCMS = tGamma*(tDTime - tBeta* tRLong);

  tBeta = tPt/tMt;
  tGamma = tMt/tM;

  double tRStarOut = tGamma*(tROut - tBeta* tDTimePairLCMS);

  tRStar = ::sqrt(tRStarOut*tRStarOut + tRStarSide*tRStarSide + tRStarLong*tRStarLong);
*/
  TVector3 tRStar3Vec = GetRStar3Vec(tPart1, tPart2);

/*
  double tKStar = ::sqrt(tKStarOut*tKStarOut + tKStarSide*tKStarSide + tKStarLong*tKStarLong);
  cout << "tRStar = " << tRStar << endl;
  cout << "tKStar1 = " << tKStar << endl;
  cout << "CalcKStar(p1,p2) = " << CalcKStar(p1,p2) << endl << endl; 
*/
  return tRStar3Vec.Mag();
}

//________________________________________________________________________________________________________________
complex<double> ThermPairAnalysis::GetStrongOnlyWaveFunction(TVector3 &aKStar3Vec, TVector3 &aRStar3Vec)
{
  if(aRStar3Vec.X()==0 && aRStar3Vec.Y()==0 && aRStar3Vec.Z()==0)  //TODO i.e. if pair originate from single resonance
  {
    double tRoot2 = sqrt(2.);
    double tRadius = 1.0;
    std::default_random_engine generator (std::clock());  //std::clock() is seed
    std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

    aRStar3Vec.SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator));
  }


  complex<double> ImI (0., 1.);
  complex<double> tF0 (0., 0.);
  double tD0 = 0.;
  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM) tF0 = complex<double>(-0.5,0.5);
  else if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) tF0 = complex<double>(0.25,0.5);
  else if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0) tF0 = complex<double>(-0.25,0.25);
  else assert(0);

  double tKdotR = aKStar3Vec.Dot(aRStar3Vec);
    tKdotR /= hbarc;
  double tKStarMag = aKStar3Vec.Mag();
    tKStarMag /= hbarc;
  double tRStarMag = aRStar3Vec.Mag();

  complex<double> tScattLenLastTerm (0., tKStarMag);
  complex<double> tScattAmp = pow((1./tF0) + 0.5*tD0*tKStarMag*tKStarMag - tScattLenLastTerm,-1);

  complex<double> tReturnWf = exp(ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  return tReturnWf;
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetStrongOnlyWaveFunctionSq(TVector3 aKStar3Vec, TVector3 aRStar3Vec)
{
  complex<double> tWf = GetStrongOnlyWaveFunction(aKStar3Vec, aRStar3Vec);
  double tWfSq = norm(tWf);
  return tWfSq;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumOrDenParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, bool aFillNumerator)
{
  TH3 *tCf3d;
  TH1 *tCfFull, *tCfPrimaryOnly, *tCfPrimaryAndShortDecays, *tCfWithoutSigmaSt, *tCfSigmaStOnly, *tCfSecondaryOnly;
  if(aFillNumerator)
  {
    tCf3d = fNum3d;
    tCfFull = fNumFull;
    tCfPrimaryOnly = fNumPrimaryOnly;
    tCfPrimaryAndShortDecays= fNumPrimaryAndShortDecays;
    tCfWithoutSigmaSt = fNumWithoutSigmaSt;
    tCfSigmaStOnly = fNumSigmaStOnly;
    tCfSecondaryOnly = fNumSecondaryOnly;
  }
  else
  {
    tCf3d = fDen3d;
    tCfFull = fDenFull;
    tCfPrimaryOnly = fDenPrimaryOnly;
    tCfPrimaryAndShortDecays = fDenPrimaryAndShortDecays;
    tCfWithoutSigmaSt = fDenWithoutSigmaSt;
    tCfSigmaStOnly = fDenSigmaStOnly;
    tCfSecondaryOnly = fDenSecondaryOnly;
  }


  ThermParticle tParticle;
  ThermV0Particle tV0;

  int tParentIndex1 = -1;
  int tParentIndex2 = -1;

  double tRStar = 0.;
  double tKStar = 0.;
  double tWeight = 1.;
  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GoodV0())
    {
      tParentIndex1 = GetParticleIndexInPidInfo(tV0.GetFatherPID());
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        tParentIndex2 = GetParticleIndexInPidInfo(tParticle.GetFatherPID());

        tRStar = CalcRStar(tV0, tParticle);
        tKStar = CalcKStar(tV0, tParticle);
        if(aFillNumerator) tWeight = GetStrongOnlyWaveFunctionSq(GetKStar3Vec(tV0, tParticle), GetRStar3Vec(tV0, tParticle));

        tCf3d->Fill(tParentIndex1, tParentIndex2, tKStar, tWeight);
        tCfFull->Fill(tKStar, tWeight);
        if(aFillNumerator)
        {
          fPairSource3d->Fill(tParentIndex1, tParentIndex2, tRStar);
          fPairSourceFull->Fill(tRStar);
        }

        if(tV0.IsPrimordial() && tParticle.IsPrimordial())
        {
          tCfPrimaryOnly->Fill(tKStar, tWeight);
          if(aFillNumerator) fPairSourcePrimaryOnly->Fill(tRStar);
        }

        if((tV0.IsPrimordial() && tParticle.IsPrimordial()) || IncludeAsPrimary(tV0.GetFatherPID(), tParticle.GetFatherPID(), 5.0))
        {
          tCfPrimaryAndShortDecays->Fill(tKStar, tWeight);
          if(aFillNumerator) fPairSourcePrimaryAndShortDecays->Fill(tRStar);
        }

        if(tV0.GetFatherPID() != kPDGSigStP && tV0.GetFatherPID() != kPDGASigStM && 
           tV0.GetFatherPID() != kPDGSigStM && tV0.GetFatherPID() != kPDGASigStP && 
           tV0.GetFatherPID() != kPDGSigSt0 && tV0.GetFatherPID() != kPDGASigSt0)
        {
          tCfWithoutSigmaSt->Fill(tKStar, tWeight);
          if(aFillNumerator) fPairSourceWithoutSigmaSt->Fill(tRStar);
        }
        else
        {
          tCfSigmaStOnly->Fill(tKStar, tWeight);
          if(aFillNumerator) fPairSourceSigmaStOnly->Fill(tRStar);
        }

        if(!tV0.IsPrimordial() && !tParticle.IsPrimordial())
        {
          tCfSecondaryOnly->Fill(tKStar, tWeight);
          if(aFillNumerator) fPairSourceSecondaryOnly->Fill(tRStar);
        }

      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumOrDenV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, bool aFillNumerator)
{
  TH3 *tCf3d;
  TH1 *tCfFull, *tCfPrimaryOnly, *tCfPrimaryAndShortDecays, *tCfWithoutSigmaSt, *tCfSigmaStOnly, *tCfSecondaryOnly;
  if(aFillNumerator)
  {
    tCf3d = fNum3d;
    tCfFull = fNumFull;
    tCfPrimaryOnly = fNumPrimaryOnly;
    tCfPrimaryAndShortDecays= fNumPrimaryAndShortDecays;
    tCfWithoutSigmaSt = fNumWithoutSigmaSt;
    tCfSigmaStOnly = fNumSigmaStOnly;
    tCfSecondaryOnly = fNumSecondaryOnly;
  }
  else
  {
    tCf3d = fDen3d;
    tCfFull = fDenFull;
    tCfPrimaryOnly = fDenPrimaryOnly;
    tCfPrimaryAndShortDecays = fDenPrimaryAndShortDecays;
    tCfWithoutSigmaSt = fDenWithoutSigmaSt;
    tCfSigmaStOnly = fDenSigmaStOnly;
    tCfSecondaryOnly = fDenSecondaryOnly;
  }


  ThermV0Particle tV01, tV02;

  int tParentIndex1 = -1;
  int tParentIndex2 = -1;

  double tRStar = 0.;
  double tKStar = 0.;
  double tWeight = 1.;
  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if(tV01.GoodV0())
    {
      tParentIndex1 = GetParticleIndexInPidInfo(tV01.GetFatherPID());
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if(tV02.GoodV0())
        {
          tParentIndex2 = GetParticleIndexInPidInfo(tV02.GetFatherPID());

          tRStar = CalcRStar(tV01, tV02);
          tKStar = CalcKStar(tV01, tV02);
          if(aFillNumerator) tWeight = GetStrongOnlyWaveFunctionSq(GetKStar3Vec(tV01, tV02), GetRStar3Vec(tV01, tV02));

          tCf3d->Fill(tParentIndex1, tParentIndex2, tKStar, tWeight);
          tCfFull->Fill(tKStar, tWeight);
          if(aFillNumerator)
          {
            fPairSource3d->Fill(tParentIndex1, tParentIndex2, tRStar);
            fPairSourceFull->Fill(tRStar);
          }

          if(tV01.IsPrimordial() && tV02.IsPrimordial())
          {
            tCfPrimaryOnly->Fill(tKStar, tWeight);
            if(aFillNumerator) fPairSourcePrimaryOnly->Fill(tRStar);
          }

          if((tV01.IsPrimordial() && tV02.IsPrimordial()) || IncludeAsPrimary(tV01.GetFatherPID(), tV02.GetFatherPID(), 5.0))
          {
            tCfPrimaryAndShortDecays->Fill(tKStar, tWeight);
            if(aFillNumerator) fPairSourcePrimaryAndShortDecays->Fill(tRStar);
          }

          if(tV01.GetFatherPID() != kPDGSigStP && tV01.GetFatherPID() != kPDGASigStM && 
             tV01.GetFatherPID() != kPDGSigStM && tV01.GetFatherPID() != kPDGASigStP && 
             tV01.GetFatherPID() != kPDGSigSt0 && tV01.GetFatherPID() != kPDGASigSt0)
          {
            tCfWithoutSigmaSt->Fill(tKStar, tWeight);
            if(aFillNumerator) fPairSourceWithoutSigmaSt->Fill(tRStar);
          }
          else
          {
            tCfSigmaStOnly->Fill(tKStar, tWeight);
            if(aFillNumerator) fPairSourceSigmaStOnly->Fill(tRStar);
          }

          if(!tV01.IsPrimordial() && !tV02.IsPrimordial())
          {
            tCfSecondaryOnly->Fill(tKStar, tWeight);
            if(aFillNumerator) fPairSourceSecondaryOnly->Fill(tRStar);
          }

        }
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumAndDenParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;

  int tParentIndex1 = -1;
  int tParentIndex2 = -1;

  double tRStar = 0.;
  double tKStar = 0.;
  double tWeight = 1.;
  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GoodV0())
    {
      tParentIndex1 = GetParticleIndexInPidInfo(tV0.GetFatherPID());
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        tParentIndex2 = GetParticleIndexInPidInfo(tParticle.GetFatherPID());

        tRStar = CalcRStar(tV0, tParticle);
        tKStar = CalcKStar(tV0, tParticle);
        tWeight = GetStrongOnlyWaveFunctionSq(GetKStar3Vec(tV0, tParticle), GetRStar3Vec(tV0, tParticle));

        fNum3d->Fill(tParentIndex1, tParentIndex2, tKStar, tWeight);
        fDen3d->Fill(tParentIndex1, tParentIndex2, tKStar);
        fPairSource3d->Fill(tParentIndex1, tParentIndex2, tRStar);

        fNumFull->Fill(tKStar, tWeight);
        fDenFull->Fill(tKStar);
        fPairSourceFull->Fill(tRStar);

        if(tV0.IsPrimordial() && tParticle.IsPrimordial())
        {
          fNumPrimaryOnly->Fill(tKStar, tWeight);
          fDenPrimaryOnly->Fill(tKStar);
          fPairSourcePrimaryOnly->Fill(tRStar);
        }

        if((tV0.IsPrimordial() && tParticle.IsPrimordial()) || IncludeAsPrimary(tV0.GetFatherPID(), tParticle.GetFatherPID(), 5.0))
        {
          fNumPrimaryAndShortDecays->Fill(tKStar, tWeight);
          fDenPrimaryAndShortDecays->Fill(tKStar);
          fPairSourcePrimaryAndShortDecays->Fill(tRStar);
        }

        if(tV0.GetFatherPID() != kPDGSigStP && tV0.GetFatherPID() != kPDGASigStM && 
           tV0.GetFatherPID() != kPDGSigStM && tV0.GetFatherPID() != kPDGASigStP && 
           tV0.GetFatherPID() != kPDGSigSt0 && tV0.GetFatherPID() != kPDGASigSt0)
        {
          fNumWithoutSigmaSt->Fill(tKStar, tWeight);
          fDenWithoutSigmaSt->Fill(tKStar);
          fPairSourceWithoutSigmaSt->Fill(tRStar);
        }
        else
        {
          fNumSigmaStOnly->Fill(tKStar, tWeight);
          fDenSigmaStOnly->Fill(tKStar);
          fPairSourceSigmaStOnly->Fill(tRStar);
        }

        if(!tV0.IsPrimordial() && !tParticle.IsPrimordial())
        {
          fNumSecondaryOnly->Fill(tKStar, tWeight);
          fDenSecondaryOnly->Fill(tKStar);
          fPairSourceSecondaryOnly->Fill(tRStar);
        }

      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumAndDenV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection)
{
  ThermV0Particle tV01, tV02;

  int tParentIndex1 = -1;
  int tParentIndex2 = -1;

  double tRStar = 0.;
  double tKStar = 0.;
  double tWeight = 1.;
  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if(tV01.GoodV0())
    {
      tParentIndex1 = GetParticleIndexInPidInfo(tV01.GetFatherPID());
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if(tV02.GoodV0())
        {
          tParentIndex2 = GetParticleIndexInPidInfo(tV02.GetFatherPID());

          tRStar = CalcRStar(tV01, tV02);
          tKStar = CalcKStar(tV01, tV02);
          tWeight = GetStrongOnlyWaveFunctionSq(GetKStar3Vec(tV01, tV02), GetRStar3Vec(tV01, tV02));

          fNum3d->Fill(tParentIndex1, tParentIndex2, tKStar, tWeight);
          fDen3d->Fill(tParentIndex1, tParentIndex2, tKStar);
          fPairSource3d->Fill(tParentIndex1, tParentIndex2, tRStar);

          fNumFull->Fill(tKStar, tWeight);
          fDenFull->Fill(tKStar);
          fPairSourceFull->Fill(tRStar);

          if(tV01.IsPrimordial() && tV02.IsPrimordial())
          {
            fNumPrimaryOnly->Fill(tKStar, tWeight);
            fDenPrimaryOnly->Fill(tKStar);
            fPairSourcePrimaryOnly->Fill(tRStar);
          }

          if((tV01.IsPrimordial() && tV02.IsPrimordial()) || IncludeAsPrimary(tV01.GetFatherPID(), tV02.GetFatherPID(), 5.0))
          {
            fNumPrimaryAndShortDecays->Fill(tKStar, tWeight);
            fDenPrimaryAndShortDecays->Fill(tKStar);
            fPairSourcePrimaryAndShortDecays->Fill(tRStar);
          }

          if(tV01.GetFatherPID() != kPDGSigStP && tV01.GetFatherPID() != kPDGASigStM && 
             tV01.GetFatherPID() != kPDGSigStM && tV01.GetFatherPID() != kPDGASigStP && 
             tV01.GetFatherPID() != kPDGSigSt0 && tV01.GetFatherPID() != kPDGASigSt0)
          {
            fNumWithoutSigmaSt->Fill(tKStar, tWeight);
            fDenWithoutSigmaSt->Fill(tKStar);
            fPairSourceWithoutSigmaSt->Fill(tRStar);
          }
          else
          {
            fNumSigmaStOnly->Fill(tKStar, tWeight);
            fDenSigmaStOnly->Fill(tKStar);
            fPairSourceSigmaStOnly->Fill(tRStar);
          }

          if(!tV01.IsPrimordial() && !tV02.IsPrimordial())
          {
            fNumSecondaryOnly->Fill(tKStar, tWeight);
            fDenSecondaryOnly->Fill(tKStar);
            fPairSourceSecondaryOnly->Fill(tRStar);
          }

        }
      }
    }
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildCorrelationFunctionsParticleV0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;
  
  aV0Collection =  aEvent.GetV0ParticleCollection(fPartType1);
  if(!fBuildMixedEventNumerators) //-- No mixing, i.e. real pairs
  {
    aParticleCollection = aEvent.GetParticleCollection(fPartType2);
    FillCorrelationFunctionsNumOrDenParticleV0(aParticleCollection, aV0Collection, true);
  }

  //-- Mixed events
  for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
  {
    aParticleCollection = aMixingEventsCollection[iMixEv].GetParticleCollection(fPartType2);
    if(!fBuildMixedEventNumerators) FillCorrelationFunctionsNumOrDenParticleV0(aParticleCollection, aV0Collection, false);
    else FillCorrelationFunctionsNumAndDenParticleV0(aParticleCollection, aV0Collection);
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildCorrelationFunctionsV0V0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection)
{
  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;

 
  aV01Collection =  aEvent.GetV0ParticleCollection(fPartType1);
  if(!fBuildMixedEventNumerators) //-- No mixing, i.e. real pairs
  {
    aV02Collection = aEvent.GetV0ParticleCollection(fPartType2);
    FillCorrelationFunctionsNumOrDenV0V0(aV01Collection, aV02Collection, true);
  }

  //-- Mixed events
  for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
  {
    aV02Collection = aMixingEventsCollection[iMixEv].GetV0ParticleCollection(fPartType2);
    if(!fBuildMixedEventNumerators) FillCorrelationFunctionsNumOrDenV0V0(aV01Collection, aV02Collection, false);
    else FillCorrelationFunctionsNumAndDenV0V0(aV01Collection, aV02Collection);
  }
}


//________________________________________________________________________________________________________________
TH1* ThermPairAnalysis::BuildFinalCf(TH1* aNum, TH1* aDen, TString aName)
{
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  int tMinNormBin = aNum->FindBin(tMinNorm);
  int tMaxNormBin = aNum->FindBin(tMaxNorm);
  double tNumScale = aNum->Integral(tMinNormBin,tMaxNormBin);

  tMinNormBin = aDen->FindBin(tMinNorm);
  tMaxNormBin = aDen->FindBin(tMaxNorm);
  double tDenScale = aDen->Integral(tMinNormBin,tMaxNormBin);

  TH1* tReturnCf = (TH1*)aNum->Clone(aName);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!tReturnCf->GetSumw2N()) tReturnCf->Sumw2();

  tReturnCf->Divide(aDen);
  tReturnCf->Scale(tDenScale/tNumScale);
  tReturnCf->SetTitle(aName);

  if(!tReturnCf->GetSumw2N()) {tReturnCf->Sumw2();}

  return tReturnCf;
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::SaveAllCorrelationFunctions(TFile *aFile)
{
  assert(aFile->IsOpen());

  fPairSource3d->Write();
  fNum3d->Write();
  fDen3d->Write();

  fPairSourceFull->Write();
  fNumFull->Write();
  fDenFull->Write();
  fCfFull = BuildFinalCf(fNumFull, fDenFull, TString::Format("CfFull%s", cAnalysisBaseTags[fAnalysisType]));
  fCfFull->Write();

  fPairSourcePrimaryOnly->Write();
  fNumPrimaryOnly->Write();
  fDenPrimaryOnly->Write();
  fCfPrimaryOnly = BuildFinalCf(fNumPrimaryOnly, fDenPrimaryOnly, TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]));
  fCfPrimaryOnly->Write();

  fPairSourcePrimaryAndShortDecays->Write();
  fNumPrimaryAndShortDecays->Write();
  fDenPrimaryAndShortDecays->Write();
  fCfPrimaryAndShortDecays = BuildFinalCf(fNumPrimaryAndShortDecays, fDenPrimaryAndShortDecays, TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]));
  fCfPrimaryAndShortDecays->Write();

  fPairSourceWithoutSigmaSt->Write();
  fNumWithoutSigmaSt->Write();
  fDenWithoutSigmaSt->Write();
  fCfWithoutSigmaSt = BuildFinalCf(fNumWithoutSigmaSt, fDenWithoutSigmaSt, TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]));
  fCfWithoutSigmaSt->Write();

  fPairSourceSigmaStOnly->Write();
  fNumSigmaStOnly->Write();
  fDenSigmaStOnly->Write();
  fCfSigmaStOnly = BuildFinalCf(fNumSigmaStOnly, fDenSigmaStOnly, TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]));
  fCfSigmaStOnly->Write();

  fPairSourceSecondaryOnly->Write();
  fNumSecondaryOnly->Write();
  fDenSecondaryOnly->Write();
  fCfSecondaryOnly = BuildFinalCf(fNumSecondaryOnly, fDenSecondaryOnly, TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]));
  fCfSecondaryOnly->Write();
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::ProcessEvent(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, double aMaxPrimaryDecayLength)
{
  if(fBuildTransformMatrices) BuildAllTransformMatrices(aEvent, aMixingEventsCollection);

  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
  {
    if(fBuildPairFractions) BuildPairFractionHistogramsV0V0(aEvent, aMaxPrimaryDecayLength);
    if(fBuildCorrelationFunctions)BuildCorrelationFunctionsV0V0(aEvent, aMixingEventsCollection);
  }
  else
  {
    if(fBuildPairFractions) BuildPairFractionHistogramsParticleV0(aEvent, aMaxPrimaryDecayLength);
    if(fBuildCorrelationFunctions)BuildCorrelationFunctionsParticleV0(aEvent, aMixingEventsCollection);
  }
}





