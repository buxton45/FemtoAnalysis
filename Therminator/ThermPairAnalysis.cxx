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
  fGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
  fAnalysisType(aAnType),
  fPartType1(kPDGNull),
  fPartType2(kPDGNull),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fMaxPrimaryDecayLength(-1.),

  fBuildUniqueParents(false),

  fUniqueParents1(0),
  fUniqueParents2(0),

  fBuildPairFractions(true),
  fBuildTransformMatrices(true),
  fBuildCorrelationFunctions(true),
  fBuild3dHists(false),
  fBuildMixedEventNumerators(false),

  fTransformStorageMapping(0),
  fTransformInfo(),
  fTransformMatrices(nullptr),

  fPairFractions(nullptr),
  fParentsMatrix(nullptr),

  fPrimaryPairInfo(0),
  fOtherPairInfo(0),

  fUnitWeightCfNums(false),
  fWeightCfsWithParentInteraction(false),
  fOnlyWeightLongDecayParents(false),

  fDrawRStarFromGaussian(false),

  fBuildPairSourcewmTInfo(false),

  fPairSource3d(nullptr),
  fNum3d(nullptr),
  fDen3d(nullptr),

  fPairSourceFull(nullptr),
  fNumFull(nullptr),
  fDenFull(nullptr),
  fCfFull(nullptr),
  fNumFull_RotatePar2(nullptr),

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
  fCfSecondaryOnly(nullptr),

  fPairKStarVsmT(nullptr),
  fPairmT3d(nullptr),

  fPairSource3d_OSLinPRF(nullptr),
  fPairSource3d_OSLinPRFPrimaryOnly(nullptr),

  fPairDeltaT_inPRF(nullptr),
  fPairDeltaT_inPRFPrimaryOnly(nullptr),

  fTrueRosl(nullptr),
  fSimpleRosl(nullptr),

  fPairSource3d_mT1vmT2vRinv(nullptr),
  fPairSource2d_PairmTvRinv(nullptr),
  fPairSource2d_mT1vRinv(nullptr),
  fPairSource2d_mT2vRinv(nullptr),
  fPairSource2d_mT1vR1PRF(nullptr),
  fPairSource2d_mT2vR2PRF(nullptr),

  fBuildCfYlm(false),
  fCfYlm(nullptr)

{
  SetPartTypes();
}



//________________________________________________________________________________________________________________
ThermPairAnalysis::~ThermPairAnalysis()
{
  if(fTransformMatrices) fTransformMatrices->Delete();

  delete fPairFractions;
  delete fParentsMatrix;

  delete fPairSource3d;
  delete fNum3d;
  delete fDen3d;

  delete fPairSourceFull;
  delete fNumFull;
  delete fDenFull;
  delete fCfFull;
  delete fNumFull_RotatePar2;

  delete fPairSourcePrimaryOnly;
  delete fNumPrimaryOnly;
  delete fDenPrimaryOnly;
  delete fCfPrimaryOnly;

  delete fPairSourcePrimaryAndShortDecays;
  delete fNumPrimaryAndShortDecays;
  delete fDenPrimaryAndShortDecays;
  delete fCfPrimaryAndShortDecays;

  delete fPairSourceWithoutSigmaSt;
  delete fNumWithoutSigmaSt;
  delete fDenWithoutSigmaSt;
  delete fCfWithoutSigmaSt;

  delete fPairSourceSigmaStOnly;
  delete fNumSigmaStOnly;
  delete fDenSigmaStOnly;
  delete fCfSigmaStOnly;

  delete fPairSourceSecondaryOnly;
  delete fNumSecondaryOnly;
  delete fDenSecondaryOnly;
  delete fCfSecondaryOnly;

  delete fPairKStarVsmT;
  delete fPairmT3d;

  delete fPairSource3d_OSLinPRF;
  delete fPairSource3d_OSLinPRFPrimaryOnly;

  delete fPairDeltaT_inPRF;
  delete fPairDeltaT_inPRFPrimaryOnly;

  delete fTrueRosl;
  delete fSimpleRosl;

  delete fPairSource3d_mT1vmT2vRinv;
  delete fPairSource2d_PairmTvRinv;
  delete fPairSource2d_mT1vRinv;
  delete fPairSource2d_mT2vRinv;
  delete fPairSource2d_mT1vR1PRF;
  delete fPairSource2d_mT2vR2PRF;

  delete fCfYlm;
}


//________________________________________________________________________________________________________________
vector<ParticlePDGType> ThermPairAnalysis::GetPartTypes(AnalysisType aAnalysisType)
{
  vector<ParticlePDGType> tReturnVec(2);
  ParticlePDGType tPartType1, tPartType2;

  switch(aAnalysisType) {
  //LamKchP-------------------------------
  case kLamKchP:
    tPartType1 = kPDGLam;
    tPartType2 = kPDGKchP;
    break;

  //ALamKchM-------------------------------
  case kALamKchM:
    tPartType1 = kPDGALam;
    tPartType2 = kPDGKchM;
    break;
  //-------------

  //LamKchM-------------------------------
  case kLamKchM:
    tPartType1 = kPDGLam;
    tPartType2 = kPDGKchM;
    break;
  //-------------

  //ALamKchP-------------------------------
  case kALamKchP:
    tPartType1 = kPDGALam;
    tPartType2 = kPDGKchP;
    break;
  //-------------

  //LamK0-------------------------------
  case kLamK0:
    tPartType1 = kPDGLam;
    tPartType2 = kPDGK0;
    break;
  //-------------

  //ALamK0-------------------------------
  case kALamK0:
    tPartType1 = kPDGALam;
    tPartType2 = kPDGK0;
    break;
  //-------------

  //LamLam-------------------------------
  case kLamLam:
    tPartType1 = kPDGLam;
    tPartType2 = kPDGLam;
    break;
  //-------------

  //K0K0-------------------------------
  case kK0K0:
    tPartType1 = kPDGK0;
    tPartType2 = kPDGK0;
    break;
  //-------------

  //KchPKchP-------------------------------
  case kKchPKchP:
    tPartType1 = kPDGKchP;
    tPartType2 = kPDGKchP;
    break;
  //-------------

  default:
    cout << "ERROR: ThermPairAnalysis::GetPartTypes:  aAnalysisType = " << aAnalysisType << " is not appropriate" << endl << endl;
    assert(0);
  }

  tReturnVec[0] = tPartType1;
  tReturnVec[1] = tPartType2;
  return tReturnVec;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetPartTypes()
{
  vector<ParticlePDGType> tTempVec = GetPartTypes(fAnalysisType);
  fPartType1 = tTempVec[0];
  fPartType2 = tTempVec[1];
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
void ThermPairAnalysis::SetBuildTransformMatrices(bool aBuild)
{
  fBuildTransformMatrices = aBuild;
  if(fBuildTransformMatrices) InitiateTransformMatrices();
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::InitiateCorrelations()
{
  cout << "_______________________________________________________________________________________" << endl;
  cout << "InitiateCorrelations called for analysis of type " << cAnalysisBaseTags[fAnalysisType] << endl;

  if(fBuild3dHists)
  {
    cout << "Warning: fBuild3dHists = true!" << endl << "\t This will use a huge amount of memory" << endl;
    cout << "\t Are you sure you want to continue? (0=No 1=Yes)" << endl;
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }

  int tNbinsKStar = 300;
  double tKStarMin = 0.;
  double tKStarMax = 3.;

  int tNbinsRStar3d = 200;
  double tRStarMin3d = 0.;
  double tRStarMax3d = 200.;

  int tNbinsRStar1d = 1000;
  double tRStarMin1d = 0.;
  double tRStarMax1d = 1000.;

  unsigned int tPidInfoSize = cPidInfo.size();
  if(fBuild3dHists)
  {
    fPairSource3d = new TH3D(TString::Format("PairSource3d%s", cAnalysisBaseTags[fAnalysisType]), 
                             TString::Format("PairSource3d%s", cAnalysisBaseTags[fAnalysisType]),
                             tPidInfoSize, 0, tPidInfoSize, 
                             tPidInfoSize, 0, tPidInfoSize,
                             tNbinsRStar3d, tRStarMin3d, tRStarMax3d);
    fNum3d = new TH3D(TString::Format("Num3d%s", cAnalysisBaseTags[fAnalysisType]), 
                             TString::Format("Num3d%s", cAnalysisBaseTags[fAnalysisType]),
                             tPidInfoSize, 0, tPidInfoSize, 
                             tPidInfoSize, 0, tPidInfoSize,
                             tNbinsKStar, tKStarMin, tKStarMax);
    fDen3d = new TH3D(TString::Format("Den3d%s", cAnalysisBaseTags[fAnalysisType]), 
                             TString::Format("Den3d%s", cAnalysisBaseTags[fAnalysisType]),
                             tPidInfoSize, 0, tPidInfoSize, 
                             tPidInfoSize, 0, tPidInfoSize,
                             tNbinsKStar, tKStarMin, tKStarMax);
    fPairSource3d->Sumw2();
    fNum3d->Sumw2();
    fDen3d->Sumw2();
  }


  fPairSourceFull = new TH1D(TString::Format("PairSourceFull%s", cAnalysisBaseTags[fAnalysisType]), 
                             TString::Format("PairSourceFull%s", cAnalysisBaseTags[fAnalysisType]), 
                             tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumFull = new TH1D(TString::Format("NumFull%s", cAnalysisBaseTags[fAnalysisType]),
                      TString::Format("NumFull%s", cAnalysisBaseTags[fAnalysisType]), 
                      tNbinsKStar, tKStarMin, tKStarMax);
  fDenFull = new TH1D(TString::Format("DenFull%s", cAnalysisBaseTags[fAnalysisType]),
                      TString::Format("DenFull%s", cAnalysisBaseTags[fAnalysisType]), 
                      tNbinsKStar, tKStarMin, tKStarMax);
  fNumFull_RotatePar2 = new TH1D(TString::Format("NumFull_RotatePar2%s", cAnalysisBaseTags[fAnalysisType]),
                                 TString::Format("NumFull_RotatePar2%s", cAnalysisBaseTags[fAnalysisType]), 
                                 tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourceFull->Sumw2();
  fNumFull->Sumw2();
  fDenFull->Sumw2();
  fNumFull_RotatePar2->Sumw2();


  fPairSourcePrimaryOnly = new TH1D(TString::Format("PairSourcePrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                    TString::Format("PairSourcePrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                    tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumPrimaryOnly = new TH1D(TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                             TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                             tNbinsKStar, tKStarMin, tKStarMax);
  fDenPrimaryOnly = new TH1D(TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                             TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                             tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourcePrimaryOnly->Sumw2();
  fNumPrimaryOnly->Sumw2();
  fDenPrimaryOnly->Sumw2();

  fPairSourcePrimaryAndShortDecays = new TH1D(TString::Format("PairSourcePrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]), 
                                              TString::Format("PairSourcePrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]), 
                                              tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumPrimaryAndShortDecays = new TH1D(TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]), 
                                       tNbinsKStar, tKStarMin, tKStarMax);
  fDenPrimaryAndShortDecays = new TH1D(TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]), 
                                       tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourcePrimaryAndShortDecays->Sumw2();
  fNumPrimaryAndShortDecays->Sumw2();
  fDenPrimaryAndShortDecays->Sumw2();


  fPairSourceWithoutSigmaSt = new TH1D(TString::Format("PairSourceWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]), 
                                       TString::Format("PairSourceWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]), 
                                       tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumWithoutSigmaSt = new TH1D(TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]),
                                TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]), 
                                tNbinsKStar, tKStarMin, tKStarMax);
  fDenWithoutSigmaSt = new TH1D(TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]),
                                TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]), 
                                tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourceWithoutSigmaSt->Sumw2();
  fNumWithoutSigmaSt->Sumw2();
  fDenWithoutSigmaSt->Sumw2();

  fPairSourceSigmaStOnly = new TH1D(TString::Format("PairSourceSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                       TString::Format("PairSourceSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                       tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumSigmaStOnly = new TH1D(TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]),
                                TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                tNbinsKStar, tKStarMin, tKStarMax);
  fDenSigmaStOnly = new TH1D(TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]),
                                TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourceSigmaStOnly->Sumw2();
  fNumSigmaStOnly->Sumw2();
  fDenSigmaStOnly->Sumw2();

  fPairSourceSecondaryOnly = new TH1D(TString::Format("PairSourceSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                      TString::Format("PairSourceSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                                      tNbinsRStar1d, tRStarMin1d, tRStarMax1d);
  fNumSecondaryOnly = new TH1D(TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                               tNbinsKStar, tKStarMin, tKStarMax);
  fDenSecondaryOnly = new TH1D(TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]), 
                               tNbinsKStar, tKStarMin, tKStarMax);
  fPairSourceSecondaryOnly->Sumw2();
  fNumSecondaryOnly->Sumw2();
  fDenSecondaryOnly->Sumw2();


  fPairKStarVsmT = new TH2D(TString::Format("PairKStarVsmT%s", cAnalysisBaseTags[fAnalysisType]),
                            TString::Format("PairKStarVsmT%s", cAnalysisBaseTags[fAnalysisType]),
                            100, 0., 1.,
                            250, 0., 2.5);
  fPairKStarVsmT->Sumw2();

  if(fBuild3dHists)
  {
    fPairmT3d = new TH3D(TString::Format("PairmT3d%s", cAnalysisBaseTags[fAnalysisType]),
                         TString::Format("PairmT3d%s", cAnalysisBaseTags[fAnalysisType]),
                         tPidInfoSize, 0, tPidInfoSize, 
                         tPidInfoSize, 0, tPidInfoSize,
                         250, 0., 2.5);
    fPairmT3d->Sumw2();
  }


  //--------------------------------------
  fPairSource3d_OSLinPRF = new TH3D(TString::Format("PairSource3d_osl%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("PairSource3d_osl%s", cAnalysisBaseTags[fAnalysisType]),
                               200, -50., 50., 
                               200, -50., 50., 
                               200, -50., 50.);

  fPairSource3d_OSLinPRFPrimaryOnly = new TH3D(TString::Format("PairSource3d_oslPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                                          TString::Format("PairSource3d_oslPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                                          200, -50., 50., 
                                          200, -50., 50., 
                                          200, -50., 50.); 
  fPairSource3d_OSLinPRF->Sumw2();
  fPairSource3d_OSLinPRFPrimaryOnly->Sumw2();

  //--------------------------------------
  fPairDeltaT_inPRF = new TH1D(TString::Format("PairDeltaT_inPRF%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("PairDeltaT_inPRF%s", cAnalysisBaseTags[fAnalysisType]),
                               200, -100., 100.);

  fPairDeltaT_inPRFPrimaryOnly = new TH1D(TString::Format("PairDeltaT_inPRFPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("PairDeltaT_inPRFPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]),
                               200, -100., 100.);

  fPairDeltaT_inPRF->Sumw2();
  fPairDeltaT_inPRFPrimaryOnly->Sumw2();


  //--------------------------------------
  fTrueRosl = new TH3D(TString::Format("TrueRosl%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("TrueRosl%s", cAnalysisBaseTags[fAnalysisType]),
                               200, 0., 200., 
                               200, 0., 200., 
                               200, 0., 200.); 

  fTrueRosl->Sumw2();


  fSimpleRosl = new TH3D(TString::Format("SimpleRosl%s", cAnalysisBaseTags[fAnalysisType]),
                               TString::Format("SimpleRosl%s", cAnalysisBaseTags[fAnalysisType]),
                               200, -50., 50., 
                               200, -50., 50., 
                               200, -50., 50.); 

  fSimpleRosl->Sumw2();

  //--------------------------------------
  if(fBuildPairSourcewmTInfo)
  {
    fPairSource3d_mT1vmT2vRinv = new TH3D(TString::Format("PairSource3d_mT1vmT2vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                          TString::Format("PairSource3d_mT1vmT2vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                          400, 0., 10., 
                                          400, 0., 10., 
                                          100, 0., 50.);
    fPairSource2d_PairmTvRinv = new TH2D(TString::Format("PairSource2d_PairmTvRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                         TString::Format("PairSource2d_PairmTvRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                         400, 0., 10., 
                                         100, 0., 50.);
    fPairSource2d_mT1vRinv = new TH2D(TString::Format("PairSource2d_mT1vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("PairSource2d_mT1vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                       400, 0., 10., 
                                       100, 0., 50.);
    fPairSource2d_mT2vRinv = new TH2D(TString::Format("PairSource2d_mT2vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("PairSource2d_mT2vRinv%s", cAnalysisBaseTags[fAnalysisType]),
                                       400, 0., 10.,
                                       100, 0., 50.);
    fPairSource2d_mT1vR1PRF = new TH2D(TString::Format("PairSource2d_mT1vR1PRF%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("PairSource2d_mT1vR1PRF%s", cAnalysisBaseTags[fAnalysisType]),
                                       400, 0., 10., 
                                       100, 0., 50.);
    fPairSource2d_mT2vR2PRF = new TH2D(TString::Format("PairSource2d_mT2vR2PRF%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("PairSource2d_mT2vR2PRF%s", cAnalysisBaseTags[fAnalysisType]),
                                       400, 0., 10.,
                                       100, 0., 50.);

    fPairSource3d_mT1vmT2vRinv->Sumw2();
    fPairSource2d_PairmTvRinv->Sumw2();
    fPairSource2d_mT1vRinv->Sumw2();
    fPairSource2d_mT2vRinv->Sumw2();
    fPairSource2d_mT1vR1PRF->Sumw2();
    fPairSource2d_mT2vR2PRF->Sumw2();
  }

}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetBuildCorrelationFunctions(bool aBuild, bool aBuild3dHists, bool aBuildPairSourcewmTInfo)
{
  fBuildCorrelationFunctions = aBuild;
  fBuild3dHists = aBuild3dHists;
  fBuildPairSourcewmTInfo = aBuildPairSourcewmTInfo;
  if(fBuildCorrelationFunctions) InitiateCorrelations();
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetBuildCfYlm(bool aSet)
{
  fBuildCfYlm = aSet;
  if(fBuildCfYlm) 
  {
    assert(fBuildCorrelationFunctions);

    TString tName = TString::Format("DirectYlmCf_%s", cAnalysisBaseTags[fAnalysisType]);
    int tMaxl=2;
    fCfYlm = new CorrFctnDirectYlm(tName.Data(), tMaxl, fNumFull->GetNbinsX(), fNumFull->GetXaxis()->GetXmin(), fNumFull->GetXaxis()->GetXmax());
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::InitiatePairFractionsAndParentsMatrix()
{
  TString tPairFractionsName = TString::Format("PairFractions%s", cAnalysisBaseTags[fAnalysisType]);
  fPairFractions = new TH1D(tPairFractionsName, tPairFractionsName, 12, 0, 12);
  fPairFractions->Sumw2();

  TString tParentsMatrixName = TString::Format("ParentsMatrix%s", cAnalysisBaseTags[fAnalysisType]);
  fParentsMatrix = new TH2D(tParentsMatrixName, tParentsMatrixName, 100, 0, 100, 135, 0, 135);
  fParentsMatrix->Sumw2();
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetBuildPairFractions(bool aBuild)
{
  fBuildPairFractions = aBuild;
  if(fBuildPairFractions) InitiatePairFractionsAndParentsMatrix();
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::LoadChargedResiduals()
{
  fChargedResiduals.clear();
  fChargedResidualsTypeMap.clear();
  fChargedResidualsTypeMap.resize(3);

  if(fAnalysisType==kLamKchP)
  {
    fChargedResidualsTypeMap[0] = kResXiCKchP;
    fChargedResidualsTypeMap[1] = kResSigStPKchP;
    fChargedResidualsTypeMap[2] = kResSigStMKchP;
  }
  else if(fAnalysisType==kALamKchM)
  {
    fChargedResidualsTypeMap[0] = kResAXiCKchM;
    fChargedResidualsTypeMap[1] = kResASigStMKchM;
    fChargedResidualsTypeMap[2] = kResASigStPKchM;
  }
  else if(fAnalysisType==kLamKchM)
  {
    fChargedResidualsTypeMap[0] = kResXiCKchM;
    fChargedResidualsTypeMap[1] = kResSigStPKchM;
    fChargedResidualsTypeMap[2] = kResSigStMKchM;
  }
  else if(fAnalysisType==kALamKchP)
  {
    fChargedResidualsTypeMap[0] = kResAXiCKchP;
    fChargedResidualsTypeMap[1] = kResASigStMKchP;
    fChargedResidualsTypeMap[2] = kResASigStPKchP;
  }
  else assert(0);

  for(unsigned int i=0; i<fChargedResidualsTypeMap.size(); i++)
  {
    fChargedResiduals.emplace_back(fChargedResidualsTypeMap[i]);
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetWeightCfsWithParentInteraction(bool aSet) 
{
  fWeightCfsWithParentInteraction = aSet;

  if((fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || 
     fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) &&
     fWeightCfsWithParentInteraction) LoadChargedResiduals();

}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::SetOnlyWeightLongDecayParents(bool aSet)
{
  fOnlyWeightLongDecayParents = aSet;
  if(fOnlyWeightLongDecayParents) fWeightCfsWithParentInteraction = fOnlyWeightLongDecayParents;
}

//________________________________________________________________________________________________________________
bool ThermPairAnalysis::IsChargedResidual(ParticlePDGType aType1, ParticlePDGType aType2)
{
  for(unsigned int i=0; i<fChargedResiduals.size(); i++)
  {
    vector<ParticlePDGType> tTypes = fChargedResiduals[i].GetPartTypes();
    if(tTypes[0]==aType1 && tTypes[1]==aType2) return true;
  }
  return false;
}

//________________________________________________________________________________________________________________
AnalysisType ThermPairAnalysis::GetChargedResidualType(ParticlePDGType aType1, ParticlePDGType aType2)
{
  for(unsigned int i=0; i<fChargedResiduals.size(); i++)
  {
    vector<ParticlePDGType> tTypes = fChargedResiduals[i].GetPartTypes();
    if(tTypes[0]==aType1 && tTypes[1]==aType2) return fChargedResiduals[i].GetResidualType();
  }
  assert(0);
  return kLamK0;
}

//________________________________________________________________________________________________________________
int ThermPairAnalysis::GetChargedResidualIndex(ParticlePDGType aType1, ParticlePDGType aType2)
{
  for(unsigned int i=0; i<fChargedResiduals.size(); i++)
  {
    vector<ParticlePDGType> tTypes = fChargedResiduals[i].GetPartTypes();
    if(tTypes[0]==aType1 && tTypes[1]==aType2) return i;
  }
  assert(0);
  return 0;
}


//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetFatherKStar(const ThermParticle &aParticle1, const ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2)
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
double ThermPairAnalysis::GetKStar(const ThermParticle &aParticle1, const ThermParticle &aParticle2)
{
  return GetFatherKStar(aParticle1, aParticle2, false, false);
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillTransformMatrixParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
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
void ThermPairAnalysis::FillTransformMatrixV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
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
void ThermPairAnalysis::BuildTransformMatrixParticleV0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;


  aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  if(!fMixEventsForTransforms)  //no mixing
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
void ThermPairAnalysis::BuildTransformMatrixV0V0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;


  aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  if(!fMixEventsForTransforms)  //no mixing
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
void ThermPairAnalysis::BuildAllTransformMatrices(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection)
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
vector<int> ThermPairAnalysis::UniqueCombineVectors(const vector<int> &aVec1, const vector<int> &aVec2)
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
void ThermPairAnalysis::FillPrimaryAndOtherPairInfo(int aParentType1, int aParentType2)
{
  bool bPairAlreadyIncluded = false;

  if(IncludeAsPrimary(aParentType1, aParentType2, fMaxPrimaryDecayLength))
  {
    for(unsigned int i=0; i<fPrimaryPairInfo.size(); i++)
    {
      if(fPrimaryPairInfo[i][0].pdgType == static_cast<ParticlePDGType>(aParentType1) &&
         fPrimaryPairInfo[i][1].pdgType == static_cast<ParticlePDGType>(aParentType2)) bPairAlreadyIncluded = true;
    }
    if(!bPairAlreadyIncluded) fPrimaryPairInfo.push_back(vector<PidInfo>{GetParticlePidInfo(aParentType1),GetParticlePidInfo(aParentType2)});
  }

  //--------------------
  if(IncludeInOthers(aParentType1, aParentType2, fMaxPrimaryDecayLength))
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
void ThermPairAnalysis::BuildPairFractionHistogramsParticleV0(const ThermEvent &aEvent)
{
  ParticlePDGType tParticleType = fTransformInfo[0].particleType2;
  ParticlePDGType tV0Type       = fTransformInfo[0].particleType1;

  const vector<ThermV0Particle> aV0Collection =  aEvent.GetV0ParticleCollection(tV0Type);
  const vector<ThermParticle>   aParticleCollection = aEvent.GetParticleCollection(tParticleType);

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

        MapAndFillPairFractionHistogramParticleV0(fPairFractions, tV0FatherType, tParticleFatherType, fMaxPrimaryDecayLength);
        FillPrimaryAndOtherPairInfo(tV0FatherType, tParticleFatherType);
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
void ThermPairAnalysis::BuildPairFractionHistogramsV0V0(const ThermEvent &aEvent)
{
  ParticlePDGType tV01Type = fTransformInfo[0].particleType1;
  ParticlePDGType tV02Type = fTransformInfo[0].particleType2;

  const vector<ThermV0Particle> aV01Collection =  aEvent.GetV0ParticleCollection(tV01Type);
  const vector<ThermV0Particle> aV02Collection =  aEvent.GetV0ParticleCollection(tV02Type);

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
          MapAndFillPairFractionHistogramV0V0(fPairFractions, tV01FatherType, tV02FatherType, fMaxPrimaryDecayLength);
          FillPrimaryAndOtherPairInfo(tV01FatherType, tV02FatherType);
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
double ThermPairAnalysis::CalcKStar(const TLorentzVector &p1, const TLorentzVector &p2)
{
  const double p_inv = (p1 + p2).Mag2(),
               q_inv = (p1 - p2).Mag2(),
           mass_diff = p1.Mag2() - p2.Mag2();

  const double tQ = ::pow(mass_diff, 2) / p_inv - q_inv;
  return ::sqrt(tQ) / 2.0;
}


//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcKStar(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  return CalcKStar(p1, p2);
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcKStar_RotatePar2(const TLorentzVector &p1, const TLorentzVector &p2)
{
  TLorentzVector p2_Rot = TLorentzVector(-1.*p2.Px(), -1.*p2.Py(), p2.Pz(), p2.E());
  return CalcKStar(p1, p2_Rot);
}


//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcKStar_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  return CalcKStar_RotatePar2(p1, p2);
}

//________________________________________________________________________________________________________________
TLorentzVector ThermPairAnalysis::Boost4VecToOSLinLCMS(const TLorentzVector &p1, const TLorentzVector &p2, const TLorentzVector &aVecToBoost)
{
  const TLorentzVector tPTot = p1 + p2;

  assert(tPTot.Mt() - tPTot.Pt() > 0.0);
  if (tPTot.Mt() == 0 || tPTot.E() == 0 || tPTot.M() == 0 || tPTot.Pt() == 0 ) assert(0);

  // Boost to LCMS
  double tBeta = tPTot.Pz()/tPTot.E();
  double tGamma = tPTot.E()/tPTot.Mt();
  double tVtbLcmsLong = tGamma*(aVecToBoost.Z() - tBeta*aVecToBoost.T());
  double tVtbLcmsT    = tGamma*(aVecToBoost.T()  - tBeta*aVecToBoost.Z());

  // Rotate in transverse plane
  double tVtbLcmsOut  = ( aVecToBoost.X()*tPTot.Px() + aVecToBoost.Y()*tPTot.Py())/tPTot.Pt();
  double tVtbLcmsSide = (-aVecToBoost.X()*tPTot.Py() + aVecToBoost.Y()*tPTot.Px())/tPTot.Pt();

  return TLorentzVector(tVtbLcmsOut, tVtbLcmsSide, tVtbLcmsLong, tVtbLcmsT);
}


//________________________________________________________________________________________________________________
TLorentzVector ThermPairAnalysis::Boost4VecToOSLinPRF(const TLorentzVector &p1, const TLorentzVector &p2, const TLorentzVector &aVecToBoost)
{
  const TLorentzVector tPTot = p1 + p2;

  assert(tPTot.Mt() - tPTot.Pt() > 0.0);
  if (tPTot.Mt() == 0 || tPTot.E() == 0 || tPTot.M() == 0 || tPTot.Pt() == 0 ) assert(0);

  // Boost to LCMS and rotate in transverse plane to OSL
  TLorentzVector tVtbOslInLcms = Boost4VecToOSLinLCMS(p1, p2, aVecToBoost);
  double tVtbLcmsOut  = tVtbOslInLcms.X();
  double tVtbLcmsSide = tVtbOslInLcms.Y();
  double tVtbLcmsLong = tVtbOslInLcms.Z();
  double tVtbLcmsT    = tVtbOslInLcms.T();


  // Boost to pair cms
  double tBeta = tPTot.Pt()/tPTot.Mt();
  double tGamma = tPTot.Mt()/tPTot.M();
  double tVtbPrfOut  = tGamma*(tVtbLcmsOut - tBeta*tVtbLcmsT);
  double tVtbPrfSide = tVtbLcmsSide;
  double tVtbPrfLong = tVtbLcmsLong;
  double tVtbPrfT    = tGamma*(tVtbLcmsT - tBeta*tVtbLcmsOut);

  return TLorentzVector(tVtbPrfOut, tVtbPrfSide, tVtbPrfLong, tVtbPrfT);
}


//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetKStar3Vec(const TLorentzVector &p1, const TLorentzVector &p2)
{
  TLorentzVector tKStar4Vec = Boost4VecToOSLinPRF(p1, p2, p1);
  return tKStar4Vec.Vect();
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetKStar3Vec(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  //---------------------------------

  return GetKStar3Vec(p1, p2);
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetKStar3Vec_RotatePar2(const TLorentzVector &p1, const TLorentzVector &p2)
{
  TLorentzVector p2_Rot = TLorentzVector(-1.*p2.Px(), -1.*p2.Py(), p2.Pz(), p2.E());
  TVector3 tKStar3Vec = GetKStar3Vec(p1, p2_Rot);
  return tKStar3Vec;
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetKStar3Vec_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  //---------------------------------

  return GetKStar3Vec_RotatePar2(p1, p2);
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::DrawRStar3VecFromGaussian(double tROut, double tMuOut, double tRSide, double tMuSide, double tRLong, double tMuLong)
{
  //Create the source Gaussians
  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2

//  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//  std::default_random_engine generator (seed);  //std::clock() is seed
  std::normal_distribution<double> tROutSource(tMuOut,tRoot2*tROut);
  std::normal_distribution<double> tRSideSource(tMuSide,tRoot2*tRSide);
  std::normal_distribution<double> tRLongSource(tMuLong,tRoot2*tRLong);

  return TVector3(tROutSource(fGenerator),tRSideSource(fGenerator),tRLongSource(fGenerator));
}

//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::DrawRStar3VecFromGaussian()
{
  double tROut, tMuOut, tRSide, tMuSide, tRLong, tMuLong;

  tROut = 5.;
  tMuOut = 0.;

  tRSide = 5.;
  tMuSide = 0.;

  tRLong = 5.;
  tMuLong = 0.;

  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM)
  {
    tMuOut = 3.;
  }
  else if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP)
  {
    tMuOut = 3.;
  }
  else if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
  {
    tMuOut = 3.;
  }
  else if(fAnalysisType==kKchPKchP)
  {
    tROut = 5.;
    tRSide = 3.;
    tRLong = 8.;
  }
  else if(fAnalysisType==kK0K0)
  {
    tROut = 5.;
    tRSide = 5.;
    tRLong = 5.;
  }
  else if(fAnalysisType==kLamLam)
  {
    tROut = 3.;
    tRSide = 3.;
    tRLong = 3.;
  }
  else assert(0);

  return DrawRStar3VecFromGaussian(tROut, tMuOut, tRSide, tMuSide, tRLong, tMuLong);
}


//________________________________________________________________________________________________________________
TLorentzVector ThermPairAnalysis::GetRStar4Vec(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2)
{
  TLorentzVector tRInLabFrame4Vec = x1 - x2;
  TLorentzVector tRStar4Vec = Boost4VecToOSLinPRF(p1, p2, tRInLabFrame4Vec);

  return tRStar4Vec;
}


//________________________________________________________________________________________________________________
TLorentzVector ThermPairAnalysis::GetRStar4Vec(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  TLorentzVector x1 = tPart1.GetFourPosition();
  TLorentzVector x2 = tPart2.GetFourPosition();

  //---------------------------------

  return GetRStar4Vec(p1, x1, p2, x2);
}


//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetRStar3Vec(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2)
{
  TLorentzVector tRStar4Vec = GetRStar4Vec(p1, x1, p2, x2);
  return tRStar4Vec.Vect();
}


//________________________________________________________________________________________________________________
TVector3 ThermPairAnalysis::GetRStar3Vec(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector tRStar4Vec = GetRStar4Vec(tPart1, tPart2);
  return tRStar4Vec.Vect();
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcRStar(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2)
{
  TVector3 tRStar3Vec = GetRStar3Vec(p1, x1, p2, x2);
  return tRStar3Vec.Mag();
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcRStar(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TVector3 tRStar3Vec = GetRStar3Vec(tPart1, tPart2);
  return tRStar3Vec.Mag();
}



//________________________________________________________________________________________________________________
vector<double> ThermPairAnalysis::CalcR1R2inPRF(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2)
{
  TLorentzVector tR1InLabFrame4Vec = x1;
  TLorentzVector tR1Star4Vec = Boost4VecToOSLinPRF(p1, p2, tR1InLabFrame4Vec);
  TVector3 tR1Star3Vec = tR1Star4Vec.Vect();

  TLorentzVector tR2InLabFrame4Vec = x2;
  TLorentzVector tR2Star4Vec = Boost4VecToOSLinPRF(p1, p2, tR2InLabFrame4Vec);
  TVector3 tR2Star3Vec = tR2Star4Vec.Vect();

  return vector<double>{tR1Star3Vec.Mag(), tR2Star3Vec.Mag()};
}

//________________________________________________________________________________________________________________
vector<double> ThermPairAnalysis::CalcR1R2inPRF(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector p1 = tPart1.GetFourMomentum();
  TLorentzVector p2 = tPart2.GetFourMomentum();

  TLorentzVector x1 = tPart1.GetFourPosition();
  TLorentzVector x2 = tPart2.GetFourPosition();

  //---------------------------------

  return CalcR1R2inPRF(p1, x1, p2, x2);
}



//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcmT(const TLorentzVector &p1, const TLorentzVector &p2)
{
  TLorentzVector tPTot = p1+p2;
  return 0.5*tPTot.Mt();
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::CalcmT(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  TLorentzVector tP1 = tPart1.GetFourMomentum();
  TLorentzVector tP2 = tPart2.GetFourMomentum();
  return CalcmT(tP1, tP2);
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillParentmT3d(TH3* aPairmT3d, const ThermParticle &tPart1, const ThermParticle &tPart2)
{
  //--------------------------------------------------------------
  TLorentzVector p1, p2;
  int tIndex1, tIndex2;
  double tmT = 0.;
  //----------
  if(tPart1.IsPrimordial())
  {
    p1 = tPart1.GetFourMomentum();
    tIndex1 = GetParticleIndexInPidInfo(tPart1.GetPID());
  }
  else
  {
    p1 = tPart1.GetFatherFourMomentum();
    tIndex1 = GetParticleIndexInPidInfo(tPart1.GetFatherPID());
  }
  //----------
  if(tPart2.IsPrimordial())
  {
    p2 = tPart2.GetFourMomentum();
    tIndex2 = GetParticleIndexInPidInfo(tPart2.GetPID());
  }
  else
  {
    p2 = tPart2.GetFatherFourMomentum();
    tIndex2 = GetParticleIndexInPidInfo(tPart2.GetFatherPID());
  }
  //--------------------------------------------------------------
  tmT = CalcmT(p1,p2);
  aPairmT3d->Fill(tIndex1, tIndex2, tmT);
}



//________________________________________________________________________________________________________________
complex<double> ThermPairAnalysis::GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec)
{
  TVector3 tRStar3Vec;
  if(aRStar3Vec.X()==0 && aRStar3Vec.Y()==0 && aRStar3Vec.Z()==0)  //TODO i.e. if pair originate from single resonance
  {
    double tRoot2 = sqrt(2.);
    double tRadius = 1.0;
    std::default_random_engine generator (std::clock());  //std::clock() is seed
    std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

    tRStar3Vec.SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator));
  }
  else tRStar3Vec = aRStar3Vec;


  complex<double> ImI (0., 1.);
  complex<double> tF0 (0., 0.);
  double tD0 = 0.;
/*
  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM) tF0 = complex<double>(-0.5,0.5);
  else if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) tF0 = complex<double>(0.25,0.5);
  else if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0) tF0 = complex<double>(-0.25,0.25);
  else assert(0);
*/
  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM)      {tF0 = complex<double>(-1.16, 0.51); tD0 = 1.08;}
  else if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) {tF0 = complex<double>(0.41, 0.47); tD0 = -4.89;}
  else if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)     {tF0 = complex<double>(-0.41, 0.20); tD0 = 2.08;}
  else if(fAnalysisType==kKchPKchP || fAnalysisType==kK0K0 || fAnalysisType==kLamLam)      {tF0 = complex<double>(-1.16, 0.51); tD0 = 1.08;}
  else assert(0);

  double tKdotR = aKStar3Vec.Dot(tRStar3Vec);
    tKdotR /= hbarc;
  double tKStarMag = aKStar3Vec.Mag();
    tKStarMag /= hbarc;
  double tRStarMag = tRStar3Vec.Mag();

  complex<double> tScattLenLastTerm (0., tKStarMag);
  complex<double> tScattAmp = pow((1./tF0) + 0.5*tD0*tKStarMag*tKStarMag - tScattLenLastTerm,-1);

//  complex<double> tReturnWf = exp(ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  complex<double> tReturnWf = exp(-ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  return tReturnWf;
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec)
{
  complex<double> tWf = GetStrongOnlyWaveFunction(aKStar3Vec, aRStar3Vec);
  double tWfSq = norm(tWf);
  return tWfSq;
}

//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetParentPairWaveFunctionSq(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
//  assert(!(tPart1.IsPrimordial() && tPart2.IsPrimordial()));
  if(fDrawRStarFromGaussian)
  {
    cout << "fDrawRStarFromGaussian=true!" << endl;
    cout << "\tIncompatible with fWeightCfsWithParentInteraction=true, ";
    cout << "\tas pair source will be drawn from random Gaussian, but parent source will still use space information from THERMINATOR" << endl;
    cout << "\t\tPREPARE FOR CRASH" << endl;
  }
  assert(!fDrawRStarFromGaussian);
  //--------------------------------------------------------------
  TLorentzVector p1, x1, p2, x2;
  ParticlePDGType tResType1, tResType2;
  //----------
  if(tPart1.IsPrimordial())
  {
    p1 = tPart1.GetFourMomentum();
    x1 = tPart1.GetFourPosition();
    tResType1 = static_cast<ParticlePDGType>(tPart1.GetPID());
  }
  else
  {
    p1 = tPart1.GetFatherFourMomentum();
    x1 = tPart1.GetFatherFourPosition();
    tResType1 = static_cast<ParticlePDGType>(tPart1.GetFatherPID());

    if(fOnlyWeightLongDecayParents && GetParticleDecayLength(tResType1) < 1000.)
    {
      p1 = tPart1.GetFourMomentum();
      x1 = tPart1.GetFourPosition();
      tResType1 = static_cast<ParticlePDGType>(tPart1.GetPID());
    }
  }
  //----------
  if(tPart2.IsPrimordial())
  {
    p2 = tPart2.GetFourMomentum();
    x2 = tPart2.GetFourPosition();
    tResType2 = static_cast<ParticlePDGType>(tPart2.GetPID());
  }
  else
  {
    p2 = tPart2.GetFatherFourMomentum();
    x2 = tPart2.GetFatherFourPosition();
    tResType2 = static_cast<ParticlePDGType>(tPart2.GetFatherPID());

    if(fOnlyWeightLongDecayParents && GetParticleDecayLength(tResType2) < 1000.)
    {
      p2 = tPart2.GetFourMomentum();
      x2 = tPart2.GetFourPosition();
      tResType2 = static_cast<ParticlePDGType>(tPart2.GetPID());
    }
  }
  //--------------------------------------------------------------

  TVector3 tRStar3Vec = GetRStar3Vec(p1, x1, p2, x2);
  TVector3 tKStar3Vec = GetKStar3Vec(p1, p2);

  //--------------------------------------------------------------

  if(!IsChargedResidual(tResType1, tResType2)) return GetStrongOnlyWaveFunctionSq(tKStar3Vec, tRStar3Vec);
  else
  {
    int tResIndex = GetChargedResidualIndex(tResType1, tResType2);
    return fChargedResiduals[tResIndex].InterpolateWfSquared(&tKStar3Vec, &tRStar3Vec);
  }

}


//________________________________________________________________________________________________________________
double ThermPairAnalysis::GetParentPairWaveFunctionSq_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2)
{
//  assert(!(tPart1.IsPrimordial() && tPart2.IsPrimordial()));
  if(fDrawRStarFromGaussian)
  {
    cout << "fDrawRStarFromGaussian=true!" << endl;
    cout << "\tIncompatible with fWeightCfsWithParentInteraction=true, ";
    cout << "\tas pair source will be drawn from random Gaussian, but parent source will still use space information from THERMINATOR" << endl;
    cout << "\t\tPREPARE FOR CRASH" << endl;
  }
  assert(!fDrawRStarFromGaussian);
  //--------------------------------------------------------------
  TLorentzVector p1, x1, p2, x2;
  ParticlePDGType tResType1, tResType2;
  //----------
  if(tPart1.IsPrimordial())
  {
    p1 = tPart1.GetFourMomentum();
    x1 = tPart1.GetFourPosition();
    tResType1 = static_cast<ParticlePDGType>(tPart1.GetPID());
  }
  else
  {
    p1 = tPart1.GetFatherFourMomentum();
    x1 = tPart1.GetFatherFourPosition();
    tResType1 = static_cast<ParticlePDGType>(tPart1.GetFatherPID());

    if(fOnlyWeightLongDecayParents && GetParticleDecayLength(tResType1) < 1000.)
    {
      p1 = tPart1.GetFourMomentum();
      x1 = tPart1.GetFourPosition();
      tResType1 = static_cast<ParticlePDGType>(tPart1.GetPID());
    }
  }
  //----------
  if(tPart2.IsPrimordial())
  {
    p2 = tPart2.GetFourMomentum();
    x2 = tPart2.GetFourPosition();
    tResType2 = static_cast<ParticlePDGType>(tPart2.GetPID());
  }
  else
  {
    p2 = tPart2.GetFatherFourMomentum();
    x2 = tPart2.GetFatherFourPosition();
    tResType2 = static_cast<ParticlePDGType>(tPart2.GetFatherPID());

    if(fOnlyWeightLongDecayParents && GetParticleDecayLength(tResType2) < 1000.)
    {
      p2 = tPart2.GetFourMomentum();
      x2 = tPart2.GetFourPosition();
      tResType2 = static_cast<ParticlePDGType>(tPart2.GetPID());
    }
  }
  //--------------------------------------------------------------

  TVector3 tRStar3Vec = GetRStar3Vec(p1, x1, p2, x2);
  TVector3 tKStar3Vec = GetKStar3Vec_RotatePar2(p1, p2);

  //--------------------------------------------------------------

  if(!IsChargedResidual(tResType1, tResType2)) return GetStrongOnlyWaveFunctionSq(tKStar3Vec, tRStar3Vec);
  else
  {
    int tResIndex = GetChargedResidualIndex(tResType1, tResType2);
    return fChargedResiduals[tResIndex].InterpolateWfSquared(&tKStar3Vec, &tRStar3Vec);
  }

}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelations(const ThermParticle &aParticle1, const ThermParticle &aParticle2, bool aFillNumerator)
{
  //For ParticleV0, aParticle1=V0 and aParticle2=Particle
  TH3 *tCf3d;
  TH1 *tCfFull, *tCfPrimaryOnly, *tCfPrimaryAndShortDecays, *tCfWithoutSigmaSt, *tCfSigmaStOnly, *tCfSecondaryOnly;

  if(aFillNumerator && aParticle1.GetPID()==aParticle2.GetPID() && aParticle1.GetEID()==aParticle2.GetEID()) return;  //Particles are same!

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

  int tParentIndex1 = -1;
  int tParentIndex2 = -1;

  double tRStar = 0.;
  double tKStar = 0.;
  double tWeight = 1.;

  tParentIndex1 = GetParticleIndexInPidInfo(aParticle1.GetFatherPID());
  tParentIndex2 = GetParticleIndexInPidInfo(aParticle2.GetFatherPID());

  TVector3 tKStar3Vec = GetKStar3Vec(aParticle1, aParticle2);
  TVector3 tKStar3Vec_RotatePar2 = GetKStar3Vec_RotatePar2(aParticle1, aParticle2);
  tKStar = CalcKStar(aParticle1, aParticle2);

  TLorentzVector tRStar4Vec = TLorentzVector(0., 0., 0., 0.);
  if(!fDrawRStarFromGaussian) tRStar4Vec = GetRStar4Vec(aParticle1, aParticle2);

  TVector3 tRStar3Vec;
  if(!fDrawRStarFromGaussian) tRStar3Vec = tRStar4Vec.Vect();
  else                        tRStar3Vec = DrawRStar3VecFromGaussian();
  tRStar = tRStar3Vec.Mag();

  if(aFillNumerator)
  {
    if(fUnitWeightCfNums) tWeight = 1.;
    else if(fWeightCfsWithParentInteraction) tWeight = GetParentPairWaveFunctionSq(aParticle1, aParticle2);
    else tWeight = GetStrongOnlyWaveFunctionSq(tKStar3Vec, tRStar3Vec);
  }

  if(fBuild3dHists) tCf3d->Fill(tParentIndex1, tParentIndex2, tKStar, tWeight);
  tCfFull->Fill(tKStar, tWeight);

  if(fBuildCfYlm)
  {
    if(aFillNumerator) fCfYlm->AddRealPair(tKStar3Vec.X(), tKStar3Vec.Y(), tKStar3Vec.Z(), tWeight);
    else               fCfYlm->AddMixedPair(tKStar3Vec.X(), tKStar3Vec.Y(), tKStar3Vec.Z(), tWeight);
  }

  if(aFillNumerator)
  {
    if(fBuild3dHists) fPairSource3d->Fill(tParentIndex1, tParentIndex2, tRStar);
    fPairSourceFull->Fill(tRStar);

    if(tKStar < 0.3)
    {
      fPairSource3d_OSLinPRF->Fill(tRStar3Vec.x(), tRStar3Vec.y(), tRStar3Vec.z());
      if(aParticle1.IsPrimordial() && aParticle2.IsPrimordial()) fPairSource3d_OSLinPRFPrimaryOnly->Fill(tRStar3Vec.x(), tRStar3Vec.y(), tRStar3Vec.z());

      if(!fDrawRStarFromGaussian)
      {
        fPairDeltaT_inPRF->Fill(tRStar4Vec.T());
        if(aParticle1.IsPrimordial() && aParticle2.IsPrimordial()) fPairDeltaT_inPRFPrimaryOnly->Fill(tRStar4Vec.T());
      }

      if(fBuildPairSourcewmTInfo)
      {
        fPairSource3d_mT1vmT2vRinv->Fill(aParticle1.GetMt(), aParticle2.GetMt(), tRStar);
        fPairSource2d_PairmTvRinv->Fill(CalcmT(aParticle1, aParticle2), tRStar);
        fPairSource2d_mT1vRinv->Fill(aParticle1.GetMt(), tRStar);
        fPairSource2d_mT2vRinv->Fill(aParticle2.GetMt(), tRStar);
        vector<double> tR1R2inPRF = CalcR1R2inPRF(aParticle1, aParticle2);
        fPairSource2d_mT1vR1PRF->Fill(aParticle1.GetMt(), tR1R2inPRF[0]);
        fPairSource2d_mT2vR2PRF->Fill(aParticle2.GetMt(), tR1R2inPRF[1]);
      }
    }
    //--------------------------------------
    if(fUnitWeightCfNums) fNumFull_RotatePar2->Fill(CalcKStar_RotatePar2(aParticle1, aParticle2), 1.);
    else if(fWeightCfsWithParentInteraction) fNumFull_RotatePar2->Fill(CalcKStar_RotatePar2(aParticle1, aParticle2), GetParentPairWaveFunctionSq_RotatePar2(aParticle1, aParticle2));
    else fNumFull_RotatePar2->Fill(CalcKStar_RotatePar2(aParticle1, aParticle2), GetStrongOnlyWaveFunctionSq(tKStar3Vec_RotatePar2, tRStar3Vec));
    //--------------------------------------    

    fPairKStarVsmT->Fill(tKStar, CalcmT(aParticle1, aParticle2));
    if(tKStar <= 0.3 && fBuild3dHists) FillParentmT3d(fPairmT3d, aParticle1, aParticle2);
  }

  if(aParticle1.IsPrimordial() && aParticle2.IsPrimordial())
  {
    tCfPrimaryOnly->Fill(tKStar, tWeight);
    if(aFillNumerator) fPairSourcePrimaryOnly->Fill(tRStar);
  }

  if((aParticle1.IsPrimordial() && aParticle2.IsPrimordial()) || IncludeAsPrimary(aParticle1.GetFatherPID(), aParticle2.GetFatherPID(), fMaxPrimaryDecayLength))
  {
    tCfPrimaryAndShortDecays->Fill(tKStar, tWeight);
    if(aFillNumerator) fPairSourcePrimaryAndShortDecays->Fill(tRStar);
  }

  if(aParticle1.GetFatherPID() != kPDGSigStP && aParticle1.GetFatherPID() != kPDGASigStM && 
     aParticle1.GetFatherPID() != kPDGSigStM && aParticle1.GetFatherPID() != kPDGASigStP && 
     aParticle1.GetFatherPID() != kPDGSigSt0 && aParticle1.GetFatherPID() != kPDGASigSt0)
  {
    tCfWithoutSigmaSt->Fill(tKStar, tWeight);
    if(aFillNumerator) fPairSourceWithoutSigmaSt->Fill(tRStar);
  }
  else
  {
    tCfSigmaStOnly->Fill(tKStar, tWeight);
    if(aFillNumerator) fPairSourceSigmaStOnly->Fill(tRStar);
  }

  if(!aParticle1.IsPrimordial() && !aParticle2.IsPrimordial())
  {
    tCfSecondaryOnly->Fill(tKStar, tWeight);
    if(aFillNumerator) fPairSourceSecondaryOnly->Fill(tRStar);
  }

}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumOrDenParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2, bool aFillNumerator)
{
  ThermParticle tParticle1, tParticle2;

  for(unsigned int iPar1=0; iPar1<aParticleCollection1.size(); iPar1++)
  {
    tParticle1 = aParticleCollection1[iPar1];

    for(unsigned int iPar2=0; iPar2<aParticleCollection2.size(); iPar2++)
    {
      tParticle2 = aParticleCollection2[iPar2];
      FillCorrelations(tParticle1, tParticle2, aFillNumerator);
    }

  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumOrDenParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection, bool aFillNumerator)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GoodV0())
    {
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        FillCorrelations(tV0, tParticle, aFillNumerator);
      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumOrDenV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection, bool aFillNumerator)
{
  ThermV0Particle tV01, tV02;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if(tV01.GoodV0())
    {
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if(tV02.GoodV0()) FillCorrelations(tV01, tV02, aFillNumerator);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumAndDenParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2)
{
  ThermParticle tParticle1, tParticle2;

  for(unsigned int iPar1=0; iPar1<aParticleCollection1.size(); iPar1++)
  {
    tParticle1 = aParticleCollection1[iPar1];

    for(unsigned int iPar2=0; iPar2<aParticleCollection2.size(); iPar2++)
    {
      tParticle2 = aParticleCollection2[iPar2];
      FillCorrelations(tParticle1, tParticle2, true);
      FillCorrelations(tParticle1, tParticle2, false);
    }

  }
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumAndDenParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GoodV0())
    {
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        FillCorrelations(tV0, tParticle, true);
        FillCorrelations(tV0, tParticle, false);
      }
    }
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::FillCorrelationFunctionsNumAndDenV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection)
{
  ThermV0Particle tV01, tV02;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if(tV01.GoodV0())
    {
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if(tV02.GoodV0())
        {
          FillCorrelations(tV01, tV02, true);
          FillCorrelations(tV01, tV02, false);
        }
      }
    }
  }
}



//________________________________________________________________________________________________________________
td2dVec ThermPairAnalysis::GetTrueRoslContributions(const ThermParticle &aParticle1, const ThermParticle &aParticle2)
{
  TLorentzVector tRStar4Vec = GetRStar4Vec(aParticle1, aParticle2);

  double tXo = tRStar4Vec.X();
  double tXs = tRStar4Vec.Y();
  double tXl = tRStar4Vec.Z();
  double tt  = tRStar4Vec.T();

  //----------------------------------------------

  const TLorentzVector tp1 = aParticle1.GetFourMomentum();
  const TLorentzVector tp2 = aParticle2.GetFourMomentum();

  const TLorentzVector tPTot = tp1+tp2;
  assert(tPTot.Mt() - tPTot.Pt() > 0.0);
  if (tPTot.Mt() == 0 || tPTot.E() == 0 || tPTot.M() == 0 || tPTot.Pt() == 0 ) assert(0);

  double tBetaL = tPTot.Pz()/tPTot.E();
  double tBetaT = tPTot.Pt()/tPTot.Mt();

  //----------------------------------------------
  //----------------- Out ------------------------
  double tXoSq = tXo*tXo;

  double tXoBetaTt = tXo*tBetaT*tt;
  double tXoBetaT = tXo*tBetaT;
  double tBetaTt = tBetaT*tt;

  double tBetaTSqtSq = tBetaT*tBetaT*tt*tt;
  double tBetaTSqt = tBetaT*tBetaT*tt;
  double tBetaTSq = tBetaT*tBetaT;

  td1dVec tVecOut = {tXoSq, tXo, tt, tXoBetaTt, tXoBetaT, tBetaTt, tBetaT, tBetaTSqtSq, tBetaTSqt, tBetaTSq};

  //----------------- Side -----------------------
  double tXsSq = tXs*tXs;
  td1dVec tVecSide = {tXsSq, tXs};


  //----------------- Long -----------------------
  double tXlSq = tXl*tXl;

  double tXlBetaLt = tXl*tBetaL*tt;
  double tXlBetaL = tXl*tBetaL;
  double tBetaLt = tBetaL*tt;

  double tBetaLSqtSq = tBetaL*tBetaL*tt*tt;
  double tBetaLSqt = tBetaL*tBetaL*tt;
  double tBetaLSq = tBetaL*tBetaL;

  td1dVec tVecLong = {tXlSq, tXl, tt, tXlBetaLt, tXlBetaL, tBetaLt, tBetaL, tBetaLSqtSq, tBetaLSqt, tBetaLSq};


  //----------------------------------------------
  td2dVec tReturnVec = {tVecOut, tVecSide, tVecLong};
  return tReturnVec;
}

//________________________________________________________________________________________________________________
void ThermPairAnalysis::AddRoslContributionToMasterVector(td2dVec &aTrueRoslMaster, int &aNPairs, const ThermParticle &aParticle1, const ThermParticle &aParticle2)
{
  //Only add pairs if k* < 0.3, and are primary or short decays
  if(CalcKStar(aParticle1, aParticle2) > 0.3) return;
  if(!(aParticle1.IsPrimordial() && aParticle2.IsPrimordial()) && !IncludeAsPrimary(aParticle1.GetFatherPID(), aParticle2.GetFatherPID(), fMaxPrimaryDecayLength)) return;
  //-----------------------------------------------

  aNPairs++;
  td2dVec tTrueRoslContribution = GetTrueRoslContributions(aParticle1, aParticle2);
    //Overkill, but ensure everything looks right...-----------------------------
    assert(tTrueRoslContribution.size()==3); //Out, side, long
    assert(tTrueRoslContribution.size()==aTrueRoslMaster.size());

    assert(tTrueRoslContribution[0].size()==10); //Out contributors
    assert(tTrueRoslContribution[0].size()==aTrueRoslMaster[0].size());

    assert(tTrueRoslContribution[1].size()==2);  //Side contributors
    assert(tTrueRoslContribution[1].size()==aTrueRoslMaster[1].size());

    assert(tTrueRoslContribution[2].size()==10); //Long contributors
    assert(tTrueRoslContribution[2].size()==aTrueRoslMaster[2].size());
    //---------------------------------------------------------------------------

  for(unsigned int iOSL=0; iOSL<tTrueRoslContribution.size(); iOSL++)
  {
    for(unsigned int iContr=0; iContr<tTrueRoslContribution[iOSL].size(); iContr++)
    {
      aTrueRoslMaster[iOSL][iContr] += tTrueRoslContribution[iOSL][iContr];
    }
  }
}

//________________________________________________________________________________________________________________
td2dVec ThermPairAnalysis::FinalizeTrueRoslMaster(td2dVec &aTrueRoslMaster, int aNPairs)
{
  //Perform final division to complete averages--------------------
  for(unsigned int iOSL=0; iOSL<aTrueRoslMaster.size(); iOSL++)
  {
    for(unsigned int iContr=0; iContr<aTrueRoslMaster[iOSL].size(); iContr++)
    {
      aTrueRoslMaster[iOSL][iContr] /= aNPairs;
    }
  }

  //------------------ Out ---------------------------------------
  //td1dVec tVecOut = {tXoSq, tXo, tt, tXoBetaTt, tXoBetaT, tBetaTt, tBetaT, tBetaTSqtSq, tBetaTSqt, tBetaTSq};

  double tAvgXoSq       = aTrueRoslMaster[0][0];
  double tAvgXo         = aTrueRoslMaster[0][1];
  double tAvgt          = aTrueRoslMaster[0][2];
  double tAvgXoBetaTt   = aTrueRoslMaster[0][3];
  double tAvgXoBetaT    = aTrueRoslMaster[0][4];
  double tAvgBetaTt     = aTrueRoslMaster[0][5];
  double tAvgBetaT      = aTrueRoslMaster[0][6];
  double tAvgBetaTSqtSq = aTrueRoslMaster[0][7];
  double tAvgBetaTSqt   = aTrueRoslMaster[0][8];
  double tAvgBetaTSq    = aTrueRoslMaster[0][9];

  double tRoSq = tAvgXoSq - tAvgXo*tAvgXo 
                -2.*tAvgXoBetaTt + 2.*tAvgt*tAvgXoBetaT + 2.*tAvgXo*tAvgBetaTt - 2.*tAvgXo*tAvgt*tAvgBetaT
                + tAvgBetaTSqtSq - 2.*tAvgt*tAvgBetaTSqt + tAvgt*tAvgt*tAvgBetaTSq;
                 

  //------------------ Side --------------------------------------
  //td1dVec tVecSide = {tXsSq, tXs};
  double tAvgXsSq = aTrueRoslMaster[1][0];
  double tAvgXs   = aTrueRoslMaster[1][1];

  double tRsSq = tAvgXsSq - tAvgXs*tAvgXs;


  //------------------ Long --------------------------------------
  //td1dVec tVecLong = {tXlSq, tXl, tt, tXlBetaLt, tXlBetaL, tBetaLt, tBetaL, tBetaLSqtSq, tBetaLSqt, tBetaLSq};

  double tAvgXlSq       = aTrueRoslMaster[2][0];
  double tAvgXl         = aTrueRoslMaster[2][1];
  assert(tAvgt==aTrueRoslMaster[2][2]);
  double tAvgXlBetaLt   = aTrueRoslMaster[2][3];
  double tAvgXlBetaL    = aTrueRoslMaster[2][4];
  double tAvgBetaLt     = aTrueRoslMaster[2][5];
  double tAvgBetaL      = aTrueRoslMaster[2][6];
  double tAvgBetaLSqtSq = aTrueRoslMaster[2][7];
  double tAvgBetaLSqt   = aTrueRoslMaster[2][8];
  double tAvgBetaLSq    = aTrueRoslMaster[2][9];

  double tRlSq = tAvgXlSq - tAvgXl*tAvgXl 
                -2.*tAvgXlBetaLt + 2.*tAvgt*tAvgXlBetaL + 2.*tAvgXl*tAvgBetaLt - 2.*tAvgXl*tAvgt*tAvgBetaL
                + tAvgBetaLSqtSq - 2.*tAvgt*tAvgBetaLSqt + tAvgt*tAvgt*tAvgBetaLSq;

  //-----------------------------------------------------------------------------------
  td1dVec tTrueRosl = {sqrt(tRoSq), sqrt(tRsSq), sqrt(tRlSq)};
  td1dVec tSimpleRosl = {tAvgXo, tAvgXs, tAvgXl};

  td2dVec tReturnVec{tTrueRosl, tSimpleRosl};
  return tReturnVec;
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTrueRoslParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2)
{
  ThermParticle tParticle1, tParticle2;

  td2dVec tTrueRoslMaster{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0},
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int tNPairs = 0;

  for(unsigned int iPar1=0; iPar1<aParticleCollection1.size(); iPar1++)
  {
    tParticle1 = aParticleCollection1[iPar1];

    for(unsigned int iPar2=0; iPar2<aParticleCollection2.size(); iPar2++)
    {
      tParticle2 = aParticleCollection2[iPar2];
      AddRoslContributionToMasterVector(tTrueRoslMaster, tNPairs, tParticle1, tParticle2);
    }
  }
  //---------------
  if(tNPairs>0)
  {
    td2dVec tTrueAndSimpleRosl = FinalizeTrueRoslMaster(tTrueRoslMaster, tNPairs);
    fTrueRosl->Fill(tTrueAndSimpleRosl[0][0], tTrueAndSimpleRosl[0][1], tTrueAndSimpleRosl[0][2]);
    fSimpleRosl->Fill(tTrueAndSimpleRosl[1][0], tTrueAndSimpleRosl[1][1], tTrueAndSimpleRosl[1][2]);
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTrueRoslParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;

  td2dVec tTrueRoslMaster{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0},
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int tNPairs = 0;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GoodV0())
    {
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];
        AddRoslContributionToMasterVector(tTrueRoslMaster, tNPairs, tV0, tParticle);
      }
    }
  }
  //---------------
  if(tNPairs>0)
  {
    td2dVec tTrueAndSimpleRosl = FinalizeTrueRoslMaster(tTrueRoslMaster, tNPairs);
    fTrueRosl->Fill(tTrueAndSimpleRosl[0][0], tTrueAndSimpleRosl[0][1], tTrueAndSimpleRosl[0][2]);
    fSimpleRosl->Fill(tTrueAndSimpleRosl[1][0], tTrueAndSimpleRosl[1][1], tTrueAndSimpleRosl[1][2]);
  }
}



//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildTrueRoslV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection)
{
  ThermV0Particle tV01, tV02;

  td2dVec tTrueRoslMaster{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0},
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int tNPairs = 0;

  for(unsigned int iV01=0; iV01<aV01Collection.size(); iV01++)
  {
    tV01 = aV01Collection[iV01];
    if(tV01.GoodV0())
    {
      for(unsigned int iV02=0; iV02<aV02Collection.size(); iV02++)
      {
        tV02 = aV02Collection[iV02];
        if(tV02.GoodV0()) AddRoslContributionToMasterVector(tTrueRoslMaster, tNPairs, tV01, tV02);
      }
    }
  }
  //---------------
  if(tNPairs>0)
  {
    td2dVec tTrueAndSimpleRosl = FinalizeTrueRoslMaster(tTrueRoslMaster, tNPairs);
    fTrueRosl->Fill(tTrueAndSimpleRosl[0][0], tTrueAndSimpleRosl[0][1], tTrueAndSimpleRosl[0][2]);
    fSimpleRosl->Fill(tTrueAndSimpleRosl[1][0], tTrueAndSimpleRosl[1][1], tTrueAndSimpleRosl[1][2]);
  }
}




//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildCorrelationFunctionsParticleParticle(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection)
{
  vector<ThermParticle> aParticleCollection1;
  vector<ThermParticle> aParticleCollection2;

  aParticleCollection1 =  aEvent.GetParticleCollection(fPartType1);
  if(!fBuildMixedEventNumerators) //-- No mixing, i.e. real pairs
  {
    aParticleCollection2 = aEvent.GetParticleCollection(fPartType2);
    FillCorrelationFunctionsNumOrDenParticleParticle(aParticleCollection1, aParticleCollection2, true);
    if(!fDrawRStarFromGaussian) BuildTrueRoslParticleParticle(aParticleCollection1, aParticleCollection2);
  }

  //-- Mixed events
  for(unsigned int iMixEv=0; iMixEv < aMixingEventsCollection.size(); iMixEv++)
  {
    aParticleCollection2 = aMixingEventsCollection[iMixEv].GetParticleCollection(fPartType2);
    if(!fBuildMixedEventNumerators) FillCorrelationFunctionsNumOrDenParticleParticle(aParticleCollection1, aParticleCollection2, false);
    else FillCorrelationFunctionsNumAndDenParticleParticle(aParticleCollection1, aParticleCollection2);
  }
}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::BuildCorrelationFunctionsParticleV0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;
  
  aV0Collection =  aEvent.GetV0ParticleCollection(fPartType1);
  if(!fBuildMixedEventNumerators) //-- No mixing, i.e. real pairs
  {
    aParticleCollection = aEvent.GetParticleCollection(fPartType2);
    FillCorrelationFunctionsNumOrDenParticleV0(aParticleCollection, aV0Collection, true);
    if(!fDrawRStarFromGaussian) BuildTrueRoslParticleV0(aParticleCollection, aV0Collection);
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
void ThermPairAnalysis::BuildCorrelationFunctionsV0V0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection)
{
  vector<ThermV0Particle> aV01Collection;
  vector<ThermV0Particle> aV02Collection;

 
  aV01Collection =  aEvent.GetV0ParticleCollection(fPartType1);
  if(!fBuildMixedEventNumerators) //-- No mixing, i.e. real pairs
  {
    aV02Collection = aEvent.GetV0ParticleCollection(fPartType2);
    FillCorrelationFunctionsNumOrDenV0V0(aV01Collection, aV02Collection, true);
    if(!fDrawRStarFromGaussian) BuildTrueRoslV0V0(aV01Collection, aV02Collection);
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

  if(fBuild3dHists) 
  {
    fPairSource3d->Write();
    fNum3d->Write();
    fDen3d->Write();
  }

  fPairSourceFull->Write();
  fNumFull->Write();
  fDenFull->Write();
  fCfFull = BuildFinalCf(fNumFull, fDenFull, TString::Format("CfFull%s", cAnalysisBaseTags[fAnalysisType]));
  fCfFull->Write();
  fNumFull_RotatePar2->Write();

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

  fPairKStarVsmT->Write();
  if(fBuild3dHists) fPairmT3d->Write();

  fPairSource3d_OSLinPRF->Write();
  fPairSource3d_OSLinPRFPrimaryOnly->Write();

  fPairDeltaT_inPRF->Write();
  fPairDeltaT_inPRFPrimaryOnly->Write();

  fTrueRosl->Write();
  fSimpleRosl->Write();

  if(fBuildPairSourcewmTInfo)
  {
    fPairSource3d_mT1vmT2vRinv->Write();
    fPairSource2d_PairmTvRinv->Write();
    fPairSource2d_mT1vRinv->Write();
    fPairSource2d_mT2vRinv->Write();
    fPairSource2d_mT1vR1PRF->Write();
    fPairSource2d_mT2vR2PRF->Write();
  }

  if(fBuildCfYlm) 
  {
    fCfYlm->Finish();
    fCfYlm->Write();
  }

}


//________________________________________________________________________________________________________________
void ThermPairAnalysis::ProcessEvent(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection)
{
  if(fBuildTransformMatrices) BuildAllTransformMatrices(aEvent, aMixingEventsCollection);

  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0 || fAnalysisType==kLamLam || fAnalysisType==kK0K0)
  {
    if(fBuildPairFractions) BuildPairFractionHistogramsV0V0(aEvent);
    if(fBuildCorrelationFunctions) BuildCorrelationFunctionsV0V0(aEvent, aMixingEventsCollection);
  }
  else if(fAnalysisType==kKchPKchP)
  {
    if(fBuildCorrelationFunctions) BuildCorrelationFunctionsParticleParticle(aEvent, aMixingEventsCollection);
  }
  else
  {
    if(fBuildPairFractions) BuildPairFractionHistogramsParticleV0(aEvent);
    if(fBuildCorrelationFunctions) BuildCorrelationFunctionsParticleV0(aEvent, aMixingEventsCollection);
  }
}





