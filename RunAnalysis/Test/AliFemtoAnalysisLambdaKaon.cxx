///
/// \file AliFemtoAnalysisLambdaKaon.cxx
///

#include "AliFemtoAnalysisLambdaKaon.h"
#include "TObjArray.h"
#include "AliESDtrack.h"
#ifdef __ROOT__
ClassImp(AliFemtoAnalysisLambdaKaon)
#endif

static const double PionMass = 0.13956995,
                    KchMass = 0.493677,
                    K0ShortMass = 0.497614,
                    ProtonMass = 0.938272013,
                    LambdaMass = 1.115683,
		    XiMass     = 1.32171;



AliFemtoAnalysisLambdaKaon::AliFemtoAnalysisLambdaKaon(AliFemtoAnalysisLambdaKaon::AnalysisType aAnalysisType, const char* name, 
                                                       unsigned int binsVertex, double minVertex, double maxVertex, 
                                                       unsigned int binsMult, double minMult, double maxMult, 
                                                       bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics) :

  AliFemtoVertexMultAnalysis(binsVertex,minVertex,maxVertex,binsMult,minMult,maxMult),
  fAnalysisType(aAnalysisType),

  fOutputName(name),
  fMultHist(NULL),
  fImplementAvgSepCuts(aImplementAvgSepCuts),
  fWritePairKinematics(aWritePairKinematics),
  fIsMCRun(aIsMCRun),
  fIsMBAnalysis(kFALSE),
  fBuildMultHist(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),

  fCollectionOfCfs(NULL),

  BasicEvCut(NULL),
  EvCutEst(NULL),

  XiCut1(NULL),
  XiCut2(NULL),

  V0Cut1(NULL),
  V0Cut2(NULL),
  
  TrackCut1(NULL),
  TrackCut2(NULL),
  
  V0PairCut(NULL),
  V0TrackPairCut(NULL),
  XiTrackPairCut(NULL),

  KStarCf(NULL),
  AvgSepCf(NULL),

  KStarModelCfs(NULL)
{
  SetParticleTypes(fAnalysisType);
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);
  SetEnablePairMonitors(fIsMCRun);
  SetMultHist(fAnalysisTags[aAnalysisType]);

  fMinCent = minMult/10.;
  fMaxCent = maxMult/10.;

  fCollectionOfCfs = new AliFemtoCorrFctnCollection;

  if(fWritePairKinematics) KStarCf = CreateCorrFctnKStar(fAnalysisTags[aAnalysisType],62,0.,0.31); //TNtuple is huge, and I don't need data out to 1 GeV
  else KStarCf = CreateCorrFctnKStar(fAnalysisTags[aAnalysisType],200,0.,1.0);

  AvgSepCf = CreateAvgSepCorrFctn(fAnalysisTags[aAnalysisType],200,0.,20.);

  KStarModelCfs = CreateModelCorrFctnKStarFull(fAnalysisTags[aAnalysisType],200,0.,1.0);

  if(fWritePairKinematics) fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
  else
  {
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)AvgSepCf);
  }

  if(fIsMCRun) fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarModelCfs);

}


AliFemtoAnalysisLambdaKaon::~AliFemtoAnalysisLambdaKaon() :
{
}


//____________________________
void AliFemtoAnalysisLambdaKaon::SetParticleTypes(AliFemtoAnalysisLambdaKaon::AnalysisType aAnType)
{
  switch(aAnType) {
  case AliFemtoAnalysisLambdaKaon::kLamK0:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 =AliFemtoAnalysisLambdaKaon:: kPDGK0;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamK0:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGK0;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamKchP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchP;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamKchP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchP;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamKchM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchM;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamKchM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchM;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamLam:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamALam:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamALam:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamPiP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGPiP;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamPiP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGPiP;
    break;

  case AliFemtoAnalysisLambdaKaon::kLamPiM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGLam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGPiM;
    break;

  case AliFemtoAnalysisLambdaKaon::kALamPiM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kV0Track;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGALam;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGPiM;
    break;

  case AliFemtoAnalysisLambdaKaon::kXiKchP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kXiTrack;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGXiC;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchP;
    break;

  case AliFemtoAnalysisLambdaKaon::kAXiKchP:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kXiTrack;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGAXiC;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchP;
    break;

  case AliFemtoAnalysisLambdaKaon::kXiKchM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kXiTrack;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGXiC;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchM;
    break;

  case AliFemtoAnalysisLambdaKaon::kAXiKchM:
    fGeneralAnalysisType = AliFemtoAnalysisLambdaKaon::kXiTrack;
    fParticlePDGType1 = AliFemtoAnalysisLambdaKaon::kPDGAXiC;
    fParticlePDGType2 = AliFemtoAnalysisLambdaKaon::kPDGKchM;
    break;

  default:
    cerr << "E-AliFemtoAnalysisLambdaKaon::SetParticleTypes: Invalid AnalysisType"
            "selection '" << aAnType << endl;
  }

  switch(fGeneralAnalysisType) {
  case AliFemtoAnalysisLambdaKaon::kV0V0:
    fGeneralParticleType1 = AliFemtoAnalysisLambdaKaon::kV0;
    fGeneralParticleType2 = AliFemtoAnalysisLambdaKaon::kV0;
    break;

  case AliFemtoAnalysisLambdaKaon::kV0Track:
    fGeneralParticleType1 = AliFemtoAnalysisLambdaKaon::kV0;
    fGeneralParticleType2 = AliFemtoAnalysisLambdaKaon::kTrack;
    break;

  case AliFemtoAnalysisLambdaKaon::kXiTrack:
    fGeneralParticleType1 = AliFemtoAnalysisLambdaKaon::kCascade;
    fGeneralParticleType2 = AliFemtoAnalysisLambdaKaon::kTrack;
    break;

  default:
    cerr << "E-AliFemtoAnalysisLambdaKaon::SetParticleTypes" << endl;
  }

}


void AliFemtoAnalysisLambdaKaon::AddCustomV0SelectionFilters(ParticlePDGType aV0Type, AliFemtoV0TrackCutNSigmaFilter* aCut)
{
  switch(aV0Type) {
  case AliFemtoAnalysisLambdaKaon::kPDGLam:
    //--Proton(+) daughter selection filter
    aCut->CreateCustomProtonNSigmaFilter();
    aCut->AddProtonTPCNSigmaCut(0.,0.8,3.);
    aCut->AddProtonTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
    aCut->AddProtonTPCNSigmaCut(0.8,1000.,3.);

    //--Pion(-) daughter selection filter
    //the standard cuts in AliFemtoV0TrackCut
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,1000.,3.);

/*
    //RequireTOFPion
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,0.5,3.);
    aCut->AddPionTPCAndTOFNSigmaCut(0.5,1000.,3.,3.);
    aCut->AddPionTPCNSigmaCut(0.5,1000.,3.);
*/
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGALam:
    //--(Anti)Proton(-) daughter selection filter
    //for now, the custom filters will match the standard cuts in AliFemtoV0TrackCut
    //these also match my personal (proton) cuts in myAliFemtoV0TrackCut
    aCut->CreateCustomProtonNSigmaFilter();
    aCut->AddProtonTPCNSigmaCut(0.,0.8,3.);
    aCut->AddProtonTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
    aCut->AddProtonTPCNSigmaCut(0.8,1000.,3.);

    //the standard cuts in AliFemtoV0TrackCut
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,1000.,3.);

/*
    //RequireTOFPion
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,0.5,3.);
    aCut->AddPionTPCAndTOFNSigmaCut(0.5,1000.,3.,3.);
    aCut->AddPionTPCNSigmaCut(0.5,1000.,3.);
*/
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGK0:
    //--Pion(+) daughter selection filter
    //the standard cuts in AliFemtoV0TrackCut
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,1000.,3.);

/*
    //RequireTOFPion
    aCut->CreateCustomPionNSigmaFilter();
    aCut->AddPionTPCNSigmaCut(0.,0.5,3.);
    aCut->AddPionTPCAndTOFNSigmaCut(0.5,1000.,3.,3.);
    aCut->AddPionTPCNSigmaCut(0.5,1000.,3.);
*/
    break;

  default:
    cerr << "E-AliFemtoAnalysisLambdaKaon::AddCustomV0SelectionFilters" << endl;
  }
}


void AliFemtoAnalysisLambdaKaon::AddCustomV0RejectionFilters(ParticlePDGType aV0Type, AliFemtoV0TrackCutNSigmaFilter* aCut)
{
  switch(aV0Type) {
  case AliFemtoAnalysisLambdaKaon::kPDGLam:
    aCut->CreateCustomV0Rejection(AliFemtoV0TrackCut::kK0s);
    aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                         0.,0.8,3.,  //positive daughter
                                         0.,0.8,3.); //negative daughter
    aCut->AddTPCAndTOFNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                               0.8,1000.,3.,3.,  //positive daughter
                                               0.8,1000.,3.,3.); //negative daughter
    aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                         0.8,1000.,3.,  //positive daughter
                                         0.8,1000.,3.); //negative daughter
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGALam:
    aCut->CreateCustomV0Rejection(AliFemtoV0TrackCut::kK0s);
    aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                         0.,0.8,3.,  //positive daughter
                                         0.,0.8,3.); //negative daughter
    aCut->AddTPCAndTOFNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                               0.8,1000.,3.,3.,  //positive daughter
                                               0.8,1000.,3.,3.); //negative daughter
    aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                         0.8,1000.,3.,  //positive daughter
                                         0.8,1000.,3.); //negative daughter
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGK0:
    //Lambda rejection
    aCut->CreateCustomV0Rejection(AliFemtoV0TrackCut::kLambda);
      //Positive daughter (Proton)
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,1,0.,0.8,3.);
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,1,0.8,1000.,3.);
      //Negative daughter (Pion)
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,-1,0.,0.5,3.);
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,-1,0.5,1000.,3.);

    //AntiLambda rejection
    aCut->CreateCustomV0Rejection(AliFemtoV0TrackCut::kAntiLambda);
      //Positive daughter (Pion)
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,1,0.,0.5,3.);
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,1,0.5,1000.,3.);
      //Negative daughter (AntiProton)
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,-1,0.,0.8,3.);
      aCut->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,-1,0.8,1000.,3.);
    break;

  default:
    cerr << "E-AliFemtoAnalysisLambdaKaon::AddCustomV0SelectionFilters" << endl;
  }
}


AliFemtoAnalysisLambdaKaon::AliFemtoV0TrackCutNSigmaFilter* CreateV0Cut(V0CutParams &aCutParams)
{
  AliFemtoV0TrackCutNSigmaFilter* tV0Cut = new AliFemtoV0TrackCutNSigmaFilter();

  tV0Cut->SetParticleType(aCutParams.v0Type);
  tV0Cut->SetMass(aCutParams.mass);
  tV0Cut->SetInvariantMassLambda(aCutParams.minInvariantMass,aCutParams.maxInvariantMass);
  tV0Cut->SetLooseInvMassCut(aCutParams.useLooseInvMassCut, aCutParams.minLooseInvMassCut, aCutParams.maxLooseInvMassCut);

  tV0Cut->SetEta(aCutParams.eta);
  tV0Cut->SetPt(aCutParams.minPt, aCutParams.maxPt);
  tV0Cut->SetOnFlyStatus(aCutParams.onFlyStatus);
  tV0Cut->SetMaxDcaV0(aCutParams.maxDcaV0);
  tV0Cut->SetMinCosPointingAngle(aCutParams.minCosPointingAngle);
  tV0Cut->SetMaxV0DecayLength(aCutParams.maxV0DecayLength);
  //-----
  tV0Cut->SetEtaDaughters(aCutParams.etaDaughters);
  tV0Cut->SetPtPosDaughter(aCutParams.minPtPosDaughter,aCutParams.maxPtPosDaughter);
  tV0Cut->SetPtNegDaughter(aCutParams.minPtNegDaughter,aCutParams.maxPtNegDaughter);
  tV0Cut->SetTPCnclsDaughters(aCutParams.minTPCnclsDaughters);
  //tV0Cut->SetNdofDaughters(4.0); //4.0
  tV0Cut->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
  tV0Cut->SetMaxDcaV0Daughters(aCutParams.maxDcaV0Daughters);
  tV0Cut->SetMinDaughtersToPrimVertex(aCutParams.minPosDaughterToPrimVertex,aCutParams.minNegDaughterToPrimVertex);

  if(aCutParams.useCustomFilter) AddCustomV0SelectionFilters(aCutParams.particlePDGType,tV0Cut);

  //Misidentification cuts -----*****-----*****-----*****-----*****-----*****-----*****
  tV0Cut->SetRemoveMisidentified(aCutParams.removeMisID);
  tV0Cut->SetUseSimpleMisIDCut(aCutParams.useSimpleMisID);
  if(!aCutParams.useSimpleMisID && aCutParams.useCustomMisID) AddCustomV0RejectionFilters(aCutParams.particlePDGType,tV0Cut);
  tV0Cut->SetBuildMisIDHistograms(aCutParams.buildMisIDHistograms);

  TString tTitle, tName;
  switch(aCutParams.particlePDGType) {
  case AliFemtoAnalysisLambdaKaon::kPDGLam:
    tName = TString("LambdaPurityAid");
    tTitle = TString("LambdaMinvBeforeFinalCut");

    tV0Cut->SetInvMassReject(AliFemtoV0TrackCut::kK0s, aCutParams.minInvMassReject,aCutParams.maxInvMassReject, aCutParams.removeMisID);

    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kLambda,100,LambdaMass-0.035,LambdaMass+0.035);
    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGALam:
    tName = TString("AntiLambdaPurityAid");
    tTitle = TString("AntiLambdaMinvBeforeFinalCut");

    tV0Cut->SetInvMassReject(AliFemtoV0TrackCut::kK0s, aCutParams.minInvMassReject,aCutParams.maxInvMassReject, aCutParams.removeMisID);

    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kAntiLambda,100,LambdaMass-0.035,LambdaMass+0.035);
    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGK0:
    tName = TString("K0ShortPurityAid");
    tTitle = TString("K0ShortMinvBeforeFinalCut");

    tV0Cut->SetInvMassReject(AliFemtoV0TrackCut::kLambda, aCutParams.minInvMassReject,aCutParams.maxInvMassReject, aCutParams.removeMisID);
    tV0Cut->SetInvMassReject(AliFemtoV0TrackCut::kAntiLambda, aCutParams.minInvMassReject,aCutParams.maxInvMassReject, aCutParams.removeMisID); 
 
    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kLambda,100,LambdaMass-0.035,LambdaMass+0.035);
    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kAntiLambda,100,LambdaMass-0.035,LambdaMass+0.035);
    tV0Cut->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);
    break;

  default:
    cerr << "E-AliFemtoAnalysisLambdaKaon::CreateV0Cut" << endl;
  }

  //-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****

  tV0Cut->SetMinvPurityAidHistoV0(tName,tTitle,aCutParams.nBinsPurity,aCutParams.minPurityMass,aCutParams.maxPurityMass);

  return tV0Cut;
}


AliFemtoAnalysisLambdaKaon::AnalysisParams 
AliFemtoAnalysisLambdaKaon::DefaultAnalysisParams()
{
  AliFemtoAnalysisLambdaKaon::AnalysisParams tReturnParams;

  tReturnParams.nBinsVertex = 10;
  tReturnParams.minVertex = -10.;
  tReturnParams.maxVertex = 10.;

  tReturnParams.nBinsMult = 20;
  tReturnParams.minMult = 0.;
  tReturnParams.maxMult = 1000.;

  tReturnParams.analysisType = AliFemtoAnalysisLambdaKaon::kLamK0;
  tReturnParams.generalAnalysisType = AliFemtoAnalysisLambdaKaon::kV0V0;

  tReturnParams.nEventsToMix = 5;
  tReturnParams.minCollectionSize = 1;

  tReturnParams.verbose = true;
  tReturnParams.enablePairMonitors = false;
  tReturnParams.isMCRun = false;

  return tReturnParams;
}


AliFemtoAnalysisLambdaKaon::EventCutParams 
AliFemtoAnalysisLambdaKaon::DefaultEventCutParams()
{
  AliFemtoAnalysisLambdaKaon::EventCutParams tReturnParams;

  tReturnParams.minCentrality = 0.;
  tReturnParams.maxCentrality = 100.;

  tReturnParams.minMult = 0.;
  tReturnParams.maxMult = 100000.;

  tReturnParams.minVertexZ = -8.;
  tReturnParams.maxVertexZ = 8.;

  return tReturnParams;
}



AliFemtoAnalysisLambdaKaon::V0CutParams 
AliFemtoAnalysisLambdaKaon::DefaultLambdaCutParams()
{
  AliFemtoAnalysisLambdaKaon::V0CutParams tReturnParams;

  tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGLam;
  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kV0;

  tReturnParams.v0Type = 0;

  tReturnParams.mass = LambdaMass;
  tReturnParams.minInvariantMass = LambdaMass-0.0038;
  tReturnParams.maxInvariantMass = LambdaMass+0.0038;

  tReturnParams.useLooseInvMassCut = true;
  tReturnParams.minLooseInvMass = LambdaMass-0.035;
  tReturnParams.maxLooseInvMass = LambdaMass+0.035;

  tReturnParams.nBinsPurity = 100;
  tReturnParams.minPurityMass = LambdaMass-0.035;
  tReturnParams.maxPurityMass = LambdaMass+0.035;

  tReturnParams.useCustomFilter = false;

  tReturnParams.removeMisID = true;
  tReturnParams.minInvMassReject = K0ShortMass-0.003677;
  tReturnParams.maxInvMassReject = K0ShortMass+0.003677;

  tReturnParams.useSimpleMisID = true;
  tReturnParams.buildMisIDHistograms = true;
  tReturnParams.useCustomMisID = false;

  tReturnParams.eta = 0.8;
  tReturnParams.minPt = 0.4;
  tReturnParams.maxPt = 100.;
  tReturnParams.onFlyStatus = false;
  tReturnParams.maxDcaV0 = 0.5;
  tReturnParams.minCosPointingAngle = 0.9993;
  tReturnParams.maxV0DecayLength = 60.;

  tReturnParams.etaDaughters = 0.8;
  tReturnParams.minPtPosDaughter = 0.5;
  tReturnParams.maxPtPosDaughter = 99.;
  tReturnParams.minPtNegDaughter = 0.16;
  tReturnParams.maxPtNegDaughter = 99.;
  tReturnParams.minTPCnclsDaughters = 80;
  tReturnParams.maxDcaV0Daughters = 0.4;
  tReturnParams.minPosDaughterToPrimVertex = 0.1;
  tReturnParams.minNegDaughterToPrimVertex = 0.3;

  return tReturnParams;
}

AliFemtoAnalysisLambdaKaon::V0CutParams 
AliFemtoAnalysisLambdaKaon::DefaultAntiLambdaCutParams()
{
  AliFemtoAnalysisLambdaKaon::V0CutParams tReturnParams;

  tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGALam;
  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kV0;

  tReturnParams.v0Type = 1;

  tReturnParams.mass = LambdaMass;
  tReturnParams.minInvariantMass = LambdaMass-0.0038;
  tReturnParams.maxInvariantMass = LambdaMass+0.0038;

  tReturnParams.useLooseInvMassCut = true;
  tReturnParams.minLooseInvMass = LambdaMass-0.035;
  tReturnParams.maxLooseInvMass = LambdaMass+0.035;

  tReturnParams.nBinsPurity = 100;
  tReturnParams.minPurityMass = LambdaMass-0.035;
  tReturnParams.maxPurityMass = LambdaMass+0.035;

  tReturnParams.useCustomFilter = false;

  tReturnParams.removeMisID = true;
  tReturnParams.minInvMassReject = K0ShortMass-0.003677;
  tReturnParams.maxInvMassReject = K0ShortMass+0.003677;

  tReturnParams.useSimpleMisID = true;
  tReturnParams.buildMisIDHistograms = true;

  tReturnParams.eta = 0.8;
  tReturnParams.minPt = 0.4;
  tReturnParams.maxPt = 100.;
  tReturnParams.onFlyStatus = false;
  tReturnParams.maxDcaV0 = 0.5;
  tReturnParams.minCosPointingAngle = 0.9993;
  tReturnParams.maxV0DecayLength = 60.;

  tReturnParams.etaDaughters = 0.8;
  tReturnParams.minPtPosDaughter = 0.16;
  tReturnParams.maxPtPosDaughter = 99.;
  tReturnParams.minPtNegDaughter = 0.3;
  tReturnParams.maxPtNegDaughter = 99.;
  tReturnParams.minTPCnclsDaughters = 80;
  tReturnParams.maxDcaV0Daughters = 0.4;
  tReturnParams.minPosDaughterToPrimVertex = 0.3;
  tReturnParams.minNegDaughterToPrimVertex = 0.1;

  return tReturnParams;
}

AliFemtoAnalysisLambdaKaon::V0CutParams 
AliFemtoAnalysisLambdaKaon::DefaultK0ShortCutParams()
{
  AliFemtoAnalysisLambdaKaon::V0CutParams tReturnParams;

  tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGK0;
  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kV0;

  tReturnParams.v0Type = 2;

  tReturnParams.mass = K0ShortMass;
  tReturnParams.minInvariantMass = K0ShortMass-0.013677;
  tReturnParams.maxInvariantMass = K0ShortMass+0.020323;

  tReturnParams.useLooseInvMassCut = true;
  tReturnParams.minLooseInvMass = K0ShortMass-0.070;
  tReturnParams.maxLooseInvMass = K0ShortMass+0.070;

  tReturnParams.nBinsPurity = 100;
  tReturnParams.minPurityMass = K0ShortMass-0.070;
  tReturnParams.maxPurityMass = K0ShortMass+0.070;

  tReturnParams.useCustomFilter = false;

  tReturnParams.removeMisID = true;
  tReturnParams.minInvMassReject = LambdaMass-0.005683;
  tReturnParams.maxInvMassReject = LambdaMass+0.005683;

  tReturnParams.useSimpleMisID = true;
  tReturnParams.buildMisIDHistograms = true;

  tReturnParams.eta = 0.8;
  tReturnParams.minPt = 0.2;
  tReturnParams.maxPt = 100.;
  tReturnParams.onFlyStatus = false;
  tReturnParams.maxDcaV0 = 0.3;
  tReturnParams.minCosPointingAngle = 0.9993;
  tReturnParams.maxV0DecayLength = 30.;

  tReturnParams.etaDaughters = 0.8;
  tReturnParams.minPtPosDaughter = 0.15;
  tReturnParams.maxPtPosDaughter = 99.;
  tReturnParams.minPtNegDaughter = 0.15;
  tReturnParams.maxPtNegDaughter = 99.;
  tReturnParams.minTPCnclsDaughters = 80;
  tReturnParams.maxDcaV0Daughters = 0.3;
  tReturnParams.minPosDaughterToPrimVertex = 0.3;
  tReturnParams.minNegDaughterToPrimVertex = 0.3;

  return tReturnParams;
}




AliFemtoAnalysisLambdaKaon::ESDCutParams 
AliFemtoAnalysisLambdaKaon::DefaultKchCutParams(int aCharge)
{
  AliFemtoAnalysisLambdaKaon::ESDCutParams tReturnParams;

  if(aCharge>0) tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGKchP;
  else tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGKchM;

  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kTrack;

  tReturnParams.minPidProbPion = 0.;
  tReturnParams.maxPidProbPion = 0.1;
  tReturnParams.minPidProbMuon = 0.;
  tReturnParams.maxPidProbMuon = 0.8;
  tReturnParams.minPidProbKaon = 0.2;
  tReturnParams.maxPidProbKaon = 1.001;
  tReturnParams.minPidProbProton = 0.;
  tReturnParams.maxPidProbProton = 0.1;
  tReturnParams.mostProbable = 3;    //this uses P().Mag() as first argument to IsKaonNSigma()
//  tReturnParams.mostProbable = 11; //this looks for Kaons, and uses Pt() as first argument to IsKaonNSigma
  tReturnParams.charge = aCharge;
  tReturnParams.mass = KchMass;

  tReturnParams.minPt = 0.14;
  tReturnParams.maxPt = 1.5;
  tReturnParams.eta = 0.8;
  tReturnParams.minTPCncls = 80;

  tReturnParams.removeKinks = true;
  tReturnParams.setLabel = false;
  tReturnParams.maxITSChiNdof = 3.0;
  tReturnParams.maxTPCChiNdof = 4.0;
  tReturnParams.maxSigmaToVertex = 3.0;
  tReturnParams.maxImpactXY = 2.4;
  tReturnParams.maxImpactZ = 3.0;

  tReturnParams.useCustomFilter = false;
  tReturnParams.useElectronRejection = true;
  tReturnParams.usePionRejection = true;

  return tReturnParams;
}




AliFemtoAnalysisLambdaKaon::XiCutParams 
AliFemtoAnalysisLambdaKaon::DefaultXiCutParams()
{
  AliFemtoAnalysisLambdaKaon::XiCutParams tReturnParams;

  tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGXiC;
  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kCascade;

  tReturnParams.charge = -1;
  tReturnParams.xiType = 0;
  tReturnParams.minPt = 0.8;
  tReturnParams.maxPt = 100.;
  tReturnParams.eta = 0.8;
  tReturnParams.mass = XiMass;
  tReturnParams.minInvariantMass = XiMass-0.003;
  tReturnParams.maxInvariantMass = XiMass+0.003;

  tReturnParams.maxDecayLengthXi = 100.;
  tReturnParams.minCosPointingAngleXi = 0.9992;
  tReturnParams.maxDcaXi = 100.;
  tReturnParams.maxDcaXiDaughters = 0.3;

  tReturnParams.minDcaXiBac = 0.03;
  tReturnParams.etaBac = 0.8;
  tReturnParams.minTPCnclsBac = 70;
  tReturnParams.minPtBac = 0.;
  tReturnParams.maxPtBac = 100.;

  tReturnParams.v0Type = 0;
  tReturnParams.minDcaV0 = 0.1;
  tReturnParams.minInvMassV0 = LambdaMass-0.005;
  tReturnParams.maxInvMassV0 = LambdaMass+0.005;
  tReturnParams.minCosPointingAngleV0 = 0.998;
  tReturnParams.etaV0 = 0.8;
  tReturnParams.minPtV0 = 0.;
  tReturnParams.maxPtV0 = 100.;
  tReturnParams.onFlyStatusV0 = false;
  tReturnParams.maxV0DecayLength = 100.;
  tReturnParams.minV0DaughtersToPrimVertex = 0.1;
  tReturnParams.maxV0DaughtersToPrimVertex = 0.1;
  tReturnParams.maxDcaV0Daughters = 0.8;
  tReturnParams.etaV0Daughters = 0.8;
  tReturnParams.minPtPosV0Daughter = 0.;
  tReturnParams.maxPtPosV0Daughter = 99.;
  tReturnParams.minPtNegV0Daughter = 0.;
  tReturnParams.maxPtNegV0Daughter = 99.;

  tReturnParams.minTPCnclsV0Daughters = 70;

  tReturnParams.setPurityAidXi = true;
  tReturnParams.setPurityAidV0 = true;

  return tReturnParams;
}

AliFemtoAnalysisLambdaKaon::XiCutParams 
AliFemtoAnalysisLambdaKaon::DefaultAXiCutParams()
{
  AliFemtoAnalysisLambdaKaon::XiCutParams tReturnParams;

  tReturnParams.particlePDGType = AliFemtoAnalysisLambdaKaon::kPDGAXiC;
  tReturnParams.generalParticleType = AliFemtoAnalysisLambdaKaon::kCascade;

  tReturnParams.charge = 1;
  tReturnParams.xiType = 1;
  tReturnParams.minPt = 0.8;
  tReturnParams.maxPt = 100.;
  tReturnParams.eta = 0.8;
  tReturnParams.mass = XiMass;
  tReturnParams.minInvariantMass = XiMass-0.003;
  tReturnParams.maxInvariantMass = XiMass+0.003;

  tReturnParams.maxDecayLengthXi = 100.;
  tReturnParams.minCosPointingAngleXi = 0.9992;
  tReturnParams.maxDcaXi = 100.;
  tReturnParams.maxDcaXiDaughters = 0.3;

  tReturnParams.minDcaXiBac = 0.03;
  tReturnParams.etaBac = 0.8;
  tReturnParams.minTPCnclsBac = 70;
  tReturnParams.minPtBac = 0.;
  tReturnParams.maxPtBac = 100.;

  tReturnParams.v0Type = 1;
  tReturnParams.minDcaV0 = 0.1;
  tReturnParams.minInvMassV0 = LambdaMass-0.005;
  tReturnParams.maxInvMassV0 = LambdaMass+0.005;
  tReturnParams.minCosPointingAngleV0 = 0.998;
  tReturnParams.etaV0 = 0.8;
  tReturnParams.minPtV0 = 0.;
  tReturnParams.maxPtV0 = 100.;
  tReturnParams.onFlyStatusV0 = true;
  tReturnParams.maxV0DecayLength = 100.;
  tReturnParams.minV0DaughtersToPrimVertex = 0.1;
  tReturnParams.maxV0DaughtersToPrimVertex = 0.1;
  tReturnParams.maxDcaV0Daughters = 0.8;
  tReturnParams.etaV0Daughters = 0.8;
  tReturnParams.minPtPosV0Daughter = 0.;
  tReturnParams.maxPtPosV0Daughter = 99.;
  tReturnParams.minPtNegV0Daughter = 0.;
  tReturnParams.maxPtNegV0Daughter = 99.;

  tReturnParams.minTPCnclsV0Daughters = 70;

  tReturnParams.setPurityAidXi = true;
  tReturnParams.setPurityAidV0 = true;

  return tReturnParams;
}



AliFemtoAnalysisLambdaKaon::PairCutParams 
AliFemtoAnalysisLambdaKaon::DefaultPairParams()
{
  AliFemtoAnalysisLambdaKaon::PairCutParams tReturnParams;

  tReturnParams.removeSameLabel = ;
  tReturnParams.shareQualityMax = ;
  tReturnParams.shareFractionMax = ;
  tReturnParams.tpcOnly = ;

  tReturnParams.tpcExitSepMinimum = ;
  tReturnParams.tpcEntranceSepMinimum = ;

  tReturnParams.minAvgSepPosPos = ;
  tReturnParams.minAvgSepPosNeg = ;
  tReturnParams.minAvgSepNegPos = ;
  tReturnParams.minAvgSepNegNeg = ;

  tReturnParams.minAvgSepTrackPos = ;
  tReturnParams.minAvgSepTrackNeg = ;

  return tReturnParams;
}





