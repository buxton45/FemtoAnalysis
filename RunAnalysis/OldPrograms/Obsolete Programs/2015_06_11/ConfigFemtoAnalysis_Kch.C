//
//  ConfigFemtoAnalysis.cxx
//
//
//  Returns a pointer to the created AliFemtoManager


#if !defined(__CINT__) || defined(__MAKECINT__)
#include "AliFemtoManager.h"
#include "AliFemtoEventReaderESDChain.h"
#include "AliFemtoEventReaderAODChain.h"
#include "AliFemtoShareQualityTPCEntranceSepPairCut.h"
#include "AliFemtoQinvCorrFctn.h"
#include "AliFemtoShareQualityCorrFctn.h"
#include "AliFemtoTPCInnerCorrFctn.h"

#include <AliFemtoTrack.h>
#include "AliFemtoV0TrackCut.h"
#include <AliFemtoAvgSepCorrFctn.h>

#include "KStarCF.h"


#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoV0PairCut.h"
#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoV0TrackCut.h"

#include "myAnalysisConstructor.h"
#endif

AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  gROOT->LoadMacro("myAliFemtoV0TrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn.cxx+g");
  gROOT->LoadMacro("/home/Analysis/K0Lam/KStarCF.cxx+g");
  gROOT->LoadMacro("myAnalysisConstructor.cxx+g");

  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kTRUE);

  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
    //Point to the data source - the reader
    mgr->SetEventReader(rdr);


  //Setup the analyses
  //-------------------------------(Anti-)Lambda-K0Short-----------------------------------------------
  myAnalysisConstructor *analy_LamK0 = new myAnalysisConstructor("LamK0",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut1 = analy_LamK0->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut1 = analy_LamK0->CreateLambdaCut();
    myAliFemtoV0TrackCut *K0Cut1 = analy_LamK0->CreateK0ShortCut();
    AliFemtoV0PairCut *PairCut1 = analy_LamK0->CreateV0PairCut();
    myAliFemtoKStarCorrFctn *KStarCF1 = analy_LamK0->CreateKStarCorrFctn("LamK0KStarCF1",75,0.,0.4);
    analy_LamK0->SetAnalysis(EvCut1,LamCut1,K0Cut1,PairCut1,KStarCF1);

  myAnalysisConstructor *analy_ALamK0 = new myAnalysisConstructor("ALamK0",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut2 = analy_ALamK0->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut2 = analy_ALamK0->CreateAntiLambdaCut();
    myAliFemtoV0TrackCut *K0Cut2 = analy_ALamK0->CreateK0ShortCut();
    AliFemtoV0PairCut *PairCut2 = analy_ALamK0->CreateV0PairCut();
    myAliFemtoKStarCorrFctn *KStarCF2 = analy_ALamK0->CreateKStarCorrFctn("ALamK0KStarCF2",75,0.,0.4);
    analy_ALamK0->SetAnalysis(EvCut2,ALamCut2,K0Cut2,PairCut2,KStarCF2);

  //-------------------------------(Anti-)Lambda-Kch-----------------------------------------------
  myAnalysisConstructor *analy_LamKchP = new myAnalysisConstructor("LamKchP",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut_LamKchP = analy_LamKchP->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut_LamKchP = analy_LamKchP->CreateLambdaCut();
    AliFemtoESDTrackCut *KchPCut_LamKchP = analy_LamKchP->CreateKchCut(1);
    AliFemtoV0TrackPairCut *PairCut_LamKchP = analy_LamKchP->CreateV0TrackPairCut();
    myAliFemtoKStarCorrFctn *KStarCF_LamKchP = analy_LamKchP->CreateKStarCorrFctn("LamKchPKStarCF",75,0.,0.4);
    analy_LamKchP->SetAnalysis(EvCut_LamKchP,LamCut_LamKchP,KchPCut_LamKchP,PairCut_LamKchP,KStarCF_LamKchP);

  myAnalysisConstructor *analy_LamKchM = new myAnalysisConstructor("LamKchM",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut_LamKchM = analy_LamKchM->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut_LamKchM = analy_LamKchM->CreateLambdaCut();
    AliFemtoESDTrackCut *KchMCut_LamKchM = analy_LamKchM->CreateKchCut(-1);
    AliFemtoV0TrackPairCut *PairCut_LamKchM = analy_LamKchM->CreateV0TrackPairCut();
    myAliFemtoKStarCorrFctn *KStarCF_LamKchM = analy_LamKchM->CreateKStarCorrFctn("LamKchMKStarCF",75,0.,0.4);
    analy_LamKchM->SetAnalysis(EvCut_LamKchM,LamCut_LamKchM,KchMCut_LamKchM,PairCut_LamKchM,KStarCF_LamKchM);

  myAnalysisConstructor *analy_ALamKchP = new myAnalysisConstructor("ALamKchP",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut_ALamKchP = analy_ALamKchP->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut_ALamKchP = analy_ALamKchP->CreateAntiLambdaCut();
    AliFemtoESDTrackCut *KchPCut_ALamKchP = analy_ALamKchP->CreateKchCut(1);
    AliFemtoV0TrackPairCut *PairCut_ALamKchP = analy_ALamKchP->CreateV0TrackPairCut();
    myAliFemtoKStarCorrFctn *KStarCF_ALamKchP = analy_ALamKchP->CreateKStarCorrFctn("ALamKchPKStarCF",75,0.,0.4);
    analy_ALamKchP->SetAnalysis(EvCut_ALamKchP,ALamCut_ALamKchP,KchPCut_ALamKchP,PairCut_ALamKchP,KStarCF_ALamKchP);

  myAnalysisConstructor *analy_ALamKchM = new myAnalysisConstructor("ALamKchM",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut_ALamKchM = analy_ALamKchM->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut_ALamKchM = analy_ALamKchM->CreateAntiLambdaCut();
    AliFemtoESDTrackCut *KchMCut_ALamKchM = analy_ALamKchM->CreateKchCut(-1);
    AliFemtoV0TrackPairCut *PairCut_ALamKchM = analy_ALamKchM->CreateV0TrackPairCut();
    myAliFemtoKStarCorrFctn *KStarCF_ALamKchM = analy_ALamKchM->CreateKStarCorrFctn("ALamKchMKStarCF",75,0.,0.4);
    analy_ALamKchM->SetAnalysis(EvCut_ALamKchM,ALamCut_ALamKchM,KchMCut_ALamKchM,PairCut_ALamKchM,KStarCF_ALamKchM);

  // Add the analyses to the manager
  mgr->AddAnalysis(analy_LamKchP);
  mgr->AddAnalysis(analy_LamKchM);
  mgr->AddAnalysis(analy_ALamKchP);
  mgr->AddAnalysis(analy_ALamKchM);

  return mgr;
}


