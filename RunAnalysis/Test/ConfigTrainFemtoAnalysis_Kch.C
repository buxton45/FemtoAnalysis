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

#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoV0PairCut.h"

#include "AliFemtoV0TrackCutNSigmaFilter.h"
#include "AliFemtoESDTrackCutNSigmaFilter.h"

#include "AliFemtoCorrFctnKStar.h"
#include "AliFemtoAvgSepCorrFctn.h"
/*
#include "myAliFemtoSepCorrFctns.h"
#include "myAliFemtoAvgSepCorrFctnCowboysAndSailors.h"
#include "myAliFemtoKStarCorrFctn2D.h"
*/

//-----04/02/2016
#include "AliFemtoModelWeightGeneratorBasicLednicky.h"
#include "AliFemtoModelCorrFctnKStarFull.h"

#include "myTrainAnalysisConstructor.h"
#endif

AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  gROOT->LoadMacro("myTrainAnalysisConstructor.cxx+g");

  bool ImplementAvgSepCuts = true;
  bool ImplementVertexCorrection = true;
  bool RunMC = false;

  //-----17/12/2015

  bool UseCustomNSigmaFiltersLambda = true;
  bool UseCustomNSigmaFiltersAntiLambda = true;

  bool UseCustomNSigmaFilterKchP = true;
  bool UseCustomNSigmaFilterKchM = true;

  //-----25/02/2016
  bool RemoveMisidentifiedMCParticles = true;

  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    //rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kFALSE);
    rdr->SetPrimaryVertexCorrectionTPCPoints(ImplementVertexCorrection);
    rdr->SetReadMC(RunMC);

  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
    //Point to the data source - the reader
    mgr->SetEventReader(rdr);


  //Setup the analyses
  //------------------------------- Lambda-KchP -----------------------------------------------
  myTrainAnalysisConstructor *analy_LamKchP = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchP, "LamKchP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamKchP->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);

  AliFemtoBasicEventCut* tEvCut = analy_LamKchP->CreateBasicEventCut();
  AliFemtoEventCutEstimators* tEvCutEst_0010 = analy_LamKchP->CreateEventCutEstimators(0.,10.);
  AliFemtoEventCutEstimators* tEvCutEst_1030 = analy_LamKchP->CreateEventCutEstimators(10.,30.);
  AliFemtoEventCutEstimators* tEvCutEst_3050 = analy_LamKchP->CreateEventCutEstimators(30.,50.);

  AliFemtoV0TrackCut* tLamCut = analy_LamKchP->CreateLambdaCut();
  AliFemtoV0TrackCut* tALamCut = analy_LamKchP->CreateAntiLambdaCut();

/*
  AliFemtoV0TrackCutNSigmaFilter* tLamCut = analy_LamKchP->CreateLambdaCut(UseCustomNSigmaFiltersLambda);
  AliFemtoV0TrackCutNSigmaFilter* tALamCut = analy_LamKchP->CreateAntiLambdaCut(UseCustomNSigmaFiltersAntiLambda);
*/

  AliFemtoESDTrackCutNSigmaFilter* tKchPCut = analy_LamKchP->CreateKchCut(1,UseCustomNSigmaFilterKchP);
  AliFemtoESDTrackCutNSigmaFilter* tKchMCut = analy_LamKchP->CreateKchCut(-1,UseCustomNSigmaFilterKchP);

  AliFemtoV0TrackPairCut* tV0TrackPairCutKchP = analy_LamKchP->CreateV0TrackPairCut(8.,0.);
  AliFemtoV0TrackPairCut* tV0TrackPairCutKchM = analy_LamKchP->CreateV0TrackPairCut(0.,8.);

    analy_LamKchP->SetAnalysis(tEvCut,tLamCut,tKchPCut,tV0TrackPairCutKchP);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_LamKchP_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchP, "LamKchP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamKchP_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchP_0010->SetAnalysis(tEvCutEst_0010,tLamCut,tKchPCut,tV0TrackPairCutKchP);

    myTrainAnalysisConstructor *analy_LamKchP_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchP, "LamKchP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamKchP_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchP_1030->SetAnalysis(tEvCutEst_1030,tLamCut,tKchPCut,tV0TrackPairCutKchP);

    myTrainAnalysisConstructor *analy_LamKchP_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchP, "LamKchP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamKchP_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchP_3050->SetAnalysis(tEvCutEst_3050,tLamCut,tKchPCut,tV0TrackPairCutKchP);



  //------------------------------- Lambda-KchM -----------------------------------------------
  myTrainAnalysisConstructor *analy_LamKchM = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchM, "LamKchM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamKchM->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_LamKchM->SetAnalysis(tEvCut,tLamCut,tKchMCut,tV0TrackPairCutKchM);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_LamKchM_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchM, "LamKchM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamKchM_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchM_0010->SetAnalysis(tEvCutEst_0010,tLamCut,tKchMCut,tV0TrackPairCutKchM);

    myTrainAnalysisConstructor *analy_LamKchM_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchM, "LamKchM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamKchM_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchM_1030->SetAnalysis(tEvCutEst_1030,tLamCut,tKchMCut,tV0TrackPairCutKchM);

    myTrainAnalysisConstructor *analy_LamKchM_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamKchM, "LamKchM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamKchM_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamKchM_3050->SetAnalysis(tEvCutEst_3050,tLamCut,tKchMCut,tV0TrackPairCutKchM);


  //------------------------------- AntiLambda-KchP -----------------------------------------------
  myTrainAnalysisConstructor *analy_ALamKchP = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchP, "ALamKchP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamKchP->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_ALamKchP->SetAnalysis(tEvCut,tALamCut,tKchPCut,tV0TrackPairCutKchP);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_ALamKchP_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchP, "ALamKchP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_ALamKchP_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchP_0010->SetAnalysis(tEvCutEst_0010,tALamCut,tKchPCut,tV0TrackPairCutKchP);

    myTrainAnalysisConstructor *analy_ALamKchP_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchP, "ALamKchP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_ALamKchP_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchP_1030->SetAnalysis(tEvCutEst_1030,tALamCut,tKchPCut,tV0TrackPairCutKchP);

    myTrainAnalysisConstructor *analy_ALamKchP_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchP, "ALamKchP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_ALamKchP_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchP_3050->SetAnalysis(tEvCutEst_3050,tALamCut,tKchPCut,tV0TrackPairCutKchP);

  //------------------------------- AntiLambda-KchM -----------------------------------------------
  myTrainAnalysisConstructor *analy_ALamKchM = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchM, "ALamKchM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamKchM->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_ALamKchM->SetAnalysis(tEvCut,tALamCut,tKchMCut,tV0TrackPairCutKchM);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_ALamKchM_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchM, "ALamKchM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_ALamKchM_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchM_0010->SetAnalysis(tEvCutEst_0010,tALamCut,tKchMCut,tV0TrackPairCutKchM);

    myTrainAnalysisConstructor *analy_ALamKchM_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchM, "ALamKchM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_ALamKchM_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchM_1030->SetAnalysis(tEvCutEst_1030,tALamCut,tKchMCut,tV0TrackPairCutKchM);

    myTrainAnalysisConstructor *analy_ALamKchM_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamKchM, "ALamKchM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_ALamKchM_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamKchM_3050->SetAnalysis(tEvCutEst_3050,tALamCut,tKchMCut,tV0TrackPairCutKchM);



  // Add the analyses to the manager
//  mgr->AddAnalysis(analy_LamKchP);
      mgr->AddAnalysis(analy_LamKchP_0010);
      //mgr->AddAnalysis(analy_LamKchP_1030);
      //mgr->AddAnalysis(analy_LamKchP_3050);
//  mgr->AddAnalysis(analy_LamKchM);
      mgr->AddAnalysis(analy_LamKchM_0010);
      //mgr->AddAnalysis(analy_LamKchM_1030);
      //mgr->AddAnalysis(analy_LamKchM_3050);
//  mgr->AddAnalysis(analy_ALamKchP);
      mgr->AddAnalysis(analy_ALamKchP_0010);
      //mgr->AddAnalysis(analy_ALamKchP_1030);
      //mgr->AddAnalysis(analy_ALamKchP_3050);
//  mgr->AddAnalysis(analy_ALamKchM);
      mgr->AddAnalysis(analy_ALamKchM_0010);
      //mgr->AddAnalysis(analy_ALamKchM_1030);
      //mgr->AddAnalysis(analy_ALamKchM_3050);

  return mgr;
}


