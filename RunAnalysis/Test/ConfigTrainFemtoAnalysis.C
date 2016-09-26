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
  bool RunMC = true;

  //-----17/12/2015

  bool UseCustomNSigmaFiltersLambda = true;
  bool UseCustomNSigmaFiltersAntiLambda = true;
  bool UseCustomNSigmaFiltersK0Short = true;

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
  //------------------------------- Lambda-K0Short -----------------------------------------------
  myTrainAnalysisConstructor *analy_LamK0 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamK0, "LamK0",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamK0->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);

  AliFemtoBasicEventCut* tEvCut = analy_LamK0->CreateBasicEventCut();
  AliFemtoEventCutEstimators* tEvCutEst_0010 = analy_LamK0->CreateEventCutEstimators(0.,10.);
  AliFemtoEventCutEstimators* tEvCutEst_1030 = analy_LamK0->CreateEventCutEstimators(10.,30.);
  AliFemtoEventCutEstimators* tEvCutEst_3050 = analy_LamK0->CreateEventCutEstimators(30.,50.);

  AliFemtoV0TrackCut* tLamCut = analy_LamK0->CreateLambdaCut();
  AliFemtoV0TrackCut* tALamCut = analy_LamK0->CreateAntiLambdaCut();
  AliFemtoV0TrackCut* tK0Cut = analy_LamK0->CreateK0ShortCut();

/*
  AliFemtoV0TrackCutNSigmaFilter* tLamCut = analy_LamK0->CreateLambdaCut(UseCustomNSigmaFiltersLambda);
  AliFemtoV0TrackCutNSigmaFilter* tALamCut = analy_LamK0->CreateAntiLambdaCut(UseCustomNSigmaFiltersAntiLambda);
  AliFemtoV0TrackCutNSigmaFilter* tK0Cut = analy_LamK0->CreateK0ShortCut(UseCustomNSigmaFiltersK0Short);
*/
  AliFemtoV0PairCut* tV0PairCut = analy_LamK0->CreateV0PairCut(6.,0.,0.,6.);

    analy_LamK0->SetAnalysis(tEvCut,tLamCut,tK0Cut,tV0PairCut);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_LamK0_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamK0, "LamK0_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamK0_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_0010->SetAnalysis(tEvCutEst_0010,tLamCut,tK0Cut,tV0PairCut);

    myTrainAnalysisConstructor *analy_LamK0_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamK0, "LamK0_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamK0_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_1030->SetAnalysis(tEvCutEst_1030,tLamCut,tK0Cut,tV0PairCut);

    myTrainAnalysisConstructor *analy_LamK0_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kLamK0, "LamK0_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamK0_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_3050->SetAnalysis(tEvCutEst_3050,tLamCut,tK0Cut,tV0PairCut);


  //------------------------------- AntiLambda-K0Short -----------------------------------------------
  myTrainAnalysisConstructor *analy_ALamK0 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamK0, "ALamK0",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamK0->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_ALamK0->SetAnalysis(tEvCut,tALamCut,tK0Cut,tV0PairCut);

    //_______Centrality dependent______________________________________________________
    myTrainAnalysisConstructor *analy_ALamK0_0010 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamK0, "ALamK0_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_0010->SetAnalysis(tEvCutEst_0010,tALamCut,tK0Cut,tV0PairCut);

    myTrainAnalysisConstructor *analy_ALamK0_1030 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamK0, "ALamK0_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_1030->SetAnalysis(tEvCutEst_1030,tALamCut,tK0Cut,tV0PairCut);

    myTrainAnalysisConstructor *analy_ALamK0_3050 = new myTrainAnalysisConstructor(myTrainAnalysisConstructor::kALamK0, "ALamK0_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_3050->SetAnalysis(tEvCutEst_3050,tALamCut,tK0Cut,tV0PairCut);



  // Add the analyses to the manager
//  mgr->AddAnalysis(analy_LamK0);
    mgr->AddAnalysis(analy_LamK0_0010);
    mgr->AddAnalysis(analy_LamK0_1030);
    mgr->AddAnalysis(analy_LamK0_3050);
//  mgr->AddAnalysis(analy_ALamK0);
    mgr->AddAnalysis(analy_ALamK0_0010);
    mgr->AddAnalysis(analy_ALamK0_1030);
    mgr->AddAnalysis(analy_ALamK0_3050);

  return mgr;
}


