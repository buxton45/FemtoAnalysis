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

#include "myAliFemtoV0TrackCut.h"
#include "myAliFemtoESDTrackCut.h"

#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoAvgSepCorrFctn.h"
#include "myAliFemtoSepCorrFctns.h"
#include "myAliFemtoAvgSepCorrFctnCowboysAndSailors.h"
#include "myAliFemtoKStarCorrFctn2D.h"
#include "myAliFemtoKStarCorrFctnMC.h"

//-----04/02/2016
#include "AliFemtoModelWeightGeneratorBasicLednicky.h"
#include "myAliFemtoModelCorrFctnKStar.h"

#include "myAnalysisConstructor.h"
#endif

AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  gROOT->LoadMacro("myAliFemtoV0TrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoESDTrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn.cxx+g");
  gROOT->LoadMacro("myAliFemtoAvgSepCorrFctn.cxx+g");
  gROOT->LoadMacro("myAliFemtoSepCorrFctns.cxx+g");
  gROOT->LoadMacro("myAliFemtoAvgSepCorrFctnCowboysAndSailors.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn2D.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctnMC.cxx+g");
  gROOT->LoadMacro("AliFemtoModelWeightGeneratorBasicLednicky.cxx+g");
  gROOT->LoadMacro("myAliFemtoModelCorrFctnKStar.cxx+g");

  gROOT->LoadMacro("myAnalysisConstructor.cxx+g");

  bool ImplementAvgSepCuts = true;
  bool ImplementVertexCorrection = true;
  bool RunMC = false;

  //-----17/12/2015
  bool UseAliFemtoV0TrackCutNSigmaFilter = false;
  bool UseCustomNSigmaFilters = false;

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
  myAnalysisConstructor *analy_LamK0 = new myAnalysisConstructor(myAnalysisConstructor::kLamK0, "LamK0",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamK0->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_LamK0->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_LamK0->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_LamK0->SetCorrectAnalysis();

    //_______Centrality dependent______________________________________________________
    myAnalysisConstructor *analy_LamK0_0010 = new myAnalysisConstructor(myAnalysisConstructor::kLamK0, "LamK0_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamK0_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamK0_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamK0_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_0010->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamK0_1030 = new myAnalysisConstructor(myAnalysisConstructor::kLamK0, "LamK0_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamK0_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamK0_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamK0_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_1030->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamK0_3050 = new myAnalysisConstructor(myAnalysisConstructor::kLamK0, "LamK0_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamK0_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamK0_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamK0_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_LamK0_3050->SetCorrectAnalysis();


  //------------------------------- AntiLambda-K0Short -----------------------------------------------
  myAnalysisConstructor *analy_ALamK0 = new myAnalysisConstructor(myAnalysisConstructor::kALamK0, "ALamK0",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamK0->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_ALamK0->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_ALamK0->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
    analy_ALamK0->SetCorrectAnalysis();

    //_______Centrality dependent______________________________________________________
    myAnalysisConstructor *analy_ALamK0_0010 = new myAnalysisConstructor(myAnalysisConstructor::kALamK0, "ALamK0_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamK0_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamK0_0010->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_0010->SetCorrectAnalysis();

    myAnalysisConstructor *analy_ALamK0_1030 = new myAnalysisConstructor(myAnalysisConstructor::kALamK0, "ALamK0_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamK0_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamK0_1030->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_1030->SetCorrectAnalysis();

    myAnalysisConstructor *analy_ALamK0_3050 = new myAnalysisConstructor(myAnalysisConstructor::kALamK0, "ALamK0_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_ALamK0_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamK0_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamK0_3050->SetRemoveMisidentifiedMCParticles(RemoveMisidentifiedMCParticles);
      analy_ALamK0_3050->SetCorrectAnalysis();



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


