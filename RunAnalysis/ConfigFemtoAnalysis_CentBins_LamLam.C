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

  gROOT->LoadMacro("myAnalysisConstructor.cxx+g");

  bool ImplementAvgSepCuts = true;
  bool ImplementVertexCorrection = true;
  bool RunMC = false;

  //-----17/12/2015
  bool UseAliFemtoV0TrackCutNSigmaFilter = false;
  bool UseCustomNSigmaFilters = false;


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
  //------------------------------- Lambda-Lambda -----------------------------------------------
  myAnalysisConstructor *analy_LamLam = new myAnalysisConstructor(myAnalysisConstructor::kLamLam, "LamLam",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamLam->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_LamLam->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_LamLam->SetCorrectAnalysis();

    //_______Centrality dependent______________________________________________________
    myAnalysisConstructor *analy_LamLam_0010 = new myAnalysisConstructor(myAnalysisConstructor::kLamLam, "LamLam_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamLam_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamLam_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamLam_0010->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamLam_1030 = new myAnalysisConstructor(myAnalysisConstructor::kLamLam, "LamLam_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamLam_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamLam_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamLam_1030->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamLam_3050 = new myAnalysisConstructor(myAnalysisConstructor::kLamLam, "LamLam_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamLam_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamLam_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamLam_3050->SetCorrectAnalysis();


  //------------------------------- AntiLambda-AntiLambda -----------------------------------------------
  myAnalysisConstructor *analy_ALamALam = new myAnalysisConstructor(myAnalysisConstructor::kALamALam, "ALamALam",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamALam->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_ALamALam->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_ALamALam->SetCorrectAnalysis();

    //_______Centrality dependent______________________________________________________
    myAnalysisConstructor *analy_ALamALam_0010 = new myAnalysisConstructor(myAnalysisConstructor::kALamALam, "ALamALam_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_ALamALam_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamALam_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamALam_0010->SetCorrectAnalysis();

    myAnalysisConstructor *analy_ALamALam_1030 = new myAnalysisConstructor(myAnalysisConstructor::kALamALam, "ALamALam_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_ALamALam_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamALam_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamALam_1030->SetCorrectAnalysis();

    myAnalysisConstructor *analy_ALamALam_3050 = new myAnalysisConstructor(myAnalysisConstructor::kALamALam, "ALamALam_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_ALamALam_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_ALamALam_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_ALamALam_3050->SetCorrectAnalysis();



  //------------------------------- Lambda-AntiLambda -----------------------------------------------
  myAnalysisConstructor *analy_LamALam = new myAnalysisConstructor(myAnalysisConstructor::kLamALam, "LamALam",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamALam->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_LamALam->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_LamALam->SetCorrectAnalysis();

    //_______Centrality dependent______________________________________________________
    myAnalysisConstructor *analy_LamALam_0010 = new myAnalysisConstructor(myAnalysisConstructor::kLamALam, "LamALam_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
      analy_LamALam_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamALam_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamALam_0010->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamALam_1030 = new myAnalysisConstructor(myAnalysisConstructor::kLamALam, "LamALam_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
      analy_LamALam_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamALam_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamALam_1030->SetCorrectAnalysis();

    myAnalysisConstructor *analy_LamALam_3050 = new myAnalysisConstructor(myAnalysisConstructor::kLamALam, "LamALam_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
      analy_LamALam_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
      analy_LamALam_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
      analy_LamALam_3050->SetCorrectAnalysis();



  // Add the analyses to the manager
//  mgr->AddAnalysis(analy_LamLam);
    mgr->AddAnalysis(analy_LamLam_0010);
    mgr->AddAnalysis(analy_LamLam_1030);
    mgr->AddAnalysis(analy_LamLam_3050);
//  mgr->AddAnalysis(analy_ALamALam);
    mgr->AddAnalysis(analy_ALamALam_0010);
    mgr->AddAnalysis(analy_ALamALam_1030);
    mgr->AddAnalysis(analy_ALamALam_3050);
//  mgr->AddAnalysis(analy_LamALam);
    mgr->AddAnalysis(analy_LamALam_0010);
    mgr->AddAnalysis(analy_LamALam_1030);
    mgr->AddAnalysis(analy_LamALam_3050);

  return mgr;
}

