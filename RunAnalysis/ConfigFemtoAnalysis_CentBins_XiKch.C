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

  bool ImplementAvgSepCuts = false;
  bool WritePairKinematics = false;
  bool ImplementVertexCorrection = false;
  bool RunMC = true;

  //-----17/12/2015
  bool UseAliFemtoV0TrackCutNSigmaFilter = false;
  bool UseCustomNSigmaFilters = false;


  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    //rdr->SetCentralityPreSelection(0, 900);
//    rdr->SetReadV0(1);  //this should probably be switched off here?
    rdr->SetReadCascade(1);
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kFALSE);
    rdr->SetPrimaryVertexCorrectionTPCPoints(ImplementVertexCorrection);
    rdr->SetReadMC(RunMC);

  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
    //Point to the data source - the reader
    mgr->SetEventReader(rdr);


  //Setup the analyses
  //------------------------------- Cascade-K+ -----------------------------------------------
  myAnalysisConstructor *analy_XiKchP = new myAnalysisConstructor(myAnalysisConstructor::kXiKchP, "XiKchP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts, WritePairKinematics);
    analy_XiKchP->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_XiKchP->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_XiKchP->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_XiKchP_0010 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchP, "XiKchP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchP_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchP_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchP_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_XiKchP_1030 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchP, "XiKchP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchP_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchP_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchP_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_XiKchP_3050 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchP, "XiKchP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchP_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchP_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchP_3050->SetCorrectAnalysis();


  //------------------------------- Cascade-K- -----------------------------------------------
  myAnalysisConstructor *analy_XiKchM = new myAnalysisConstructor(myAnalysisConstructor::kXiKchM, "XiKchM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts, WritePairKinematics);
    analy_XiKchM->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_XiKchM->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_XiKchM->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_XiKchM_0010 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchM, "XiKchM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchM_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchM_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchM_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_XiKchM_1030 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchM, "XiKchM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchM_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchM_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchM_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_XiKchM_3050 = new myAnalysisConstructor(myAnalysisConstructor::kXiKchM, "XiKchM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_XiKchM_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_XiKchM_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_XiKchM_3050->SetCorrectAnalysis();


  //------------------------------- AntiCascade-K+ -----------------------------------------------
  myAnalysisConstructor *analy_AXiKchP = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchP, "AXiKchP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts, WritePairKinematics);
    analy_AXiKchP->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_AXiKchP->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_AXiKchP->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_AXiKchP_0010 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchP, "AXiKchP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchP_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchP_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchP_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_AXiKchP_1030 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchP, "AXiKchP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchP_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchP_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchP_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_AXiKchP_3050 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchP, "AXiKchP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchP_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchP_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchP_3050->SetCorrectAnalysis();


  //------------------------------- AntiCascade-K- -----------------------------------------------
  myAnalysisConstructor *analy_AXiKchM = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchM, "AXiKchM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts, WritePairKinematics);
    analy_AXiKchM->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_AXiKchM->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_AXiKchM->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_AXiKchM_0010 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchM, "AXiKchM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchM_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchM_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchM_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_AXiKchM_1030 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchM, "AXiKchM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchM_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchM_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchM_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_AXiKchM_3050 = new myAnalysisConstructor(myAnalysisConstructor::kAXiKchM, "AXiKchM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts, WritePairKinematics);
          analy_AXiKchM_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_AXiKchM_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_AXiKchM_3050->SetCorrectAnalysis();


  // Add the analyses to the manager
//  mgr->AddAnalysis(analy_XiKchP);
      mgr->AddAnalysis(analy_XiKchP_0010);
/*
      mgr->AddAnalysis(analy_XiKchP_1030);
      mgr->AddAnalysis(analy_XiKchP_3050);
//  mgr->AddAnalysis(analy_XiKchM);
*/
      mgr->AddAnalysis(analy_XiKchM_0010);
/*
      mgr->AddAnalysis(analy_XiKchM_1030);
      mgr->AddAnalysis(analy_XiKchM_3050);
//  mgr->AddAnalysis(analy_AXiKchP);
*/
      mgr->AddAnalysis(analy_AXiKchP_0010);
/*
      mgr->AddAnalysis(analy_AXiKchP_1030);
      mgr->AddAnalysis(analy_AXiKchP_3050);
//  mgr->AddAnalysis(analy_AXiKchM);
*/
      mgr->AddAnalysis(analy_AXiKchM_0010);
/*
      mgr->AddAnalysis(analy_AXiKchM_1030);
      mgr->AddAnalysis(analy_AXiKchM_3050);
*/
  return mgr;
}


