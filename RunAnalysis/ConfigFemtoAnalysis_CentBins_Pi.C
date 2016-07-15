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
  //------------------------------- Lambda-Pi+ -----------------------------------------------
  myAnalysisConstructor *analy_LamPiP = new myAnalysisConstructor(myAnalysisConstructor::kLamPiP, "LamPiP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamPiP->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_LamPiP->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_LamPiP->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_LamPiP_0010 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiP, "LamPiP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
          analy_LamPiP_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiP_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiP_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_LamPiP_1030 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiP, "LamPiP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
          analy_LamPiP_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiP_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiP_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_LamPiP_3050 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiP, "LamPiP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
          analy_LamPiP_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiP_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiP_3050->SetCorrectAnalysis();


  //------------------------------- Lambda-Pi- -----------------------------------------------
  myAnalysisConstructor *analy_LamPiM = new myAnalysisConstructor(myAnalysisConstructor::kLamPiM, "LamPiM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_LamPiM->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_LamPiM->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_LamPiM->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_LamPiM_0010 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiM, "LamPiM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
          analy_LamPiM_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiM_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiM_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_LamPiM_1030 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiM, "LamPiM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
          analy_LamPiM_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiM_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiM_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_LamPiM_3050 = new myAnalysisConstructor(myAnalysisConstructor::kLamPiM, "LamPiM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
          analy_LamPiM_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_LamPiM_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_LamPiM_3050->SetCorrectAnalysis();


  //------------------------------- AntiLambda-Pi+ -----------------------------------------------
  myAnalysisConstructor *analy_ALamPiP = new myAnalysisConstructor(myAnalysisConstructor::kALamPiP, "ALamPiP",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamPiP->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_ALamPiP->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_ALamPiP->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_ALamPiP_0010 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiP, "ALamPiP_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
          analy_ALamPiP_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiP_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiP_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_ALamPiP_1030 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiP, "ALamPiP_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
          analy_ALamPiP_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiP_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiP_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_ALamPiP_3050 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiP, "ALamPiP_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
          analy_ALamPiP_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiP_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiP_3050->SetCorrectAnalysis();


  //------------------------------- AntiLambda-Pi- -----------------------------------------------
  myAnalysisConstructor *analy_ALamPiM = new myAnalysisConstructor(myAnalysisConstructor::kALamPiM, "ALamPiM",10,-10.,10., 20, 0, 1000, RunMC, ImplementAvgSepCuts);
    analy_ALamPiM->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
    analy_ALamPiM->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
    analy_ALamPiM->SetCorrectAnalysis();

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_ALamPiM_0010 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiM, "ALamPiM_0010",10,-10.,10., 2, 0, 100, RunMC, ImplementAvgSepCuts);
          analy_ALamPiM_0010->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiM_0010->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiM_0010->SetCorrectAnalysis();

	myAnalysisConstructor *analy_ALamPiM_1030 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiM, "ALamPiM_1030",10,-10.,10., 4, 100, 300, RunMC, ImplementAvgSepCuts);
          analy_ALamPiM_1030->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiM_1030->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiM_1030->SetCorrectAnalysis();

	myAnalysisConstructor *analy_ALamPiM_3050 = new myAnalysisConstructor(myAnalysisConstructor::kALamPiM, "ALamPiM_3050",10,-10.,10., 4, 300, 500, RunMC, ImplementAvgSepCuts);
          analy_ALamPiM_3050->SetUseAliFemtoV0TrackCutNSigmaFilter(UseAliFemtoV0TrackCutNSigmaFilter);
          analy_ALamPiM_3050->SetUseCustomNSigmaFilters(UseCustomNSigmaFilters);
          analy_ALamPiM_3050->SetCorrectAnalysis();


  // Add the analyses to the manager
//  mgr->AddAnalysis(analy_LamPiP);
      mgr->AddAnalysis(analy_LamPiP_0010);
      mgr->AddAnalysis(analy_LamPiP_1030);
      mgr->AddAnalysis(analy_LamPiP_3050);
//  mgr->AddAnalysis(analy_LamPiM);
      mgr->AddAnalysis(analy_LamPiM_0010);
      mgr->AddAnalysis(analy_LamPiM_1030);
      mgr->AddAnalysis(analy_LamPiM_3050);
//  mgr->AddAnalysis(analy_ALamPiP);
      mgr->AddAnalysis(analy_ALamPiP_0010);
      mgr->AddAnalysis(analy_ALamPiP_1030);
      mgr->AddAnalysis(analy_ALamPiP_3050);
//  mgr->AddAnalysis(analy_ALamPiM);
      mgr->AddAnalysis(analy_ALamPiM_0010);
      mgr->AddAnalysis(analy_ALamPiM_1030);
      mgr->AddAnalysis(analy_ALamPiM_3050);

  return mgr;
}


