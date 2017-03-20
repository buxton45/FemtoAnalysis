/*
 * Run1.C
 *
 */

#ifndef __CINT__

#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TInterpreter.h>
#include <TString.h>
#include <AliAnalysisAlien.h>
#include <AliAnalysisManager.h>
#include <AliAODInputHandler.h>
#include <AliAnalysisTaskSE.h>
#include <AliAnalysisTaskFemto.h>

#include "AliFemtoAnalysisLambdaKaon.h"

#endif

bool RunMC = false;
TString aConfigMacro = "ConfigTrainFemtoAnalysis.C";


TString outputname = "Results_cLamK0_20160928_Bm1_New.root";

TString tConfiguration = "tMacroPath='%%/ConfigFemtoAnalysis.C'; tContainerName = 'cLamK0_femtolist'; ";
//TString tParams = "@implementVertexCorrections = true; $|Lam|maxInvariantMass=1.165683; ";
TString tParams = "@implementVertexCorrections = true; @analysisType = AliFemtoAnalysisLambdaKaon::kProtPiM; [0:10:30:50]";

void RunMe()
{
    cout << "[RunMe] Begin\n";
cout << "tParams = " << tParams << endl;

    // Setup includes
    gInterpreter->AddIncludePath("$ALICE_PHYSICS/include");
    gInterpreter->AddIncludePath("$ALICE_ROOT/include");
    gInterpreter->AddIncludePath("$ALICE_PHYSICS/../src/PWGCF/FEMTOSCOPY/AliFemto");
    gInterpreter->AddIncludePath("$ALICE_PHYSICS/../src/PWGCF/FEMTOSCOPY/AliFemtoUser");

    //
    gSystem->Load("libVMC.so");
    gSystem->Load("libTree.so");
    gSystem->Load("libAOD.so");
    gSystem->Load("libANALYSIS.so");
    gSystem->Load("libANALYSISalice.so");
    gSystem->Load("libPWGCFfemtoscopy.so");
    gSystem->Load("libPWGCFfemtoscopyUser.so");

    gROOT->LoadMacro("AliFemtoAnalysisLambdaKaon.cxx+g");
    gROOT->LoadMacro("AddTaskLambdaKaon.C+g");

    // Create Analysis Manager
    AliAnalysisManager *mgr = new AliAnalysisManager("mgr", "My Analysis Manager");
//    mgr->SetDebugLevel(2);
//    mgr->SetUseProgressBar(kTRUE);

    // Create AOD input event handler
    AliAODInputHandler *aod_handler = new AliAODInputHandler();
      aod_handler->CreatePIDResponse(kFALSE);
      mgr->SetInputEventHandler(aod_handler);

    // Must be run for PID to work correctly
    gROOT->LoadMacro("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
      AddTaskPIDResponse(RunMC);

    // Must be run for V0 to work correctly
    gROOT->LoadMacro("$ALICE_ROOT/ANALYSIS/macros/AddTaskVZEROEPSelection.C");
      AddTaskVZEROEPSelection();

    // Create the AliFemto task using configuration from ConfigFemtoAnalysis.C
  AliAnalysisTaskFemto *taskfemto = AddTaskLambdaKaon(tConfiguration,tParams,"");

  // Initialize the analysis
  if (!mgr->InitAnalysis()) {
      cerr << "Error Initting Analysis. Exiting.\n";
      exit(1);
  }

  // output status to make sure things look good
  mgr->PrintStatus();


  // Create a TChain of input data files
  TChain *input_files = new TChain("aodTree");
  if(!RunMC)
  {

    input_files->Add("/aliceData/data/2011/LHC11h_2/000170593/ESDs/pass2/AOD145/0001/AliAOD.root");
/*
    input_files->Add("/aliceData/data/2011/LHC11h_2/000170593/ESDs/pass2/AOD145/0002/AliAOD.root");
    input_files->Add("/aliceData/data/2011/LHC11h_2/000170572/ESDs/pass2/AOD145/0001/AliAOD.root");
    input_files->Add("/aliceData/data/2011/LHC11h_2/000170572/ESDs/pass2/AOD145/0002/AliAOD.root");
    input_files->Add("/aliceData/data/2011/LHC11h_2/000170388/ESDs/pass2/AOD145/0001/AliAOD.root");
    input_files->Add("/aliceData/data/2011/LHC11h_2/000170388/ESDs/pass2/AOD145/0002/AliAOD.root");
*/

    //input_files->Add("~/Analysis/K0Lam/Data/LHC11h/000169846_ESDs_pass2_AOD145_1262/AliAOD.root");  //Jai Baseline
  }
  if(RunMC)
  {
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170593/AOD149/0001/AliAOD.root");
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170593/AOD149/0002/AliAOD.root");
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170572/AOD149/0001/AliAOD.root");
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170572/AOD149/0002/AliAOD.root");
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170388/AOD149/0001/AliAOD.root");
    input_files->Add("/aliceData/sim/2012/LHC12a17a_fix/170388/AOD149/0002/AliAOD.root");
  }

  // Tell manager to do a local analysis on the supplied input files.
  mgr->StartAnalysis("local", input_files);
//  std::cout << ConfigFemtoAnalysis()->Finish() << '\n';

  
  cout << "[RunMe] End.\n";
//     exit(0);
}
