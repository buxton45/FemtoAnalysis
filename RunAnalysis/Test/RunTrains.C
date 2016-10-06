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

#endif

// These are the runs that are submitted to the grid - MUST END IN 0 to indicate stop!
// Jai's Runs :: {170593, 170572, 170388, 170387, 170315, 170313, 170312, 170311, 170309, 170308, 170306, 170270, 170269, 170268, 170230, 170228, 170207, 170204, 170203, 170193, 170163};
//--RUNS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  //-----B=++ :: 41 runs, split 21 (Bp1) and 20 (Bp2)
  int runs_Bp1[] = {170593, 170572, 170388, 170387, 170315, 170313, 170312, 170311, 170309, 170308, 170306, 170270, 170269, 170268, 170230, 170228, 170207, 170204, 170203, 170193, 170163, 0};
  int runs_Bp2[] = {170159, 170155, 170091, 170089, 170088, 170085, 170084, 170083, 170081, 170040, 170027, 169965, 169923, 169859, 169858, 169855, 169846, 169838, 169837, 169835, 0};

  //-----B=--  :: 67 runs, split 23 (Bm1), 22 (Bm2), and 22 (Bm3)
  int runs_Bm1[] = {169591, 169590, 169588, 169587, 169586, 169557, 169555, 169554, 169553, 169550, 169515, 169512, 169506, 169504, 169498, 169475, 169420, 169419, 169418, 169417, 169415, 169411, 169238, 0};
  int runs_Bm2[] = {169167, 169160, 169156, 169148, 169145, 169144, 169138, 169099, 169094, 169091, 169045, 169044, 169040, 169035, 168992, 168988, 168826, 168777, 168514, 168512, 168511, 168467, 0};
  int runs_Bm3[] = {168464, 168460, 168458, 168362, 168361, 168342, 168341, 168325, 168322, 168311, 168310, 168115, 168108, 168107, 168105, 168076, 168069, 167988, 167987, 167985, 167920, 167915, 0};

  int runs_subset[] = {170593, 170572, 170388, 170387, 170315, 170313, 170312, 0};
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

int *use_runs;
use_runs = runs_Bm1;

bool RunGrid = true;
bool RunFull = true;  //set to false when merging
bool RunMC = false;
TString aConfigMacro = "ConfigTrainFemtoAnalysis.C";
TString AdditionalLibs = "libPWGCFfemtoscopy.so libPWGCFfemtoscopyUser.so myTrainAnalysisConstructor.h myTrainAnalysisConstructor.cxx ConfigTrainFemtoAnalysis.C";
TString aGridWorkingDir = "Results_cLamK0_20161006/Bm1_Old";
TString aGridOutputDir = "output_data";
TString outputname = "Results_cLamK0_20161006_Bm1_Old.root";



void RunTrains()
{
    cout << "[RunMe] Begin\n";

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

    gROOT->LoadMacro("myTrainAnalysisConstructor.cxx+g");

    // Create Analysis Manager
    AliAnalysisManager *mgr = new AliAnalysisManager("mgr", "My Analysis Manager");
//    mgr->SetDebugLevel(2);
//    mgr->SetUseProgressBar(kTRUE);

    //Run on grid?
    if (RunGrid) {
        cout << "*** Creating Alien Handler ";
        AliAnalysisAlien *alienHandler = new AliAnalysisAlien();
        if (!alienHandler) {
            cerr << "Error. Could not create Alien Handler. Exiting.\n";
            exit(1);
        }

        alienHandler->SetOverwriteMode();
	if (RunFull) {alienHandler->SetRunMode("full");}
	else{alienHandler->SetRunMode("terminate");} // Set to Terminate to merge the results of a 'full' run

        alienHandler->SetAPIVersion("V1.1x");
//        alienHandler->SetROOTVersion("v5-34-30-alice-8");  //apparently this is no longer needed
//        alienHandler->SetAliROOTVersion("v5-07-20-4");
        alienHandler->SetAliPhysicsVersion("vAN-20161006-1");

        alienHandler->SetAnalysisSource("myTrainAnalysisConstructor.cxx"); // Add any cxx files which need compiled here (.cxx files)
        alienHandler->SetAdditionalLibs(AdditionalLibs); // Add any files which need copied to grid here (.h,.cxx,.C files)

        alienHandler->AddIncludePath("$ALICE_PHYSICS/include");
        alienHandler->AddIncludePath("$ALICE_ROOT/include");
        alienHandler->AddIncludePath("$ALICE_PHYSICS/../src/PWGCF/FEMTOSCOPY/AliFemto/");
        alienHandler->AddIncludePath("$ALICE_PHYSICS/../src/PWGCF/FEMTOSCOPY/AliFemtoUser/");

        // These select the files to run over
	if(!RunMC)
	{
          alienHandler->SetGridDataDir("/alice/data/2011/LHC11h_2");
          alienHandler->SetDataPattern("*ESDs/pass2/AOD145/*/AliAOD.root"); // '*' will go over all data files - replace with 0001 or something to only run on one
	}
	if(RunMC)
	{
	  alienHandler->SetGridDataDir("/alice/sim/2012/LHC12a17a_fix");
	  //alienHandler->SetGridDataDir("/alice/sim/2012/LHC12a17d_fix");
	  alienHandler->SetDataPattern("*AOD149/*/AliAOD.root");
	}
        if (!RunMC) {alienHandler->SetRunPrefix("000");} // "000" for real (non-simulated) data

        // Tell grid to use run numbers defined at top of file
	int i = 0;
	for (i = 0; use_runs[i] != 0; i++) {
          alienHandler->AddRunNumber(use_runs[i]);
        }
        alienHandler->SetNrunsPerMaster(i);

        // Working/Output directories (on grid)
        alienHandler->SetGridWorkingDir(aGridWorkingDir);
        alienHandler->SetGridOutputDir(aGridOutputDir);

        alienHandler->SetCheckCopy(kFALSE);

        // Must be here for merge to work!
        alienHandler->SetMergeViaJDL(kTRUE);
        alienHandler->SetMaxMergeFiles(15);
        alienHandler->SetMaxMergeStages(5);

	// 29 June 2015
	alienHandler->SetSplitMaxInputFileNumber(50); //Jai(30), Andrew(60), default(100)

        cout << "*** Finished.\n";

        mgr->SetGridHandler(alienHandler);
    }

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
  AliAnalysisTaskFemto *taskfemto = new AliAnalysisTaskFemto("TaskFemto", aConfigMacro, "", kTRUE);
  taskfemto->SelectCollisionCandidates(AliVEvent::kMB | AliVEvent::kCentral | AliVEvent::kSemiCentral);

  // Creates the container (output root file) for manager to... create
  AliAnalysisDataContainer *femtolist  = mgr->CreateContainer("femtolist",
                                                              TList::Class(),
                                                              AliAnalysisManager::kOutputContainer,
                                                              outputname);
  // Add tasks to the alimanager
  mgr->AddTask(taskfemto);

  // Tell the task to get input from manager and output to the created container
  mgr->ConnectInput(taskfemto, 0, mgr->GetCommonInputContainer());
  mgr->ConnectOutput(taskfemto, 0, femtolist);

  // Initialize the analysis
  if (!mgr->InitAnalysis()) {
      cerr << "Error Initting Analysis. Exiting.\n";
      exit(1);
  }

  // output status to make sure things look good
  mgr->PrintStatus();

  // Tell manager to run on grid
  if (RunGrid)
  {
    mgr->StartAnalysis("grid");
  }
  else
  {
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
//    std::cout << ConfigFemtoAnalysis()->Finish() << '\n';
  }
  
  cout << "[RunMe] End.\n";
//     exit(0);
}
