///
/// \file PWGCF/FEMTOSCOPY/macros/Train/LambdaKaonFemto/AddTaskLambdaKaon.C
/// \author Jesse Buxton, Ohio State University, jesse.thomas.buxton@cern.ch
///
/// \ but based largely off of work done by Andrew Kubera in
/// \ file PWGCF/FEMTOSCOPY/macros/Train/PionPionFemto/AddTaskPionPion.C
/// \   Andrew Kubera, Ohio State University, andrew.kubera@cern.ch
///


AliAnalysisTaskFemto* AddTaskLambdaKaon(TString aConfiguration,
                                        TString aParams,
                                        TString aSubwagonSuffix="")
{ // Adds a Lambda-Kaon Femtoscopy task to the manager

  const TString AUTO_DIRECTORY = "$ALICE_PHYSICS/PWGCF/FEMTOSCOPY/macros/Train/LambdaKaonFemto"
              , DEFAULT_MACRO = "%%/ConfigFemtoAnalysis.C"
              , DEFAULT_CONTAINER_NAME = "LambdaKaon_femtolist"
              , DEFAULT_OUTPUT_CONTAINER = "PWG2FEMTO"
              , DEFAULT_TASK_NAME = "TaskLambdaKaon"
              ;


  // Get the pointer to the existing analysis manager via the static access method.
  AliAnalysisManager *mgr = AliAnalysisManager::GetAnalysisManager();
  if (!mgr) {
    Error("AddTaskLambdaKaon", "No analysis manager to connect to.");
    return NULL;
  }

  // Check the analysis type using the event handlers connected to the analysis
  // manager. The availability of MC handler cann also be checked here.
  if (!mgr->GetInputEventHandler()) {
    ::Error("AddTaskFemto", "This task requires an input event handler");
    return NULL;
  }
  TString type = mgr->GetInputEventHandler()->GetDataType(); // can be "ESD" or "AOD"
  cout << "Found " <<type << " event handler" << endl;

  TString tMacroPath = DEFAULT_MACRO
        , tOutputFilename = mgr->GetCommonFileName()
        , tTaskName = DEFAULT_TASK_NAME
        , tContainerName = DEFAULT_CONTAINER_NAME
        , tOutputContainerName = DEFAULT_OUTPUT_CONTAINER
        ;

  bool tVerbose = kFALSE;

  TObjArray* tLines = aConfiguration.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    TString tCmd = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    tCmd.ReplaceAll("'", '"');
    gROOT->ProcessLineFast(tCmd + ';');
  }


  // Replace %% with this directory for convenience
  tMacroPath.ReplaceAll("%%", AUTO_DIRECTORY);

  cout << "[AddTaskLambdaKaon]\n"
          "   output: '" << tOutputFilename << "'\n"
          "   macro: '" << tMacroPath << "'\n"
          "   params: '" << aParams << "'\n";

  // The analysis config macro for LambdaKaonFemto accepts a single string
  // argument, which it interprets.
  // This line escapes some escapable characters (backslash, newline, tab)
  // and wraps that string in double quotes, ensuring that the interpreter
  // reads a string when passing to the macro.
  const TString tAnalysisParams = '"' + aParams.ReplaceAll("\\", "\\\\")
                                               .ReplaceAll("\n", "\\n")
                                               .ReplaceAll("\t", "\\t") + '"';

  AliAnalysisTaskFemto *tTaskFemto = new AliAnalysisTaskFemto(
    tTaskName,
    tMacroPath,
    tAnalysisParams,
    tVerbose
  );

  mgr->AddTask(tTaskFemto);

  const TString tOutputFile = (tOutputContainerName == "")
                            ? tOutputFilename
                            : TString::Format("%s:%s", tOutputFilename.Data(), tOutputContainerName.Data());

  AliAnalysisDataContainer *tOutContainer = mgr->CreateContainer(tContainerName,
                                                                 TList::Class(),
                                                                 AliAnalysisManager::kOutputContainer,
                                                                 tOutputFile);

  // connect task to the containers
  mgr->ConnectInput(tTaskFemto, 0, mgr->GetCommonInputContainer());
  mgr->ConnectOutput(tTaskFemto, 0, tOutContainer);

  // Return the task pointer
  return tTaskFemto;


}
