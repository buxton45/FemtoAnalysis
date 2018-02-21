#include "SystematicAnalysis.h"
class SystematicAnalysis;

#include "Types_SysFileInfo.h"

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  TString tGeneralAnTypeName = "cLamcKch";
  bool bWriteToFile = true;
  SystematicAnalysis::DiffHistFitType tFitType = SystematicAnalysis::kExpDecay;
  bool tFixOffsetParam = false;

  SystematicsFileInfo tFileInfo = GetFileInfo_LamK(12);
    TString tResultsDate = tFileInfo.resultsDate;
    TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
    vector<double> tModifierValues1 = tFileInfo.modifierValues1;
    TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
    vector<double> tModifierValues2 = tFileInfo.modifierValues2;
    bool tAllCent = tFileInfo.allCentralities;

  CentralityType tMaxCentType = kMB;
  if(!tAllCent) tMaxCentType = k1030;


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tOutputFileName = tDirectoryBase + TString("AllFitValues.txt");
  std::ofstream tOutputFile;
  if(bWriteToFile) tOutputFile.open(tOutputFileName);

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;

  for(int iAnType=kLamKchP; iAnType<kXiKchP; iAnType++)
  {
    for(int iCent=k0010; iCent<tMaxCentType; iCent++)
    {
      SystematicAnalysis* tSysAn = new SystematicAnalysis(tFileLocationBase, static_cast<AnalysisType>(iAnType), static_cast<CentralityType>(iCent), tDirNameModifierBase1, tModifierValues1, tDirNameModifierBase2, tModifierValues2);
      tSysAn->SetSaveDirectory(tDirectoryBase);
      if(bWriteToFile) tSysAn->GetAllFits(tFitType,tFixOffsetParam,tOutputFile);
      else tSysAn->GetAllFits(tFitType,tFixOffsetParam);
    //tSysAn->DrawAll();
    //tSysAn->DrawAllDiffs(true,tFitType,tFixOffsetParam,false);
    }
  }

cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
