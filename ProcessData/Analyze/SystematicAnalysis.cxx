///////////////////////////////////////////////////////////////////////////
// SystematicAnalysis:                                                   //
///////////////////////////////////////////////////////////////////////////


#include "SystematicAnalysis.h"

#ifdef __ROOT__
ClassImp(SystematicAnalysis)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
SystematicAnalysis::SystematicAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       TString aDirNameModifierBase2="", vector<double> &aModifierValues2 = vector<double>()) :
  fFileLocationBase(aFileLocationBase),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),
  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2),

  fAnalyses(0)

{
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    fAnalyses.emplace_back(fFileLocationBase,fAnalysisType,fCentralityType,kTrain,2,tDirNameModifier);
  }


}


//________________________________________________________________________________________________________________
SystematicAnalysis::~SystematicAnalysis()
{

}




