///////////////////////////////////////////////////////////////////////////
// FitSystematicAnalysis:                                                //
///////////////////////////////////////////////////////////////////////////


#include "FitSystematicAnalysis.h"

#ifdef __ROOT__
ClassImp(FitSystematicAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fApplyNonFlatBackgroundCorrection(false),
  fApplyMomResCorrection(false),
  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),
  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2)

{
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fApplyNonFlatBackgroundCorrection(false),
  fApplyMomResCorrection(false),
  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(0),
  fModifierValues1(aModifierValues1),
  fModifierValues2(0)

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  FitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aDirNameModifierBase1, aModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams);
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::~FitSystematicAnalysis()
{
/*no-op*/
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::RunAllFits(bool aSave)
{
  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    FitGenerator* tFitGenerator = new FitGenerator(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, kTrainSys, 2, fCentralityType, fFitGeneratorType, fShareLambdaParams, tDirNameModifier);

    tFitGenerator->DoFit(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(false);
    if(aSave)
    {
      TString tSaveName = fSaveDirectory;
      tSaveName += tKStarwFitsCan->GetTitle();
      TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
      if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

      tSaveName += tDirNameModifier + TString(".pdf");
      tKStarwFitsCan->SaveAs(tSaveName);
    }

  }
}



