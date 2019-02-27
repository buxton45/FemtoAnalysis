#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "FitValuesWriter.h"
#include "FitValuesWriterwSysErrs.h"
#include "FitValuesLatexTableHelperWriterwSysErrs.h"

#include "TObjString.h"


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  TString tResultsDate = "20180505";
  bool bUseDefaults = true;
  //-----

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType_LamKch = kPolynomial;
  NonFlatBgdFitType tNonFlatBgdFitType_LamK0  = kLinear;
  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{tNonFlatBgdFitType_LamK0, tNonFlatBgdFitType_LamK0, 
                                                tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch};

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  bool FixD0 = false;

  bool bUseStavCf=false;
  bool FixAllLambdaTo1 = false;
  bool FixAllNormTo1 = false;
  bool FixRadii = false;
  bool FixAllScattParams = false;

  bool tShareLambdaParams = true;
  bool tAllShareSingleLambdaParam = false;
  bool UsemTScalingOfResidualRadii = false;
  bool tIsDualie=true;

  bool tDualieShareLambda = true;
  bool tDualieShareRadii = true;





  //*******************************************************************



  TString tFitInfoTString = FitValuesWriter::BuildFitInfoTString(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes,
                                                                 tIncludeResidualsType, tResPrimMaxDecayType, 
                                                                 tChargedResidualsType, FixD0, 
                                                                 bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams, 
                                                                 tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, tIsDualie, 
                                                                 tDualieShareLambda, tDualieShareRadii);

  FitValuesLatexTableHelperWriterwSysErrs* tFitValLatTabHelpWriterwSysErrs = new FitValuesLatexTableHelperWriterwSysErrs();
  tFitValLatTabHelpWriterwSysErrs->WriteSingleLatexTableHelperTriple(tResultsDate, tFitInfoTString);


//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
