#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "FitValuesWriter.h"
#include "FitValuesLatexTableHelperWriter.h"

#include "TObjString.h"

LednickyFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateFitFunction(npar,f,par);
}


//______________________________________________________________________________
td1dVec ReadLine(TString aLine)
{
  td1dVec tReturnVec(0);

  TObjArray* tValues = aLine.Tokenize("  ");

  double tValue;
  for(int i=0; i<tValues->GetEntries(); i++)
  {
    tValue = ((TObjString*)tValues->At(i))->String().Atof();
    tReturnVec.push_back(tValue);
  }
  return tReturnVec;
}


//______________________________________________________________________________
vector<int> GetNParamsAndRowWidth(ifstream &aStream)
{
  std::string tStdString;
  TString tLine;

  int tNParams = 0;
  int tRowWidth = 0;
  while(getline(aStream, tStdString))
  {
    tLine = TString(tStdString);
    if(tLine.Contains("*")) continue;
    if(tLine.Contains("PARAMETER")) continue;
    if(tLine.Contains("NO.")) continue;

    TObjArray* tValues = tLine.Tokenize("  ");
    if(tNParams==0) tRowWidth = tValues->GetEntries();
    if(tValues->GetEntries() == tRowWidth) tNParams++;
  }
  aStream.clear();
  aStream.seekg(0, ios::beg);

  return vector<int>{tNParams, tRowWidth};
}

//______________________________________________________________________________
void FinishMatrix(td2dVec &aMatrix, vector<int> &aNParamsAndRowWidth)
{
  //Due to how things are printed, and therefore, how I read them,
  // the first RowWidth rows only have RowWidth entries, whereas the
  // remaining (NParams-RowWidth) rows have full NParams entries
  int tNParams = aNParamsAndRowWidth[0];
  int tRowWidth = aNParamsAndRowWidth[1];

  for(int i=0; i<tRowWidth; i++)
  {
    for(int j=tRowWidth; j<tNParams; j++)
    {
      aMatrix[i].push_back(aMatrix[j][i]);
    }
  }

  //The matrix should now be symmetric about the diagonal, check this
  assert((int)aMatrix.size()==tNParams);
  for(unsigned int i=0; i<aMatrix.size(); i++)
  {
    assert((int)aMatrix[i].size()==tNParams);
    for(unsigned int j=0; j<aMatrix[i].size(); j++)
    {
      assert(aMatrix[i][j]==aMatrix[j][i]);
    }
  }

}


//______________________________________________________________________________
void PrintMatrix(td2dVec &aMatrix)
{
  for(unsigned int i=0; i<aMatrix.size(); i++)
  {
    for(unsigned int j=0; j<aMatrix[i].size(); j++)
    {
      printf("% 05.3f  ", aMatrix[i][j]);
    }
    cout << endl;
  }
}



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
//Be sure to set the following...
/*
  TString FileLocation_cLamK0_Bp1 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bp1.root";
  TString FileLocation_cLamK0_Bp2 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bp2.root";
  TString FileLocation_cLamK0_Bm1 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm1.root";
  TString FileLocation_cLamK0_Bm2 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm2.root";
  TString FileLocation_cLamK0_Bm3 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923_Bm3.root";

  FitPartialAnalysis* tFitPartialAnalysis_Bp1 = new FitPartialAnalysis(FileLocation_cLamK0_Bp1, "LamK0_0010_Bp1", kLamK0, k0010, kBp1);
  cout << sizeof(tFitPartialAnalysis_Bp1) << endl;
*/
/*
  FitPartialAnalysis* tFitPartialAnalysis_Bp2 = new FitPartialAnalysis(FileLocation_cLamK0_Bp2, "LamK0_0010_Bp2", kLamK0, k0010, kBp2);
  FitPartialAnalysis* tFitPartialAnalysis_Bm1 = new FitPartialAnalysis(FileLocation_cLamK0_Bm1, "LamK0_0010_Bm1", kLamK0, k0010, kBm1);
  FitPartialAnalysis* tFitPartialAnalysis_Bm2 = new FitPartialAnalysis(FileLocation_cLamK0_Bm2, "LamK0_0010_Bm2", kLamK0, k0010, kBm2);
  FitPartialAnalysis* tFitPartialAnalysis_Bm3 = new FitPartialAnalysis(FileLocation_cLamK0_Bm3, "LamK0_0010_Bm3", kLamK0, k0010, kBm3);
*/
/*
  vector<FitPartialAnalysis*> tempVec;
  tempVec.push_back(tFitPartialAnalysis);

  FitPairAnalysis* tFitPairAnalysis = new FitPairAnalysis("LamK0",tempVec);
*/

/*
  TString aFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20171227/ParameterCorrelations/ParameterCorrelationCoefficients_LamKchMwConj_MomResCrctn_NonFlatBgdCrctn_NoRes.txt";
  ifstream tFileIn(aFileLocation);
  if(!tFileIn.is_open()) cout << "FAILURE - FILE NOT OPEN: " << aFileLocation << endl;
  assert(tFileIn.is_open());

  vector<int> tNParamsAndRowWidth = GetNParamsAndRowWidth(tFileIn);
  cout << "tNParams = " << tNParamsAndRowWidth[0] << endl;
  cout << "tRowWidth = " << tNParamsAndRowWidth[1] << endl << endl;

  td2dVec tValuesMatrix;
  int tCounter = -1;

  std::string tStdString;
  TString tLine;
  while(getline(tFileIn, tStdString))
  {
    tLine = TString(tStdString);
    if(tLine.Contains("*")) continue;
    if(tLine.Contains("PARAMETER")) continue;
    if(tLine.Contains("NO.")) continue;

    td1dVec tValuesVec = ReadLine(tLine);

    if((int)tValuesVec.size() == tNParamsAndRowWidth[1])
    {
      tValuesVec.erase(tValuesVec.begin(), tValuesVec.begin()+2);  //First value is parameter number
                                                                   //Second value is global correlation value

      tValuesMatrix.push_back(tValuesVec);
      tCounter++;
    }
    else
    {
      tValuesMatrix[tCounter].insert(tValuesMatrix[tCounter].end(), tValuesVec.begin(), tValuesVec.end());
    }
  }
  tFileIn.close();

  //Due to the erase call above, RowWidth has decreased by 2
  tNParamsAndRowWidth[1] -= 2;
  FinishMatrix(tValuesMatrix, tNParamsAndRowWidth);
  PrintMatrix(tValuesMatrix);
*/

/*
  TString tFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tResultsDate = "20180505";
  AnalysisType tAnType = kLamKchP;

  FitValuesWriter* tFitValWriter = new FitValuesWriter();
  vector<vector<FitParameter*> > tAllFitResults = tFitValWriter->GetAllFitResults(tFileLocation, "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly", "");
  vector<FitParameter*> tFitResults = tFitValWriter->GetFitResults(tFileLocation, "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly", "", kALamKchM, k1030);
  FitParameter* tFitParam = tFitValWriter->GetFitParameter(tFileLocation, "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly", "", kALamKchM, k1030, kLambda);
*/


  TString tFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tResultsDate = "20180505";
  AnalysisType tAnType = kLamKchP;

  TString tHelperBaseLocation = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter/testHelper";

  FitValuesLatexTableHelperWriter* tFitValLaTaHelpWriter = new FitValuesLatexTableHelperWriter();
  tFitValLaTaHelpWriter->WriteLatexTableHelper(tHelperBaseLocation, tFileLocation, tAnType, kInclude3Residuals);
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
