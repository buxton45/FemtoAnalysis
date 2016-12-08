#include "FitGenerator.h"
class FitGenerator;

#include <limits>

int main(int argc, char **argv) 
{
//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
//TODO For now, this only works for tCentType = kMB and tGenType = kPairwConj

  TString tResultsDate = "20161027";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = false;

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  FitGenerator* tFitGen = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,tAnRunType,tNPartialAnalysis,tCentType,tGenType,tShareLambdaParams);
//  tFitGen->SetSaveLocationBase(tDirectoryBase,tSaveNameModifier);

  //Set any limits:
//  tFitGen->SetScattParamLimits(-10.0,10.0,-10.0,10.0,-10.0,10.0);

  double tGlobalChi2 = 1000000;
  double tChi2;

  double tTolerance = 0.00000000001;

  td1dVec tTestVec;
  if(tGlobalChi2 == 1000000) tTestVec = td1dVec {0.1,0.2,0.3};
  else tTestVec = td1dVec {0.4,0.5,0.6};

  td1dVec tRef0Values {-0.2,0.0};
  td1dVec tImf0Values {-0.2,0.0};
  td1dVec td0Values {-0.2,0.0};
  vector<int> tWinningIndices {0,0,0};
  vector<vector<int> > tTyingWinningIndices;

  td2dVec tAllValues {tRef0Values,tImf0Values,td0Values};

  int tTotalCalls = tRef0Values.size()*tImf0Values.size()*td0Values.size();
  cout << "Total Number of Calls = " << tTotalCalls << endl << endl;
  int iCall=0;
  for(unsigned int iRe=0; iRe<tRef0Values.size(); iRe++)
  {
    for(unsigned int iIm=0; iIm<tImf0Values.size(); iIm++)
    {
      for(unsigned int id0=0; id0<td0Values.size(); id0++)
      {
        tFitGen->SetScattParamStartValues(tRef0Values[iRe],tImf0Values[iIm],td0Values[id0]);
        tFitGen->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);
        tChi2 = tFitGen->GetChi2();

        iCall++;
        cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
        cout << "Finished iCall = " << iCall << " of " << tTotalCalls << " total " << endl;
        cout << "iRe = " << iRe << " of " << tRef0Values.size()-1 << " total" << endl;
        cout << "iIm = " << iIm << " of " << tImf0Values.size()-1 << " total" << endl;
        cout << "id0 = " << id0 << " of " << td0Values.size()-1 << " total" << endl;
        cout << "\tPREVIOUS tGlobalChi2 = " << std::setprecision(30) << tGlobalChi2 << endl;

        if(fabs(tChi2-tGlobalChi2) < tTolerance)
        {
          cout << "We have a TIE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
          vector<int> tTempVec;
          tTempVec.push_back(tWinningIndices[0]);
          tTempVec.push_back(tWinningIndices[1]);
          tTempVec.push_back(tWinningIndices[2]);
          tTempVec.push_back(tChi2);

          tTyingWinningIndices.push_back(tTempVec);
        }
        else if(tChi2 < tGlobalChi2)
        {
          cout << "We have a NEW CHAMPION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
          tGlobalChi2 = tChi2;
          tWinningIndices[0] = iRe;
          tWinningIndices[1] = iIm;
          tWinningIndices[2] = id0;

          tTyingWinningIndices.clear();
        }
        else cout << "We have a LOSER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

        cout << "\ttChi2                = " << std::setprecision(30) << tChi2 << endl;
        cout << "\tCURRENT tGlobalChi2  = " << std::setprecision(30) << tGlobalChi2 << endl;
        cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
      }
    }
  }

  cout << "tTolerance = " << tTolerance << endl;
  cout << "tGlobalChi2 = " << tGlobalChi2 << endl;
  cout << "\t Winning values:" << endl;
  for(unsigned int i=0; i<tWinningIndices.size(); i++)
  {
    cout << std::setprecision(6) << "\t\t" << i << ": Winning index = " << tWinningIndices[i] << "\t Winning Value = " << tAllValues[i][tWinningIndices[i]] << endl;
  }
  cout << endl;

  if(tTyingWinningIndices.size() != 0)
  {
    cout << "HOWEVER, there are some other initial values which result in the same Chi2 value" << endl;
    for(unsigned int iWinner=0; iWinner<tTyingWinningIndices.size(); iWinner++)
    {
      for(unsigned int iVal=0; iVal<tTyingWinningIndices[iWinner].size(); iVal++)
      {
        if(iVal<3) cout << std::setprecision(6) << "\t\t" << iVal << ": Winning index = " << tTyingWinningIndices[iWinner][iVal] << "\t Winning Value = " << tAllValues[iVal][tTyingWinningIndices[iWinner][iVal]] << endl;
        else cout << std::setprecision(30) << "\t\t tChi2 = " << tTyingWinningIndices[iWinner][iVal] << endl;
      }
    }
  }
  cout << endl;


//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
