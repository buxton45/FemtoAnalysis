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
                                       TString aDirNameModifierBase2, vector<double> &aModifierValues2) :
  fFileLocationBase(aFileLocationBase),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),
  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2),

  fAnalyses()

{
  fAnalyses.reserve(fModifierValues1.size());
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    fAnalyses.emplace_back(fFileLocationBase,fAnalysisType,fCentralityType,kTrainSys,2,tDirNameModifier);
    fAnalyses[i].BuildKStarHeavyCf();
  }


}

//________________________________________________________________________________________________________________
SystematicAnalysis::SystematicAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1) :
  fFileLocationBase(aFileLocationBase),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(0),
  fModifierValues1(aModifierValues1),
  fModifierValues2(0),

  fAnalyses()

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  SystematicAnalysis(aFileLocationBase,aAnalysisType,aCentralityType,aDirNameModifierBase1,aModifierValues1,fDirNameModifierBase2,fModifierValues2);
}


//________________________________________________________________________________________________________________
SystematicAnalysis::~SystematicAnalysis()
{

}


//________________________________________________________________________________________________________________
int SystematicAnalysis::Factorial(int aInput)
{
  if(aInput == 0) return 1;

  return aInput*Factorial(aInput-1);
}


//________________________________________________________________________________________________________________
int SystematicAnalysis::nChoosek(int aN, int aK)
{
  return Factorial(aN)/(Factorial(aK)*Factorial(aN-aK));
}

//________________________________________________________________________________________________________________
TH1* SystematicAnalysis::GetDiffHist(TH1* aHist1, TH1* aHist2)
{
  assert(aHist1->GetNbinsX() == aHist2->GetNbinsX());

  TH1* tDiffHist = (TH1*)aHist1->Clone("tDiffHist");
  tDiffHist->Add(aHist2,-1);

  double tBinError;
  for(int i=0; i<tDiffHist->GetNbinsX(); i++)
  {
    tBinError = sqrt(abs(pow(aHist1->GetBinError(i),2) - pow(aHist2->GetBinError(i),2)));
    tDiffHist->SetBinError(i,tBinError);
  }

  return tDiffHist;
}

//________________________________________________________________________________________________________________
double SystematicAnalysis::GetPValueCorrelated(TH1* aHist1, TH1* aHist2)
{
  assert(aHist1->GetNbinsX() == aHist2->GetNbinsX());

  TH1* tDiffHist = GetDiffHist(aHist1,aHist2);

  double tMinFit = 0.;
  double tMaxFit = 1.;
  TF1* tFit = new TF1("tFit","0",tMinFit,tMaxFit);

  double tChi2 = tDiffHist->Chisquare(tFit);

  int tBinLow = tDiffHist->FindBin(tMinFit);
  int tBinHigh = tDiffHist->FindBin(tMaxFit);
    if(tDiffHist->GetBinLowEdge(tBinHigh) == tMaxFit) tBinHigh--;
  int tNbins = tBinHigh-tBinLow+1;
  double tPValue = TMath::Prob(tChi2,tNbins);

  return tPValue;
}


//________________________________________________________________________________________________________________
void SystematicAnalysis::GetAllPValues()
{
  cout << "AnalysisType = " << cAnalysisBaseTags[fAnalyses[0].GetAnalysisType()] << endl;
  cout << "CentralityType = " << cPrettyCentralityTags[fAnalyses[0].GetCentralityType()] << endl << endl;

  double tPVal = 0.;
  TString tCutVal1a, tCutVal1b, tCutVal1Tot;
  TString tCutVal2a, tCutVal2b, tCutVal2Tot;
  for(unsigned int i=0; i<fAnalyses.size(); i++)
  {

    for(unsigned int j=i+1; j<fAnalyses.size(); j++)
    {
      TH1* tHist1 = fAnalyses[i].GetKStarHeavyCf()->GetHeavyCf();
      TH1* tHist2 = fAnalyses[j].GetKStarHeavyCf()->GetHeavyCf();

      tPVal = GetPValueCorrelated(tHist1,tHist2);

      tCutVal1a = fDirNameModifierBase1;
        tCutVal1a.Remove(TString::kBoth,'_');
        tCutVal1a += TString::Format(" = %0.6f",fModifierValues1[i]);

      tCutVal2a = fDirNameModifierBase1;
        tCutVal2a.Remove(TString::kBoth,'_');
        tCutVal2a += TString::Format(" = %0.6f",fModifierValues1[j]);


      tCutVal1Tot = tCutVal1a;
      tCutVal2Tot = tCutVal2a;

      if(!fDirNameModifierBase2.IsNull())
      {
        tCutVal1b = fDirNameModifierBase2;
          tCutVal1b.Remove(TString::kBoth,'_');
          tCutVal1b += TString::Format(" = %0.6f",fModifierValues2[i]);

        tCutVal2b = fDirNameModifierBase2;
          tCutVal2b.Remove(TString::kBoth,'_');
          tCutVal2b += TString::Format(" = %0.6f",fModifierValues2[j]);

        tCutVal1Tot += TString::Format(" and %s",tCutVal1b.Data());
        tCutVal2Tot += TString::Format(" and %s",tCutVal2b.Data());
      }

      cout << tCutVal1Tot << endl;
      cout << tCutVal2Tot << endl;
      cout << "p value = " << std::setprecision(6) << tPVal << endl << endl;
    }
  }
}


//________________________________________________________________________________________________________________
void SystematicAnalysis::DrawAll()
{
  TString tCanTitle = TString::Format("canSys%s%s",cAnalysisBaseTags[fAnalyses[0].GetAnalysisType()],cCentralityTags[fAnalyses[0].GetCentralityType()]);
  TCanvas* tReturnCan = new TCanvas(tCanTitle,tCanTitle);
    tReturnCan->Divide(1,fAnalyses.size());

  for(unsigned int i=0; i<fAnalyses.size(); i++)
  {
    tReturnCan->cd(i+1);
    TH1* tHist1 = fAnalyses[i].GetKStarHeavyCf()->GetHeavyCf();
    tHist1->Draw();
  }

}

//________________________________________________________________________________________________________________
void SystematicAnalysis::DrawAllDiffs()
{
  gStyle->SetOptFit();

  TString tCanTitle = TString::Format("canSysDiffs%s%s",cAnalysisBaseTags[fAnalyses[0].GetAnalysisType()],cCentralityTags[fAnalyses[0].GetCentralityType()]);
  TCanvas* tReturnCan = new TCanvas(tCanTitle,tCanTitle);
  int tNPads = nChoosek(fAnalyses.size(),2);
    tReturnCan->Divide(1,tNPads);

  int tPad=1;
  for(unsigned int i=0; i<fAnalyses.size(); i++)
  {
    for(unsigned int j=i+1; j<fAnalyses.size(); j++)
    {
      tReturnCan->cd(tPad);
      TH1* tHist1 = fAnalyses[i].GetKStarHeavyCf()->GetHeavyCf();
      TH1* tHist2 = fAnalyses[j].GetKStarHeavyCf()->GetHeavyCf();
      TH1* tDiffHist = GetDiffHist(tHist1,tHist2);

      tDiffHist->Draw();
      tPad++;
    }
  }
}


