/*HistInfoPrinter.cxx			*/

#include "HistInfoPrinter.h"

#ifdef __ROOT__
ClassImp(HistInfoPrinter)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
HistInfoPrinter::HistInfoPrinter()

{

}



//________________________________________________________________________________________________________________
HistInfoPrinter::~HistInfoPrinter()
{

}

//________________________________________________________________________________________________________________
bool HistInfoPrinter::AreApproxEqual(double aVal1, double aVal2, double aPrecision)
{
  double tRel;
  if(aVal1==0. && aVal2==0.) return true;
  else if(aVal1==0. || aVal2==0.) tRel = fabs(aVal1-aVal2);
  else tRel = fabs(aVal1-aVal2)/fabs(aVal1);
  
  bool tAreSame = false;
  tAreSame = (tRel <= aPrecision) ? true : false;
  
  return tAreSame;
}


//________________________________________________________________________________________________________________
void HistInfoPrinter::PrintHistInfo(TH1* aHist, FILE* aOutput, double aXAxisLow, double aXAxisHigh)
{
  fprintf(aOutput, "aHist->GetName() = %s\n", aHist->GetName());
  fprintf(aOutput, "aHist->GetTitle() = %s\n", aHist->GetTitle());
  
  int tBinLow=1;
  int tBinHigh=aHist->GetNbinsX();
  if(!(aXAxisLow==0. && aXAxisHigh==0.))
  {
    tBinLow  = aHist->FindBin(aXAxisLow);
    tBinHigh = aHist->FindBin(aXAxisHigh);
  }  
  
  for(int i=tBinLow; i<=tBinHigh; i++)
  {
    fprintf(aOutput, "bin %d: X = %f \t Y = %f \t errY = %f\n", i, aHist->GetBinCenter(i), aHist->GetBinContent(i), aHist->GetBinError(i));
  }
}

//________________________________________________________________________________________________________________
void HistInfoPrinter::PrintHistInfowStatAndSyst(TH1* aHistStat, TH1* aHistSyst, FILE* aOutput, double aXAxisLow, double aXAxisHigh)
{
  fprintf(aOutput, "aHistStat->GetName() = %s\n", aHistStat->GetName());
  fprintf(aOutput, "aHistStat->GetTitle() = %s\n", aHistStat->GetTitle());  
  
  fprintf(aOutput, "aHistSyst->GetName() = %s\n", aHistSyst->GetName());
  fprintf(aOutput, "aHistSyst->GetTitle() = %s\n", aHistSyst->GetTitle());    

  assert(aHistStat->GetNbinsX()==aHistSyst->GetNbinsX());
  
  int tBinLow=1;
  int tBinHigh=aHistStat->GetNbinsX();
  if(!(aXAxisLow==0. && aXAxisHigh==0.))
  {
    tBinLow  = aHistStat->FindBin(aXAxisLow);
    tBinHigh = aHistStat->FindBin(aXAxisHigh);
  }
  
  for(int i=tBinLow; i<=tBinHigh; i++)
  {
    assert(aHistStat->GetBinCenter(i)==aHistSyst->GetBinCenter(i));
    //assert(aHistStat->GetBinContent(i)==aHistSyst->GetBinContent(i));
    assert(AreApproxEqual(aHistStat->GetBinContent(i), aHistSyst->GetBinContent(i)));
    fprintf(aOutput, "bin %d: X = %f \t Y = %f \t errY = %f \t syserrY = %f\n", i, aHistStat->GetBinCenter(i), aHistStat->GetBinContent(i), aHistStat->GetBinError(i), aHistSyst->GetBinError(i));
  }
}


