#include "CorrFctnDirectYlmTherm.h"

#include <TMath.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <iostream>

using namespace std;

//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm::CorrFctnDirectYlmTherm(TString aFileLocation, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale) :
  CorrFctnDirectYlm(TString::Format("DirectYlmCf_%s", cAnalysisBaseTags[aAnType]).Data(), aMaxl, aNbins, aKStarMin, aKStarMax), 
  fFile(nullptr),
  fImpactParam(aImpactParam),
  fAnalysisType(aAnType),
  fSavedNameMod(""),
  fNewNameMod(""),
  fRebin(aRebin),
  fNumScale(aNumScale)

{
  fSavedNameMod = TString::Format("DirectYlmCf_%s", cAnalysisBaseTags[fAnalysisType]);
  fNewNameMod   = TString::Format("_b%d", fImpactParam);

  fFile = TFile::Open(aFileLocation);
  ReadFromThermFile(fRebin);
}

//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm::~CorrFctnDirectYlmTherm()
{

}


//________________________________________________________________________________________________________________
TH1* CorrFctnDirectYlmTherm::Get1dHisto(TString aHistoName, TString aNewName)
{
  TH1 *tHisto = (TH1*)fFile->Get(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1 *ReturnHisto = (TH1*)tHisto->Clone(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH1*)ReturnHisto;
}



//________________________________________________________________________________________________________________
TH2* CorrFctnDirectYlmTherm::Get2dHisto(TString aHistoName, TString aNewName)
{
  TH2 *tHisto = (TH2*)fFile->Get(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH2 *ReturnHisto = (TH2*)tHisto->Clone(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH2*)ReturnHisto;
}

//________________________________________________________________________________________________________________
TH3* CorrFctnDirectYlmTherm::Get3dHisto(TString aHistoName, TString aNewName)
{
  TH3 *tHisto = (TH3*)fFile->Get(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "3dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH3 *ReturnHisto = (TH3*)tHisto->Clone(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH3*)ReturnHisto;
}



//________________________________________________________________________________________________________________
void CorrFctnDirectYlmTherm::ReadFromThermFile(int aRebin)
{
  cout << "Reading in numerators and denominators" << endl;
  cout << "Reading function " << fSavedNameMod << fNewNameMod << endl;
  char bufname[200];
  for (int ihist=0; ihist<maxjm; ihist++) {
    sprintf(bufname, "NumReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], fSavedNameMod.Data());
    if (numsreal[ihist]) delete numsreal[ihist];
    cout << "Getting " << bufname << endl;
    numsreal[ihist] = new TH1D(*((TH1D *) Get1dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));

    sprintf(bufname, "NumImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], fSavedNameMod.Data());
    if (numsimag[ihist]) delete numsimag[ihist];
    numsimag[ihist] = new TH1D(*((TH1D *) Get1dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));

    sprintf(bufname, "DenReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], fSavedNameMod.Data());
    if (densreal[ihist]) delete densreal[ihist];
    densreal[ihist] = new TH1D(*((TH1D *) Get1dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));

    sprintf(bufname, "DenImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], fSavedNameMod.Data());
    if (densimag[ihist]) delete densimag[ihist];
    densimag[ihist] = new TH1D(*((TH1D *) Get1dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));

    if(aRebin != 1)
    {
      numsreal[ihist]->Rebin(aRebin);
      numsimag[ihist]->Rebin(aRebin);
      densreal[ihist]->Rebin(aRebin);
      densimag[ihist]->Rebin(aRebin);
    }
  }

  if (covnum) delete covnum;
  sprintf(bufname, "CovNum%s", fSavedNameMod.Data());
  if (Get3dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data())))
    covnum = new TH3D (*((TH3D *) Get3dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));
  else
    covnum = 0;

  if (covden) delete covden;
  sprintf(bufname, "CovDen%s", fSavedNameMod.Data());
  if (Get3dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data())))
    covden = new TH3D (*((TH3D *) Get3dHisto(bufname, TString::Format("%s%s", bufname, fNewNameMod.Data()))));
  else
    covden = 0;

  if(aRebin != 1)  //for covariance 3d histograms, only x-axis gets rebinned
  {
    covnum->RebinX(aRebin);
    covden->RebinX(aRebin);
    if(covcfc) covcfc->RebinX(aRebin);
  }

  if ((covnum) && (covden)) {
    cout << "Unpacking covariance matrices from file " << endl;
    UnpackCovariances();
  }
  else {

    cout << "Creating fake covariance matrices" << endl;

    for (int ibin=1; ibin<=numsreal[0]->GetNbinsX(); ibin++) {
      double nent = numsreal[0]->GetEntries();
      double nentd = densreal[0]->GetEntries();
      for (int ilmx=0; ilmx<GetMaxJM(); ilmx++) {
	for (int ilmy=0; ilmy<GetMaxJM(); ilmy++) {
	  double t1t2rr = numsreal[ilmx]->GetBinContent(ibin)*numsreal[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ri = numsreal[ilmx]->GetBinContent(ibin)*numsimag[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ir = numsimag[ilmx]->GetBinContent(ibin)*numsreal[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ii = numsimag[ilmx]->GetBinContent(ibin)*numsimag[ilmy]->GetBinContent(ibin)/nent/nent;
	  if (ilmx == ilmy) {
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nent*(TMath::Power(numsreal[ilmx]->GetBinError(ibin)/nent,2)*(nent-1) + t1t2rr);
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nent*t1t2ri;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nent*t1t2ir;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nent*(TMath::Power(numsimag[ilmx]->GetBinError(ibin)/nent,2)*(nent-1) + t1t2rr);
	  }
	  else {
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nent*t1t2rr;
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nent*t1t2ri;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nent*t1t2ir;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nent*t1t2ii;
	  }
	  t1t2rr = densreal[ilmx]->GetBinContent(ibin)*densreal[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ri = densreal[ilmx]->GetBinContent(ibin)*densimag[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ir = densimag[ilmx]->GetBinContent(ibin)*densreal[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ii = densimag[ilmx]->GetBinContent(ibin)*densimag[ilmy]->GetBinContent(ibin)/nentd/nentd;

	  covmden[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nentd*t1t2rr;
	  covmden[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nentd*t1t2ri;
	  covmden[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nentd*t1t2ir;
	  covmden[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nentd*t1t2ii;
	}
      }
    }
  }

  //Add fNewNameMod to all Cfs etc (already done on Nums and Dens above)
  for(int ihist=0; ihist<maxjm; ihist++)
  {
    cfctreal[ihist]->SetName(TString::Format("%s%s", cfctreal[ihist]->GetName(), fNewNameMod.Data()));
    cfctreal[ihist]->SetTitle(TString::Format("%s%s", cfctreal[ihist]->GetTitle(), fNewNameMod.Data()));

    cfctimag[ihist]->SetName(TString::Format("%s%s", cfctimag[ihist]->GetName(), fNewNameMod.Data()));
    cfctimag[ihist]->SetTitle(TString::Format("%s%s", cfctimag[ihist]->GetTitle(), fNewNameMod.Data()));

    if(aRebin != 1)
    {
      cfctreal[ihist]->Rebin(aRebin);
      cfctimag[ihist]->Rebin(aRebin);
    }
  }
  binctn->SetName(TString::Format("%s%s", binctn->GetName(), fNewNameMod.Data()));
  binctn->SetTitle(TString::Format("%s%s", binctn->GetTitle(), fNewNameMod.Data()));

  binctd->SetName(TString::Format("%s%s", binctd->GetName(), fNewNameMod.Data()));
  binctd->SetTitle(TString::Format("%s%s", binctd->GetTitle(), fNewNameMod.Data()));

  if(aRebin != 1)
  {
    binctn->Rebin(aRebin);
    binctd->Rebin(aRebin);
  }

  // Recalculating the correlation functions
  Finish();
}



//________________________________________________________________________________________________________________
TH1D* CorrFctnDirectYlmTherm::GetYlmHist(YlmComponent aComponent, YlmHistType aHistType, int al, int am)
{
  if     (aComponent==kYlmReal && aHistType==kYlmNum) return (TH1D*)GetNumRealHist(al, am)->Clone();
  else if(aComponent==kYlmImag && aHistType==kYlmNum) return (TH1D*)GetNumImagHist(al, am)->Clone();

  else if(aComponent==kYlmReal && aHistType==kYlmDen) return (TH1D*)GetDenRealHist(al, am)->Clone();
  else if(aComponent==kYlmImag && aHistType==kYlmDen) return (TH1D*)GetDenImagHist(al, am)->Clone();

  else if(aComponent==kYlmReal && aHistType==kYlmCf) return (TH1D*)GetCfnRealHist(al, am)->Clone();
  else if(aComponent==kYlmImag && aHistType==kYlmCf) return (TH1D*)GetCfnImagHist(al, am)->Clone();
  else assert(0);

  return nullptr;
}











