#include "CorrFctnDirectYlmHeavy.h"

#include <TMath.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <iostream>

using namespace std;

//________________________________________________________________________________________________________________
CorrFctnDirectYlmHeavy::CorrFctnDirectYlmHeavy(vector<CorrFctnDirectYlmLite*> &aYlmCfLiteCollection) :
  fYlmCfLiteCollection(aYlmCfLiteCollection)
{

}



//________________________________________________________________________________________________________________
CorrFctnDirectYlmHeavy::~CorrFctnDirectYlmHeavy()
{

}

//________________________________________________________________________________________________________________
TH1D* CorrFctnDirectYlmHeavy::GetYlmCfnHist(YlmComponent aComponent, int al, int am)
{
  double tScale = 0.;
  double tTempScale = 0.;

  TH1D* tReturnCf;
  if(aComponent==kYlmReal) tReturnCf = (TH1D*)fYlmCfLiteCollection[0]->GetCfnRealHist(al, am)->Clone();
  else                     tReturnCf = (TH1D*)fYlmCfLiteCollection[0]->GetCfnImagHist(al, am)->Clone();
  if(!tReturnCf->GetSumw2N()) {tReturnCf->Sumw2();}

  tReturnCf->SetTitle(TString::Format("%sHeavy", tReturnCf->GetTitle()));
  tReturnCf->SetName(TString::Format("%sHeavy", tReturnCf->GetName()));


  tTempScale = fYlmCfLiteCollection[0]->GetNumScale();
  tScale += tTempScale;
  tReturnCf->Scale(tTempScale);

  for(unsigned int i=1; i<fYlmCfLiteCollection.size(); i++)
  {
    tTempScale = fYlmCfLiteCollection[i]->GetNumScale();

    if(aComponent==kYlmReal) tReturnCf->Add((TH1D*)fYlmCfLiteCollection[i]->GetCfnRealHist(al, am)->Clone(),tTempScale);
    else                     tReturnCf->Add((TH1D*)fYlmCfLiteCollection[i]->GetCfnImagHist(al, am)->Clone(),tTempScale);
    tScale += tTempScale;
  }

  tReturnCf->Scale(1./tScale);
  return tReturnCf;
}



