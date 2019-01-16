#include <iostream>
#include <iomanip>

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "CorrFctnDirectYlmTherm.h"



//_________________________________________________________________________________________
void DrawSHCfComponent(TPad* aPad, Analysis* aAnaly, YlmComponent aComponent, int al, int am, int aRebin/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
{
  aPad->cd();
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.025);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.3;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.07;
  }
  else
  {
    tYLow = -0.03;
    tYHigh = 0.03;
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  //--------------------------------------------------------------
//  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  TH1D* tSHCf = (TH1D*)aAnaly->GetYlmCfnHist(aComponent, al, am, aRebin);
  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  tSHCf->Draw();

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(cAnalysisRootTags[aAnaly->GetAnalysisType()]);
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();

}

//_________________________________________________________________________________________
void DrawSHCfComponent(TPad* aPad, Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int al, int am, int aRebin/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
{
  aPad->cd();
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.025);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.3;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.07;
  }
  else
  {
    tYLow = -0.03;
    tYHigh = 0.03;
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  //--------------------------------------------------------------
  vector<CorrFctnDirectYlmLite*> tYlmLiteCollAn = aAnaly->GetYlmCfHeavy(aRebin)->GetYlmCfLiteCollection();
  vector<CorrFctnDirectYlmLite*> tYlmLiteCollConjAn = aConjAnaly->GetYlmCfHeavy(aRebin)->GetYlmCfLiteCollection();

  double tOverallScale = 0.;
  TH1D* tSHCf = tYlmLiteCollAn[0]->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf->Scale(tYlmLiteCollAn[0]->GetNumScale());
  tOverallScale += tYlmLiteCollAn[0]->GetNumScale();

  if(!tSHCf->GetSumw2N()) tSHCf->Sumw2();

  for(unsigned int i=1; i<tYlmLiteCollAn.size(); i++)
  {
    tSHCf->Add(tYlmLiteCollAn[i]->GetYlmHist(aComponent, kYlmCf, al, am), tYlmLiteCollAn[i]->GetNumScale());
    tOverallScale += tYlmLiteCollAn[i]->GetNumScale();
  }
  for(unsigned int i=0; i<tYlmLiteCollConjAn.size(); i++)
  {
    tSHCf->Add(tYlmLiteCollConjAn[i]->GetYlmHist(aComponent, kYlmCf, al, am), tYlmLiteCollConjAn[i]->GetNumScale());
    tOverallScale += tYlmLiteCollConjAn[i]->GetNumScale();
  }
  tSHCf->Scale(1./tOverallScale);


  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  tSHCf->Draw();

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(TString::Format("%s & %s", cAnalysisRootTags[aAnaly->GetAnalysisType()], cAnalysisRootTags[aConjAnaly->GetAnalysisType()]));
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();

}


//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm* GetYlmCfTherm(TString aFileLocation, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale=0.)
{
  CorrFctnDirectYlmTherm* tCfYlmTherm = new CorrFctnDirectYlmTherm(aFileLocation, aImpactParam, aAnType, aMaxl, aNbins, aKStarMin, aKStarMax, aRebin, aNumScale);
  return tCfYlmTherm;
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm, YlmComponent aComponent, int al, int am/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, int aMarkerStyle=20, int aColor=1)
{
  aPad->cd();

  TH1D* tSHCf = (TH1D*)aCfYlmTherm->GetYlmHist(aComponent, kYlmCf, al, am);

  tSHCf->SetMarkerStyle(aMarkerStyle);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(aColor);
  tSHCf->SetLineColor(aColor);

  tSHCf->Draw("same");
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmThermAn, CorrFctnDirectYlmTherm* aCfYlmThermConjAn, YlmComponent aComponent, int al, int am/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, int aMarkerStyle=20, int aColor=1)
{
  aPad->cd();

  TH1D* tSHCf = (TH1D*)aCfYlmThermAn->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCfConjAn = (TH1D*)aCfYlmThermConjAn->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf->Add(tSHCfConjAn);
  tSHCf->Scale(0.5);

  tSHCf->SetMarkerStyle(aMarkerStyle);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(aColor);
  tSHCf->SetLineColor(aColor);

  tSHCf->Draw("same");
}

//_________________________________________________________________________________________
TCanvas* DrawFirstSixComponents(Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int aRebin)
{
  vector<TString> tRealOrImag{"Re", "Im"};
  TString tCanName = TString::Format("CanCfYlm%sFirstSixComps_%s%s", tRealOrImag[aComponent].Data(), cAnalysisBaseTags[aAnaly->GetAnalysisType()], cCentralityTags[aAnaly->GetCentralityType()]);
  TCanvas *tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->Divide(3,3);

  int tCan=0;
  for(unsigned int il=0; il<3; il++)
  {
    for(unsigned int im=0; im<=il; im++)
    {
      tCan = il*3 + im +1;
      DrawSHCfComponent((TPad*)tReturnCan->cd(tCan), aAnaly, aConjAnaly, aComponent, il, im, aRebin);
    }
  }

  return tReturnCan;
}



//_________________________________________________________________________________________
//*****************************************************************************************
//_________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;
  //------------------------------------
  TString tResultsDate = "20181205";
  AnalysisType tAnType = kLamKchP;
  AnalysisType tConjAnType;
  if((int)tAnType %2 == 0) tConjAnType = static_cast<AnalysisType>((int)tAnType+1);
  else                     tConjAnType = static_cast<AnalysisType>((int)tAnType-1);

  bool bCombineConjugates = true;
  bool bDrawThermCfs = true;
  bool bDrawFirstSix = true;
  bool bSaveFigures = false;

  int tl = 1;
  int tm = 1;
  YlmComponent tComponent = kYlmReal;

  double tMinNorm=0.32;
  double tMaxNorm=0.40;
  int tRebin=2;

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

//  TString tSaveDirectoryBase = tDirectoryBase;
  TString tSaveDirectoryBase = "/home/jesse/Analysis/Presentations/AliFemto/20190116/Figures/";
//-----------------------------------------------------------------------------

  Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, tAnRunType, 2, "", false);
  Analysis* tAnaly1030 = new Analysis(tFileLocationBase, tAnType, k1030, tAnRunType, 2, "", false);
  Analysis* tAnaly3050 = new Analysis(tFileLocationBase, tAnType, k3050, tAnRunType, 2, "", false);

  Analysis* tConjAnaly0010 = new Analysis(tFileLocationBase, tConjAnType, k0010, tAnRunType, 2, "", false);
  Analysis* tConjAnaly1030 = new Analysis(tFileLocationBase, tConjAnType, k1030, tAnRunType, 2, "", false);
  Analysis* tConjAnaly3050 = new Analysis(tFileLocationBase, tConjAnType, k3050, tAnRunType, 2, "", false);

  //----------
  TString tCanNameAll, tCanName0010;
  TCanvas *tCanAll, *tCan0010;

  if(!bCombineConjugates)
  {
    tCanNameAll = TString::Format("CanCfYlmReC00C11_%s_All", cAnalysisBaseTags[tAnType]);
    tCanAll = new TCanvas(tCanNameAll, tCanNameAll);
    tCanAll->Divide(2, 3);

    DrawSHCfComponent((TPad*)tCanAll->cd(1), tAnaly0010, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(2), tAnaly0010, tComponent, 1, 1, tRebin);

    DrawSHCfComponent((TPad*)tCanAll->cd(3), tAnaly1030, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(4), tAnaly1030, tComponent, 1, 1, tRebin);

    DrawSHCfComponent((TPad*)tCanAll->cd(5), tAnaly3050, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(6), tAnaly3050, tComponent, 1, 1, tRebin);

    //----------

    tCanName0010 = TString::Format("CanCfYlmReC00C11_%s_0010", cAnalysisBaseTags[tAnType]);
    tCan0010 = new TCanvas(tCanName0010, tCanName0010);
    tCan0010->Divide(2, 1);

    DrawSHCfComponent((TPad*)tCan0010->cd(1), tAnaly0010, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCan0010->cd(2), tAnaly0010, tComponent, 1, 1, tRebin);
  }
  else
  {
    tCanNameAll = TString::Format("CanCfYlmReC00C11_%s%s_All", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    tCanAll = new TCanvas(tCanNameAll, tCanNameAll);
    tCanAll->Divide(2, 3);

    DrawSHCfComponent((TPad*)tCanAll->cd(1), tAnaly0010, tConjAnaly0010, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(2), tAnaly0010, tConjAnaly0010, tComponent, 1, 1, tRebin);

    DrawSHCfComponent((TPad*)tCanAll->cd(3), tAnaly1030, tConjAnaly1030, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(4), tAnaly1030, tConjAnaly1030, tComponent, 1, 1, tRebin);

    DrawSHCfComponent((TPad*)tCanAll->cd(5), tAnaly3050, tConjAnaly3050, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCanAll->cd(6), tAnaly3050, tConjAnaly3050, tComponent, 1, 1, tRebin);

    //----------

    tCanName0010 = TString::Format("CanCfYlmReC00C11_%s%s_0010", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    tCan0010 = new TCanvas(tCanName0010, tCanName0010);
    tCan0010->Divide(2, 1);

    DrawSHCfComponent((TPad*)tCan0010->cd(1), tAnaly0010, tConjAnaly0010, tComponent, 0, 0, tRebin);
    DrawSHCfComponent((TPad*)tCan0010->cd(2), tAnaly0010, tConjAnaly0010, tComponent, 1, 1, tRebin);
  }

  if(bDrawThermCfs)
  {
    int tImpactParam = 2;
    TString aCfDescriptor = "Full";

    TString tFileNameBaseTherm = "CorrelationFunctions_wOtherPairs_BuildCfYlm";
    TString tFileNameModifierTherm = "";

    TString tFileNameTherm = TString::Format("%s%s.root", tFileNameBaseTherm.Data(), tFileNameModifierTherm.Data());

    TString tFileDirTherm = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    TString tFileLocationTherm = TString::Format("%s%s", tFileDirTherm.Data(), tFileNameTherm.Data());

    CorrFctnDirectYlmTherm* tCfYlmThermAn = GetYlmCfTherm(tFileLocationTherm, tImpactParam, tAnType, 2, 300, 0., 3., tRebin);
    CorrFctnDirectYlmTherm* tCfYlmThermConjAn = GetYlmCfTherm(tFileLocationTherm, tImpactParam, tConjAnType, 2, 300, 0., 3., tRebin);

    if(!bCombineConjugates)
    {
      DrawSHCfThermComponent((TPad*)tCanAll->cd(1), tCfYlmThermAn, tComponent, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCanAll->cd(2), tCfYlmThermAn, tComponent, 1, 1, 29, kOrange);
      //----------
      DrawSHCfThermComponent((TPad*)tCan0010->cd(1), tCfYlmThermAn, tComponent, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCan0010->cd(2), tCfYlmThermAn, tComponent, 1, 1, 29, kOrange);
    }
    else
    {
      DrawSHCfThermComponent((TPad*)tCanAll->cd(1), tCfYlmThermAn, tCfYlmThermConjAn, tComponent, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCanAll->cd(2), tCfYlmThermAn, tCfYlmThermConjAn, tComponent, 1, 1, 29, kOrange);
      //----------
      DrawSHCfThermComponent((TPad*)tCan0010->cd(1), tCfYlmThermAn, tCfYlmThermConjAn, tComponent, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCan0010->cd(2), tCfYlmThermAn, tCfYlmThermConjAn, tComponent, 1, 1, 29, kOrange);
    }
  }

  //-----------------------------------------------------------------------------
  if(bDrawFirstSix)
  {
    TCanvas* tCanFirstSixReal = DrawFirstSixComponents(tAnaly0010, tConjAnaly0010, kYlmReal, tRebin);
    TCanvas* tCanFirstSixImag = DrawFirstSixComponents(tAnaly0010, tConjAnaly0010, kYlmImag, tRebin);
    if(bSaveFigures)
    {
      tCanFirstSixReal->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCanFirstSixReal->GetName()));
      tCanFirstSixImag->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCanFirstSixImag->GetName()));
    }
  }



  if(bSaveFigures)
  {
    tCanAll->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCanNameAll.Data()));
    tCan0010->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCanName0010.Data()));
  }

/*
  Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, tAnRunType, 2, "", false);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();


  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  tTestCfn->Draw();
*/
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
