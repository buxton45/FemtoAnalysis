///////////////////////////////////////////////////////////////////////////
// DataAndModel:                                                         //
///////////////////////////////////////////////////////////////////////////

#include "DataAndModel.h"

#ifdef __ROOT__
ClassImp(DataAndModel)
#endif



//________________________________________________________________________________________________________________
DataAndModel::DataAndModel(Analysis *aAnalysisData, Analysis *aAnalysisModel, double aMinNorm, double aMaxNorm, int aRebin) :
  fAnalysisData(aAnalysisData),
  fAnalysisModel(aAnalysisModel),
  fAnalysisType(aAnalysisData->GetAnalysisType()),


  fKStarCfUncorrected(0),
  fKStarCfCorrectedwTrueHist(0),
  fKStarCfCorrectedwTrueFit(0),
  fKStarCfCorrectedwFakeHist(0),
  fKStarCfCorrectedwFakeFit(0),

  fKStarCfTrue(0),
  fKStarCfTrueIdeal(0),
  fKStarCfFake(0),
  fKStarCfFakeIdeal(0),

  fTrueCorrectionHist(0),
  fTrueCorrectionFit(0),

  fFakeCorrectionHist(0),
  fFakeCorrectionFit(0)

{
  //constructor

  //First, make sure everything I need is built
  fAnalysisData->BuildKStarHeavyCf(aMinNorm,aMaxNorm,aRebin);

  fAnalysisModel->BuildModelCfTrueIdealCfTrueRatio(aMinNorm,aMaxNorm,aRebin);
  fAnalysisModel->BuildModelCfFakeIdealCfFakeRatio(aMinNorm,aMaxNorm,aRebin);
  fAnalysisModel->FitModelCfTrueIdealCfTrueRatio();
  fAnalysisModel->FitModelCfFakeIdealCfFakeRatio();

  //Now, grab/set everything I need
  fKStarCfUncorrected = (TH1*)fAnalysisData->GetKStarHeavyCf()->GetHeavyCfClone();

  fKStarCfTrue = (TH1*)fAnalysisModel->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
  fKStarCfTrueIdeal = (TH1*)fAnalysisModel->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();

  fKStarCfFake = (TH1*)fAnalysisModel->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
  fKStarCfFakeIdeal = (TH1*)fAnalysisModel->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();


  fTrueCorrectionHist = (TH1*)fAnalysisModel->GetModelCfTrueIdealCfTrueRatio();
  fTrueCorrectionFit = (TF1*)fAnalysisModel->GetMomResFit();
  fFakeCorrectionHist = (TH1*)fAnalysisModel->GetModelCfFakeIdealCfFakeRatio();
  fFakeCorrectionFit = (TF1*)fAnalysisModel->GetMomResFitFake();

  //------------------------------------
  fKStarCfCorrectedwTrueHist = (TH1*)fAnalysisData->GetKStarHeavyCf()->GetHeavyCfClone();
  //fKStarCfCorrectedwTrueHist->SetNameTitle("fKStarCfCorrectedwTrueHist","fKStarCfCorrectedwTrueHist");
  fKStarCfCorrectedwTrueHist->Multiply(fTrueCorrectionHist);

  fKStarCfCorrectedwTrueFit = (TH1*)fAnalysisData->GetKStarHeavyCf()->GetHeavyCfClone();
  //fKStarCfCorrectedwTrueFit->SetNameTitle("fKStarCfCorrectedwTrueFit","fKStarCfCorrectedwTrueFit");
  fKStarCfCorrectedwTrueFit->Multiply(fTrueCorrectionFit);

  fKStarCfCorrectedwFakeHist = (TH1*)fAnalysisData->GetKStarHeavyCf()->GetHeavyCfClone();
  //fKStarCfCorrectedwFakeHist->SetNameTitle("fKStarCfCorrectedwFakeHist","fKStarCfCorrectedwFakeHist");
  fKStarCfCorrectedwFakeHist->Multiply(fFakeCorrectionHist);

  fKStarCfCorrectedwFakeFit = (TH1*)fAnalysisData->GetKStarHeavyCf()->GetHeavyCfClone();
  //fKStarCfCorrectedwFakeFit->SetNameTitle("fKStarCfCorrectedwFakeFit","fKStarCfCorrectedwFakeFit");
  fKStarCfCorrectedwFakeFit->Multiply(fFakeCorrectionFit);

}



//________________________________________________________________________________________________________________
DataAndModel::~DataAndModel()
{
  //destructor
}

//________________________________________________________________________________________________________________
void DataAndModel::PrepareHistToPlot(TH1* aHist, double aMarkerStyle, double aColor)
{
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerColor(aColor);
  aHist->SetLineColor(aColor);


}



//________________________________________________________________________________________________________________
void DataAndModel::DrawAllCorrectedCfs(TPad* aPad)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  double Style1 = 20;
  double Style2 = 21;
  double Style3 = 22;
  double Style4 = 23;
  double Style5 = 29;

  double Color1 = 1;
  double Color2 = 2;
  double Color3 = 4;
  double Color4 = 7;
  double Color5 = 8;
  
  TH1* tKStarCfUncorrected = (TH1*)fKStarCfUncorrected->Clone();
    PrepareHistToPlot(tKStarCfUncorrected,Style1,Color1);

  TH1* tKStarCfCorrectedwTrueHist = (TH1*)fKStarCfCorrectedwTrueHist->Clone();
    PrepareHistToPlot(tKStarCfCorrectedwTrueHist,Style2,Color2);
  TH1* tKStarCfCorrectedwTrueFit = (TH1*)fKStarCfCorrectedwTrueFit->Clone();
    PrepareHistToPlot(tKStarCfCorrectedwTrueFit,Style3,Color3);
  TH1* tKStarCfCorrectedwFakeHist = (TH1*)fKStarCfCorrectedwFakeHist->Clone();
    PrepareHistToPlot(tKStarCfCorrectedwFakeHist,Style4,Color4);
  TH1* tKStarCfCorrectedwFakeFit = (TH1*)fKStarCfCorrectedwFakeFit->Clone();
    PrepareHistToPlot(tKStarCfCorrectedwFakeFit,Style5,Color5);

  TLegend *tLeg = new TLegend(0.60,0.12,0.89,0.32);
    tLeg->SetFillColor(0);
    tLeg->AddEntry(tKStarCfUncorrected,"UnCorrected","lp");
    tLeg->AddEntry(tKStarCfCorrectedwTrueHist,"CorrectedwTrueHist","lp");
    tLeg->AddEntry(tKStarCfCorrectedwTrueFit,"CorrectedwTrueFit","lp");
    tLeg->AddEntry(tKStarCfCorrectedwFakeHist,"CorrectedwFakeHist","lp");
    tLeg->AddEntry(tKStarCfCorrectedwFakeFit,"CorrectedwFakeFit","lp");

  
  double tYmin = 0.8;
  double tYmax = 1.2;

  tKStarCfUncorrected->GetYaxis()->SetRangeUser(tYmin,tYmax);
  tKStarCfUncorrected->DrawCopy();
  tKStarCfCorrectedwTrueHist->DrawCopy("same");
  tKStarCfCorrectedwTrueFit->DrawCopy("same");
  tKStarCfCorrectedwFakeHist->DrawCopy("same");
  tKStarCfCorrectedwFakeFit->DrawCopy("same");
  tLeg->Draw();

  tKStarCfUncorrected->DrawCopy("same");

}

//________________________________________________________________________________________________________________
void DataAndModel::DrawTrueCorrectionwFit(TPad* aPad)
{
  aPad->cd();
  //gStyle->SetOptStat(0);

  TH1* tTrueCorrectionHist = (TH1*)fTrueCorrectionHist->Clone();
  TF1* tTrueCorrectionFit = (TF1*)fTrueCorrectionFit->Clone();

  double tYmin = 0.8;
  double tYmax = 1.2;

  tTrueCorrectionHist->GetYaxis()->SetRangeUser(tYmin,tYmax);
  tTrueCorrectionHist->Draw();
  tTrueCorrectionFit->Draw("same");
}

//________________________________________________________________________________________________________________
void DataAndModel::DrawFakeCorrectionwFit(TPad* aPad)
{
  aPad->cd();
  //gStyle->SetOptStat(0);

  TH1* tFakeCorrectionHist = (TH1*)fFakeCorrectionHist->Clone();
  TF1* tFakeCorrectionFit = (TF1*)fFakeCorrectionFit->Clone();

  double tYmin = 0.992;
  double tYmax = 1.01;

  for(int i=1; i<=tFakeCorrectionHist->GetNbinsX(); i++) tFakeCorrectionHist->SetBinError(i,0.001);

  tFakeCorrectionHist->GetYaxis()->SetRangeUser(tYmin,tYmax);
  tFakeCorrectionHist->Draw();
  tFakeCorrectionFit->Draw("same");
}

//________________________________________________________________________________________________________________
TH1D* DataAndModel::MatchBinSize(CfHeavy* aHeavyCf, TH2* aTrueVsRecHist)
{
  double tBinWidthCf = aHeavyCf->GetHeavyCf()->GetBinWidth(1);
  double tBinWidthMatrixX = aTrueVsRecHist->GetXaxis()->GetBinWidth(1);
  double tBinWidthMatrixY = aTrueVsRecHist->GetYaxis()->GetBinWidth(1);

  if(tBinWidthCf != tBinWidthMatrixX)
  {
    int tRebinX = tBinWidthCf/tBinWidthMatrixX;
    if(tRebinX >= 1) aTrueVsRecHist->RebinX(tRebinX);
    else
    {
      tRebinX = tBinWidthMatrixX/tBinWidthCf;
      aHeavyCf->Rebin(tRebinX);
    }
  }

  tBinWidthCf = aHeavyCf->GetHeavyCf()->GetBinWidth(1);

  if(tBinWidthCf != tBinWidthMatrixY)
  {
    int tRebinY = tBinWidthCf/tBinWidthMatrixY;
    if(tRebinY >= 1) aTrueVsRecHist->RebinY(tRebinY);
    else
    {
      tRebinY = tBinWidthMatrixY/tBinWidthCf;
      aHeavyCf->Rebin(tRebinY);
    }
  }

  TH1D *tReturnHist = (TH1D*)aHeavyCf->GetHeavyCfClone();
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* DataAndModel::GetKStarCorrectedwMatrix(int aMethod, KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh, bool aGetCorrectionFactorInstead)
{
  TString tReturnName = "KStarCfCorrectedw";
  TString tNameTrue = "KStarCfTrue";
  TString tNameRec = "KStarCfRec";

  TString Method0Tag = "wMatrix_" + TString(cAnalysisBaseTags[fAnalysisData->GetAnalysisType()]);
  TString Method1Tag = "wMatrixFit_" + TString(cAnalysisBaseTags[fAnalysisData->GetAnalysisType()]);

  if(aMethod==0) {tReturnName += Method0Tag; tNameTrue += Method0Tag; tNameRec += Method0Tag;}
  else if(aMethod==1) {tReturnName += Method1Tag; tNameTrue += Method1Tag; tNameRec += Method1Tag;}

  TH2* tMomResMatrix;
  fAnalysisModel->BuildAllModelKStarTrueVsRecTotal();

  if(aMethod==0)
  {
    tMomResMatrix = fAnalysisModel->GetModelKStarTrueVsRecTotal(aType);
  }
  else if(aMethod==1)
  {
    fAnalysisModel->BuildMomResMatrixFit(aType,aKStarLow,aKStarHigh);
    tMomResMatrix = fAnalysisModel->GetMomResMatrixFit(aType);
  }

  TH1D* tKStarCfUncorrected = MatchBinSize(fAnalysisData->GetKStarHeavyCfCopy(),tMomResMatrix);
  TH1D* tKStarCfTrueOG = MatchBinSize(fAnalysisModel->GetModelKStarHeavyCfFakeIdeal(),tMomResMatrix);

/*
  int tNbins = (aKStarHigh-aKStarLow)/tKStarCfUncorrected->GetBinWidth(1);

cout << "\t\t\t\t In new method: tNbins = " << tNbins << endl;

  TH1D* tKStarCorrected = new TH1D(tReturnName,tReturnName,tNbins,aKStarLow,aKStarHigh);
    tKStarCorrected->Sumw2();
  TH1D* tKStarRec = new TH1D(tNameRec,tNameRec,tNbins,aKStarLow,aKStarHigh);
    tKStarRec->Sumw2();
  TH1D* tKStarCfTrue = new TH1D(tNameTrue,tNameTrue,tNbins,aKStarLow,aKStarHigh);
    tKStarCfTrue->Sumw2();
*/

  int tNbins = (1.-0.)/tKStarCfUncorrected->GetBinWidth(1);

cout << "\t\t\t\t In new method: tNbins = " << tNbins << endl;

  TH1D* tKStarCorrected = new TH1D(tReturnName,tReturnName,tNbins,0.,1.);
    tKStarCorrected->Sumw2();
  TH1D* tKStarRec = new TH1D(tNameRec,tNameRec,tNbins,0.,1.);
    tKStarRec->Sumw2();
  TH1D* tKStarCfTrue = new TH1D(tNameTrue,tNameTrue,tNbins,0.,1.);
    tKStarCfTrue->Sumw2();

  for(int i=1; i<=tNbins; i++)
  {
    tKStarCorrected->SetBinContent(i,tKStarCfUncorrected->GetBinContent(i));
    tKStarCorrected->SetBinError(i,tKStarCfUncorrected->GetBinError(i));

    tKStarCfTrue->SetBinContent(i,tKStarCfTrueOG->GetBinContent(i));
    tKStarCfTrue->SetBinError(i,tKStarCfTrueOG->GetBinError(i));

//    tKStarCfTrue->SetBinContent(i,1);
//    tKStarCfTrue->SetBinError(i,.01);
  }

  for(int j=1; j<=tKStarCfTrue->GetNbinsX(); j++)
  {
    double tValue = 0.;
    assert(tKStarCfTrue->GetBinCenter(j) == tMomResMatrix->GetYaxis()->GetBinCenter(j));
    for(int i=1; i<=tMomResMatrix->GetNbinsX(); i++)
    {
      assert(tKStarCfTrue->GetBinCenter(i) == tMomResMatrix->GetXaxis()->GetBinCenter(i));
      assert(tKStarCfTrue->GetBinContent(i) > 0.);
      tValue += tKStarCfTrue->GetBinContent(i)*tMomResMatrix->GetBinContent(i,j);
    }
    tValue /= tMomResMatrix->Integral(1,tMomResMatrix->GetNbinsX(),j,j);
cout << "\t\t\t\t\t\t\t tValue = " << tValue << endl;
    tKStarRec->SetBinContent(j,tValue);
  }

  TString tNameCorrection = "Correction_";
    if(aMethod==0) tNameCorrection += Method0Tag;
    else if(aMethod==1) tNameCorrection += Method1Tag;
  TH1D* tCorrection = new TH1D();
  tCorrection = (TH1D*)tKStarCfTrue->Clone(tNameCorrection);
  tCorrection->Divide(tKStarRec);

  tKStarCorrected->Multiply(tCorrection);


  if(aMethod==0) 
  {
    tKStarCorrected->SetLineColor(2); tKStarCorrected->SetMarkerColor(2); tKStarCorrected->SetMarkerStyle(21);
    tCorrection->SetLineColor(2); tCorrection->SetMarkerColor(2); tCorrection->SetMarkerStyle(21);
  }
  else if(aMethod==1) 
  {
    tKStarCorrected->SetLineColor(8); tKStarCorrected->SetMarkerColor(8); tKStarCorrected->SetMarkerStyle(21);
    tCorrection->SetLineColor(8); tCorrection->SetMarkerColor(8); tCorrection->SetMarkerStyle(21);
  }

  if(aGetCorrectionFactorInstead) return tCorrection;
  else return tKStarCorrected;
}

//________________________________________________________________________________________________________________
TH1D* DataAndModel::GetKStarCorrectedwMatrixNumDenSmeared(KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh, bool aGetCorrectionFactorInstead)
{
  TString tReturnName = "KStarCfCorrected";
  TString tNameTrue = "KStarCfTrue";
  TString tNameRec = "KStarCfRec";
  TString tNameTag = "wMatrixNumDenSmeared_" + TString(cAnalysisBaseTags[fAnalysisData->GetAnalysisType()]);
  tReturnName += tNameTag; tNameTrue += tNameTag; tNameRec += tNameTag;

  TH2* tMomResMatrix;
  fAnalysisModel->BuildAllModelKStarTrueVsRecTotal();
  tMomResMatrix = fAnalysisModel->GetModelKStarTrueVsRecTotal(aType);

  TH1D* tKStarCfUncorrected = MatchBinSize(fAnalysisData->GetKStarHeavyCfCopy(),tMomResMatrix);
  TH1D* tKStarCfTrueOG = MatchBinSize(fAnalysisModel->GetModelKStarHeavyCfFakeIdeal(),tMomResMatrix);

  fAnalysisModel->BuildModelKStarHeavyCfFakeIdealSmeared(tMomResMatrix,0.32,0.4,2);
  TH1* tKStarCfRecOG = fAnalysisModel->GetModelKStarHeavyCfFakeIdealSmeared()->GetHeavyCfClone();

  int tNbins = (1.-0.)/tKStarCfUncorrected->GetBinWidth(1);



  TH1D* tKStarCorrected = new TH1D(tReturnName,tReturnName,tNbins,0.,1.);
    tKStarCorrected->Sumw2();
  TH1D* tKStarCfRec = new TH1D(tNameRec,tNameRec,tNbins,0.,1.);
    tKStarCfRec->Sumw2();
  TH1D* tKStarCfTrue = new TH1D(tNameTrue,tNameTrue,tNbins,0.,1.);
    tKStarCfTrue->Sumw2();

  for(int i=1; i<=tNbins; i++)
  {
    tKStarCorrected->SetBinContent(i,tKStarCfUncorrected->GetBinContent(i));
    tKStarCorrected->SetBinError(i,tKStarCfUncorrected->GetBinError(i));

    tKStarCfTrue->SetBinContent(i,tKStarCfTrueOG->GetBinContent(i));
    tKStarCfTrue->SetBinError(i,tKStarCfTrueOG->GetBinError(i));

    tKStarCfRec->SetBinContent(i,tKStarCfRecOG->GetBinContent(i));
    tKStarCfRec->SetBinError(i,tKStarCfRecOG->GetBinError(i));

//    tKStarCfTrue->SetBinContent(i,1);
//    tKStarCfTrue->SetBinError(i,.01);
  }


  TString tNameCorrection = "CorrectionNumDenSmeared_" + tNameTag;

  TH1D* tCorrection = new TH1D();
  tCorrection = (TH1D*)tKStarCfTrue->Clone(tNameCorrection);
  tCorrection->Divide(tKStarCfRec);

  tKStarCorrected->Multiply(tCorrection);
 
  tKStarCorrected->SetLineColor(4); tKStarCorrected->SetMarkerColor(4); tKStarCorrected->SetMarkerStyle(21);
  tCorrection->SetLineColor(4); tCorrection->SetMarkerColor(4); tCorrection->SetMarkerStyle(21);

  if(aGetCorrectionFactorInstead) return tCorrection;
  else return tKStarCorrected;
}




