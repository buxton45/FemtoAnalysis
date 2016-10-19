///////////////////////////////////////////////////////////////////////////
// FitGenerator:                                                         //
///////////////////////////////////////////////////////////////////////////


#include "FitGenerator.h"

#ifdef __ROOT__
ClassImp(FitGenerator)
#endif


//GLOBAL!!!!!!!!!!!!!!!
LednickyFitter *GlobalFitter = NULL;

//______________________________________________________________________________
void GlobalFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalFitter->CalculateChi2PML(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, int aNPartialAnalysis, bool aIsTrainResults, CentralityType aCentralityType, FitGeneratorType aGeneratorType) :
  fContainsMC(false),
  fNAnalyses(0),
  fGeneratorType(aGeneratorType),
  fPairType(kLamK0), fConjPairType(kALamK0),
  fCentralityType(aCentralityType),

  fSharedAn(0),
  fLednickyFitter(0)

{
  switch(aAnalysisType) {
  case kLamK0:
  case kALamK0:
    fPairType = kLamK0;
    fConjPairType = kALamK0;
    break;

  case kLamKchP:
  case kALamKchM:
    fPairType = kLamKchP;
    fConjPairType = kALamKchM;
    break;

  case kLamKchM:
  case kALamKchP:
    fPairType = kLamKchM;
    fConjPairType = kALamKchP;
    break;

  default:
    cout << "Error in FitGenerator constructor, invalide aAnalysisType = " << aAnalysisType << " selected." << endl;
    assert(0);
  }

  vector<FitPairAnalysis*> tVecOfPairAn;
  switch(fCentralityType) {
  case k0010:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case k1030:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case k3050:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,fCentralityType,aNPartialAnalysis,aIsTrainResults));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case kMB:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k0010,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,k1030,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,k3050,aNPartialAnalysis,aIsTrainResults));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k0010,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,k1030,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,k3050,aNPartialAnalysis,aIsTrainResults));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k0010,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k0010,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,k1030,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,k1030,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fPairType,k3050,aNPartialAnalysis,aIsTrainResults));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,fConjPairType,k3050,aNPartialAnalysis,aIsTrainResults));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  default:
    cout << "Error in FitGenerator constructor, invalide aCentralityType = " << aCentralityType << " selected." << endl;
    assert(0);
  }


  fSharedAn = new FitSharedAnalyses(tVecOfPairAn);
  SetNAnalyses();

}


//________________________________________________________________________________________________________________
FitGenerator::~FitGenerator()
{

}

//________________________________________________________________________________________________________________
void FitGenerator::SetNAnalyses()
{
  switch(fCentralityType) {
  case k0010:
  case k1030:
  case k3050:
    switch(fGeneratorType) {
    case kPair:
    case kConjPair:
      fNAnalyses = 1;
      break;

    case kPairwConj:
      fNAnalyses = 2;
      break;

    default:
      cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fGeneratorType = " << fGeneratorType << endl;
    }
    break;


  case kMB:
    switch(fGeneratorType) {
    case kPair:
    case kConjPair:
      fNAnalyses = 3;
      break;

    case kPairwConj:
      fNAnalyses = 6;
      break;

    default:
      cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fGeneratorType = " << fGeneratorType << endl;
    }
    break;


  default:
    cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fCentralityType = " << fCentralityType << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
void FitGenerator::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetRangeUser(aMin,aMax);

  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}

//________________________________________________________________________________________________________________
void FitGenerator::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}



//________________________________________________________________________________________________________________
void FitGenerator::DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCf();
    SetupAxis(tCfToDraw->GetXaxis(),"k* (GeV/c)");
    SetupAxis(tCfToDraw->GetYaxis(),"C(k*)");

  tCfToDraw->GetXaxis()->SetRangeUser(aXmin,aXmax);
  tCfToDraw->GetYaxis()->SetRangeUser(aYmin,aYmax);

  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);

  tCfToDraw->Draw(aOption);

  TLine *line = new TLine(aXmin,1,aXmax,1);
  line->SetLineColor(14);
  line->Draw();
}

//________________________________________________________________________________________________________________
void FitGenerator::DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  gStyle->SetOptFit();

  gStyle->SetStatH(0.15);
  gStyle->SetStatW(0.30);

  gStyle->SetStatX(0.85);
  gStyle->SetStatY(0.60);


  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCf();
    SetupAxis(tCfToDraw->GetXaxis(),"k* (GeV/c)");
    SetupAxis(tCfToDraw->GetYaxis(),"C(k*)");

  tCfToDraw->GetXaxis()->SetRangeUser(aXmin,aXmax);
  tCfToDraw->GetYaxis()->SetRangeUser(aYmin,aYmax);

  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);

  tCfToDraw->Draw(aOption);

  TF1* tFit = fSharedAn->GetFitPairAnalysis(aPairAnNumber)->GetFit();
  tFit->SetLineColor(1);
  tFit->Draw("same");

  TLine *line = new TLine(aXmin,1,aXmax,1);
  line->SetLineColor(14);
  line->Draw();
}


//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfs()
{
  TString tCanvasName = TString("canKStarCf") + TString(cAnalysisBaseTags[fPairType]) + TString("&") 
                        + TString(cAnalysisBaseTags[fConjPairType]) + TString(cCentralityTags[fCentralityType]);

  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  if(fNAnalyses == 6) tReturnCan->Divide(2,3);
  else if(fNAnalyses == 2 || fNAnalyses==1) tReturnCan->Divide(fNAnalyses,1);
  else if(fNAnalyses == 3) tReturnCan->Divide(1,fNAnalyses);
  else assert(0);

  for(int i=0; i<fNAnalyses; i++)
  {
    DrawSingleKStarCf((TPad*)tReturnCan->cd(i+1),i,0.9,1.04);
  }

  return tReturnCan;
}
/*
//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfswFits()
{
  TString tCanvasName = TString("canKStarCfwFits") + TString(cAnalysisBaseTags[fPairType]) + TString("&") 
                        + TString(cAnalysisBaseTags[fConjPairType]) + TString(cCentralityTags[fCentralityType]);

  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  if(fNAnalyses == 6) tReturnCan->Divide(2,3);
  else if(fNAnalyses == 2 || fNAnalyses==1) tReturnCan->Divide(fNAnalyses,1);
  else if(fNAnalyses == 3) tReturnCan->Divide(1,fNAnalyses);
  else assert(0);

  for(int i=0; i<fNAnalyses; i++)
  {
    DrawSingleKStarCfwFit((TPad*)tReturnCan->cd(i+1),i,0.9,1.04);
  }

  return tReturnCan;
}
*/

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfswFits()
{
  TString tCanvasName = TString("canKStarCfwFits") + TString(cAnalysisBaseTags[fPairType]) + TString("And") 
                        + TString(cAnalysisBaseTags[fConjPairType]) + TString(cCentralityTags[fCentralityType]);

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else assert(0);

  double tXLow = -0.01;
  double tXHigh = 0.49;
  double tYLow = 0.9;
  double tYHigh = 1.04;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCf(),"");
      tCanPart->AddGraph(i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetFit(),"");
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)");

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
//TODO
void FitGenerator::SetDefaultSharedParameters()
{
  vector<int> Share01(2);
    Share01[0] = 0;
    Share01[1] = 1;

  vector<int> Share23(2);
    Share23[0] = 2;
    Share23[1] = 3;

  vector<int> Share45(2);
    Share45[0] = 4;
    Share45[1] = 5;


  //TODO make default for each type of analysis
  double tLambda = 0.19;
  double tRadius = 5.0;

  double tRef0 = -1.7;
  double tImf0 = 1.1;
  double td0 = 3.;

  //Always shared amongst all
  SetSharedParameter(kRef0,tRef0);
  SetSharedParameter(kImf0,tImf0);
  SetSharedParameter(kd0,td0);

  if(fNAnalyses==1)
  {
    SetSharedParameter(kLambda,tLambda,0.1,1.0);
    SetSharedParameter(kRadius,tRadius);
  }
  else if(fNAnalyses==2)
  {
    SetSharedParameter(kLambda,Share01,0.5,0.1,1.0);
    SetSharedParameter(kRadius,Share01,5.0);
  }
  else if(fNAnalyses==3)
  {

  }
  else if(fNAnalyses==6)
  {
    SetSharedParameter(kLambda,Share01,0.5,0.1,1.0);
    SetSharedParameter(kLambda,Share23,0.5,0.1,1.0);
    SetSharedParameter(kLambda,Share45,0.5,0.1,1.0);

    SetSharedParameter(kRadius,Share01,5.0);
    SetSharedParameter(kRadius,Share23,4.0);
    SetSharedParameter(kRadius,Share45,3.0);
  }
  else
  {
    cout << "ERROR:  FitGenerator::SetDefaultSharedParameters:: Incorrect fNAnalyses = " << fNAnalyses << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
void FitGenerator::DoFit()
{
  fSharedAn->CreateMinuitParameters();

  fLednickyFitter = new LednickyFitter(fSharedAn);
  fLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN);
  GlobalFitter = fLednickyFitter;

  fLednickyFitter->DoFit();
}




