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
//  GlobalFitter->CalculateChi2PML(npar,f,par);
  GlobalFitter->CalculateFitFunction(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, AnalysisRunType aRunType, int aNPartialAnalysis, CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams, TString aDirNameModifier) :
  fSaveLocationBase(""),
  fSaveNameModifier(""),
  fContainsMC(false),
  fNAnalyses(0),
  fGeneratorType(aGeneratorType),
  fPairType(kLamK0), fConjPairType(kALamK0),
  fCentralityType(aCentralityType),

  fRadiusFitParams(),
  fScattFitParams(),
  fLambdaFitParams(),
  fShareLambdaParams(aShareLambdaParams),
  fFitParamsPerPad(),

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

  fRadiusFitParams.reserve(10);  //Not really needed, but without these reserve calls, every time a new item is emplaced back
  fScattFitParams.reserve(10);   //  instead of simply adding the object to the vector, the vector is first copied into a new vector
  fLambdaFitParams.reserve(10);  //  with a larger size, and then the new object is included
                                 //  This is why I was initially seeing my FitParameter destructor being called

  fScattFitParams.emplace_back(kRef0, 0.0);
  fScattFitParams.emplace_back(kImf0, 0.0);
  fScattFitParams.emplace_back(kd0, 0.0);

  vector<FitPairAnalysis*> tVecOfPairAn;
  fRadiusFitParams.emplace_back(kRadius, 0.0);
  switch(fCentralityType) {
  case k0010:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case k1030:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case k3050:
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
    break;

  case kMB:
    fRadiusFitParams.emplace_back(kRadius,0.0);
    fRadiusFitParams.emplace_back(kRadius,0.0);
    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k0010,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k1030,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k3050,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k0010,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k1030,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k3050,aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k0010,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k0010,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k1030,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k1030,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,k3050,aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,k3050,aRunType,aNPartialAnalysis,aDirNameModifier));
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
  SetDefaultSharedParameters();

}


//________________________________________________________________________________________________________________
FitGenerator::~FitGenerator()
{

}

//________________________________________________________________________________________________________________
void FitGenerator::SetNAnalyses()
{
  fLambdaFitParams.emplace_back(kLambda,0.0);
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
      if(!fShareLambdaParams) fLambdaFitParams.emplace_back(kLambda,0.0);
      break;

    default:
      cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fGeneratorType = " << fGeneratorType << endl;
    }
    break;


  case kMB:
    fLambdaFitParams.emplace_back(kLambda,0.0);
    fLambdaFitParams.emplace_back(kLambda,0.0);
    switch(fGeneratorType) {
    case kPair:
    case kConjPair:
      fNAnalyses = 3;
      break;

    case kPairwConj:
      fNAnalyses = 6;
      if(!fShareLambdaParams)
      {
        fLambdaFitParams.emplace_back(kLambda,0.0);
        fLambdaFitParams.emplace_back(kLambda,0.0);
        fLambdaFitParams.emplace_back(kLambda,0.0);
      }
      break;

    default:
      cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fGeneratorType = " << fGeneratorType << endl;
    }
    break;


  default:
    cout << "ERROR:  FitGenerator::SetNAnalyses():  Invalid fCentralityType = " << fCentralityType << endl;
    assert(0);
  }
  fFitParamsPerPad.clear();
  fFitParamsPerPad.resize(fNAnalyses);
  for(int i=0; i<fNAnalyses; i++)
  {
    fFitParamsPerPad[i].emplace_back(kLambda,0.);
    fFitParamsPerPad[i].emplace_back(kRadius,0.);
    fFitParamsPerPad[i].emplace_back(kRef0,0.);
    fFitParamsPerPad[i].emplace_back(kImf0,0.);
    fFitParamsPerPad[i].emplace_back(kd0,0.);
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
void FitGenerator::CreateParamInitValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else assert(0);

  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;

  tLambda = fFitParamsPerPad[tPosition][0].GetStartValue();
  tRadius = fFitParamsPerPad[tPosition][1].GetStartValue();
  tReF0 = fScattFitParams[0].GetStartValue();
  tImF0 = fScattFitParams[1].GetStartValue();
  tD0 = fScattFitParams[2].GetStartValue();

  assert(tReF0 == fFitParamsPerPad[tPosition][2].GetStartValue());
  assert(tImF0 == fFitParamsPerPad[tPosition][3].GetStartValue());
  assert(tD0 == fFitParamsPerPad[tPosition][4].GetStartValue());

  TPaveText *tText = aCanPart->SetupTPaveText("Initial Values",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.3f",tLambda));
  tText->AddText(TString::Format("R = %0.3f",tRadius));
  tText->AddText(TString::Format("Re[f0] = %0.3f",tReF0));
  tText->AddText(TString::Format("Im[f0] = %0.3f",tImF0));
  tText->AddText(TString::Format("d0 = %0.3f",tD0));

  tText->SetTextAlign(33);

//  tText->GetLine(0)->SetTextSize(0.08);
  tText->GetLine(0)->SetTextFont(63);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGenerator::AddTextCorrectionInfo(CanvasPartition *aCanPart, int aNx, int aNy, bool aMomResCorrect, bool aNonFlatCorrect, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  if(!aMomResCorrect && !aNonFlatCorrect) return;

  TPaveText *tText = aCanPart->SetupTPaveText("",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  if(aMomResCorrect) tText->AddText("Mom. Res. Correction");
  if(aNonFlatCorrect) tText->AddText("Non-flat Bgd Correction");
  tText->SetTextAlign(33);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGenerator::DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCfClone();
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


  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCfClone();
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

/*
//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfs()
{
  TString tCanvasName = TString("canKStarCf") + TString(cAnalysisBaseTags[fPairType]) + TString("wConj") 
                        + TString(cCentralityTags[fCentralityType]);

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
*/

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfs(bool aSaveImage, bool aDrawSysErrors)
{
  TString tCanvasName = TString("canKStarCfs") + TString(cAnalysisBaseTags[fPairType]) + TString("wConj") 
                      + TString(cCentralityTags[fCentralityType]);

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  int tMarkerColor = 1;
  double tMarkerSize = 0.5;

  if(fPairType==kLamK0 || fPairType==kALamK0) tMarkerColor = 1;
  else if(fPairType==kLamKchP || fPairType==kALamKchM) tMarkerColor = 2;
  else if(fPairType==kLamKchM || fPairType==kALamKchP) tMarkerColor = 4;
  else tMarkerColor=1;

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
      TH1* tHistToPlot;
      if(aDrawSysErrors)
      {
        TH1* tHistToPlot = (TH1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCfwSysErrors();
          tHistToPlot->SetFillStyle(0);  //for box error bars to draw correctly
      }

//TODO 
//If the binnings are unequal, I must regenerate the plots with Analyze/Systematics/BuildErrorBars.C
//This is because a Cf should not simply be rebinned, but rather the Num and Den should be rebinned, and the Cf rebuilt
//Ths incorrect method would be Cf->Rebin(aRebin); Cf->Scale(1./aRebin);
/*
      double tDesiredBinWidth, tBinWidth;
      tDesiredBinWidth = ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1);
      tBinWidth = tHistToPlot->GetBinWidth(1);

      if( (tDesiredBinWidth != tBinWidth) && (fmod(tDesiredBinWidth,tBinWidth) == 0) )
      {
        int tScale = tDesiredBinWidth/tBinWidth;
        tHistToPlot->Rebin(tScale);
        tHistToPlot->Scale(1./tScale);
      }
      else if(tDesiredBinWidth != tBinWidth)
      {
        cout << "ERROR: FitGenerator::DrawKStarCfs: Histogram containing systematic error bars does not have the correct bin size and" << endl;
        cout << "DNE an appropriate scale to resolve the issue" << endl;
        assert(0);
      }
*/
//      assert(tHistToPlot->GetBinWidth(1) == tDesiredBinWidth);
      if(aDrawSysErrors) assert(tHistToPlot->GetBinWidth(1) == ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1));
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",tMarkerStyle,tMarkerColor,tMarkerSize);
      if(aDrawSysErrors) tCanPart->AddGraph(i,j,tHistToPlot,"",tMarkerStyle,tMarkerColor,tMarkerSize,"e2psame");

      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString(".pdf"));
  }

  return tCanPart->GetCanvas();
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
TCanvas* FitGenerator::DrawKStarCfswFits(bool aMomResCorrectFit, bool aNoFlatBgdCorrectFit, bool aSaveImage, bool aDrawSysErrors)
{
  TString tCanvasName = TString("canKStarCfwFits") + TString(cAnalysisBaseTags[fPairType]) + TString("wConj") 
                      + TString(cCentralityTags[fCentralityType]);

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      int tColor;
      AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      if(tAnType==kLamK0 || tAnType==kALamK0) tColor=1;
      else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=2;
      else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=4;
      else tColor=1;

      TH1* tCorrectedFitHisto = (TH1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCorrectedFitHisto(aMomResCorrectFit,aNoFlatBgdCorrectFit);
        tCorrectedFitHisto->SetLineWidth(2);

      //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
      TH1* tHistToPlot;
      if(aDrawSysErrors)
      {
        tHistToPlot = (TH1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCfwSysErrors();
          tHistToPlot->SetFillStyle(0);  //for box error bars to draw correctly
      }

//TODO 
//If the binnings are unequal, I must regenerate the plots with Analyze/Systematics/BuildErrorBars.C
//This is because a Cf should not simply be rebinned, but rather the Num and Den should be rebinned, and the Cf rebuilt
//Ths incorrect method would be Cf->Rebin(aRebin); Cf->Scale(1./aRebin);
/*
      double tDesiredBinWidth, tBinWidth;
      tDesiredBinWidth = ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1);
      tBinWidth = tHistToPlot->GetBinWidth(1);

      if( (tDesiredBinWidth != tBinWidth) && (fmod(tDesiredBinWidth,tBinWidth) == 0) )
      {
        int tScale = tDesiredBinWidth/tBinWidth;
        tHistToPlot->Rebin(tScale);
        tHistToPlot->Scale(1./tScale);
      }
      else if(tDesiredBinWidth != tBinWidth)
      {
        cout << "ERROR: FitGenerator::DrawKStarCfswFits: Histogram containing systematic error bars does not have the correct bin size and" << endl;
        cout << "DNE an appropriate scale to resolve the issue" << endl;
        assert(0);
      }
*/
//      assert(tHistToPlot->GetBinWidth(1) == tDesiredBinWidth);
      if(aDrawSysErrors) assert(tHistToPlot->GetBinWidth(1) == ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1));
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5);
      if(aDrawSysErrors) tCanPart->AddGraph(i,j,tHistToPlot,"",20,tColor,0.5,"e2psame");
      tCanPart->AddGraph(i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetFit(),"");
      tCanPart->AddGraph(i,j,tCorrectedFitHisto,"",20,6,0.5,"lsame");
      tCanPart->AddGraph(i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetNonFlatBackground(),"",20,3);



      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      CreateParamInitValuesText(tCanPart,i,j,0.25,0.20,0.15,0.45,43,10);
      AddTextCorrectionInfo(tCanPart,i,j,aMomResCorrectFit,aNoFlatBgdCorrectFit,0.25,0.08,0.15,0.10,43,7.5);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString(".pdf"));
  }

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
void FitGenerator::SetUseLimits(vector<FitParameter> &aVec, bool aUse)
{
  for(unsigned int i=0; i<aVec.size(); i++)
  {
    if(aUse == false)
    {
      aVec[i].SetLowerBound(0.);
      aVec[i].SetUpperBound(0.);
    }
    else
    {
      if( (aVec[i].GetLowerBound() == 0.) && (aVec[i].GetUpperBound() == 0.) )
      {
        cout << "Warning:  FitGenerator::SetUseLimits set to true but no limits have been set" << endl;
        cout << "Is this alright? (0=No, 1=Yes)" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusStartValue(double aRad, CentralityType aCentType)
{
  if(fCentralityType == kMB)
  {
    assert(aCentType != kMB);
    fRadiusFitParams[aCentType].SetStartValue(aRad);
  }
  else fRadiusFitParams[0].SetStartValue(aRad);
}


//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusStartValues(double aRad0010, double aRad1030, double aRad3050)
{
  assert(fCentralityType == kMB);
  fRadiusFitParams[k0010].SetStartValue(aRad0010);
  fRadiusFitParams[k1030].SetStartValue(aRad1030);
  fRadiusFitParams[k3050].SetStartValue(aRad3050);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusLimits(double aMin, double aMax, CentralityType aCentType)
{
  if(fCentralityType == kMB)
  {
    assert(aCentType != kMB);
    fRadiusFitParams[aCentType].SetLowerBound(aMin);
    fRadiusFitParams[aCentType].SetUpperBound(aMax);
  }
  else
  {
    fRadiusFitParams[0].SetLowerBound(aMin);
    fRadiusFitParams[0].SetUpperBound(aMax);
  }
}


//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusLimits(double aMin0010, double aMax0010, double aMin1030, double aMax1030, double aMin3050, double aMax3050)
{
  assert(fCentralityType == kMB);
  fRadiusFitParams[k0010].SetLowerBound(aMin0010);
  fRadiusFitParams[k0010].SetUpperBound(aMax0010);
  fRadiusFitParams[k1030].SetLowerBound(aMin1030);
  fRadiusFitParams[k1030].SetUpperBound(aMax1030);
  fRadiusFitParams[k3050].SetLowerBound(aMin3050);
  fRadiusFitParams[k3050].SetUpperBound(aMax3050);
}


//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamStartValues(double aReF0, double aImF0, double aD0)
{
  fScattFitParams[0].SetStartValue(aReF0);
  fScattFitParams[1].SetStartValue(aImF0);
  fScattFitParams[2].SetStartValue(aD0);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamLimits(double aMinReF0, double aMaxReF0, double aMinImF0, double aMaxImF0, double aMinD0, double aMaxD0)
{
  fScattFitParams[0].SetLowerBound(aMinReF0);
  fScattFitParams[0].SetUpperBound(aMaxReF0);

  fScattFitParams[1].SetLowerBound(aMinImF0);
  fScattFitParams[1].SetUpperBound(aMaxImF0);

  fScattFitParams[2].SetLowerBound(aMinD0);
  fScattFitParams[2].SetUpperBound(aMaxD0);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetLambdaParamStartValue(double aLam, bool tConjPair, CentralityType aCentType)
{
  //TODO write function to find correct vector bin
  int tBinNumber = -1;
  if(fNAnalyses==1) tBinNumber=0;

  else if(fNAnalyses==2)
  {
    if(fShareLambdaParams) tBinNumber=0;
    else
    {
      if(!tConjPair) tBinNumber=0;
      else tBinNumber=1;
    }
  }

  else if(fNAnalyses==3)
  {
    assert(aCentType != kMB);
    tBinNumber = aCentType;
  }

  else if(fNAnalyses==6)
  {
    assert(aCentType != kMB);
    if(fShareLambdaParams) tBinNumber = aCentType;
    else
    {
      int tRow = aCentType;
      int tPosition = -1;
      if(!tConjPair) tPosition = 2*tRow;
      else tPosition = 2*tRow+1;

      tBinNumber = tPosition;
    }
  }
  else
  {
    cout << "ERROR:  FitGenerator::SetLambdaParamStartValue:  Incorrect fNAnalyses = " << fNAnalyses << endl;
    assert(0);
  }

  fLambdaFitParams[tBinNumber].SetStartValue(aLam);
}


//________________________________________________________________________________________________________________
void FitGenerator::SetLambdaParamLimits(double aMin, double aMax, bool tConjPair, CentralityType aCentType)
{
  //TODO write function to find correct vector bin
  int tBinNumber = -1;
  if(fNAnalyses==1) tBinNumber=0;

  else if(fNAnalyses==2)
  {
    if(fShareLambdaParams) tBinNumber=0;
    else
    {
      if(!tConjPair) tBinNumber=0;
      else tBinNumber=1;
    }
  }

  else if(fNAnalyses==3)
  {
    assert(aCentType != kMB);
    tBinNumber = aCentType;
  }

  else if(fNAnalyses==6)
  {
    assert(aCentType != kMB);
    if(fShareLambdaParams) tBinNumber = aCentType;
    else
    {
      int tRow = aCentType;
      int tPosition = -1;
      if(!tConjPair) tPosition = 2*tRow;
      else tPosition = 2*tRow+1;

      tBinNumber = tPosition;
    }
  }
  else
  {
    cout << "ERROR:  FitGenerator::SetLambdaParamLimits:  Incorrect fNAnalyses = " << fNAnalyses << endl;
    assert(0);
  }

  fLambdaFitParams[tBinNumber].SetLowerBound(aMin);
  fLambdaFitParams[tBinNumber].SetUpperBound(aMax);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetDefaultSharedParameters()
{
//TODO this seems like overkill because this is already handled at the FitPartialAnalysis level
  const double** tStartValuesPair;
  const double** tStartValuesConjPair;

  switch(fPairType) {
  case kLamK0:
    tStartValuesPair = cLamK0StartValues;
    tStartValuesConjPair = cALamK0StartValues;
    break;

  case kLamKchP:
    tStartValuesPair = cLamKchPStartValues;
    tStartValuesConjPair = cALamKchMStartValues;
    break;

  case kLamKchM:
    tStartValuesPair = cLamKchMStartValues;
    tStartValuesConjPair = cALamKchPStartValues;
    break;

  default:
    cout << "ERROR: FitGenerator::SetDefaultSharedParameters:  Invalid fPairType = " << fPairType << endl;
    assert(0);
  }


  switch(fCentralityType) {
  case k0010:
  case k1030:
  case k3050:
    if(fGeneratorType==kPair)
    {
      SetRadiusStartValue(tStartValuesPair[fCentralityType][1],fCentralityType);
      SetRadiusLimits(2.,8.,fCentralityType);

      SetScattParamStartValues(tStartValuesPair[fCentralityType][2],tStartValuesPair[fCentralityType][3],tStartValuesPair[fCentralityType][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesPair[fCentralityType][0]);
      SetLambdaParamLimits(0.1,0.8);
    }

    else if(fGeneratorType==kConjPair)
    {
      SetRadiusStartValue(tStartValuesConjPair[fCentralityType][1],fCentralityType);
      SetRadiusLimits(2.,8.,fCentralityType);

      SetScattParamStartValues(tStartValuesConjPair[fCentralityType][2],tStartValuesConjPair[fCentralityType][3],tStartValuesConjPair[fCentralityType][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesConjPair[fCentralityType][0]);
      SetLambdaParamLimits(0.1,0.8);
    }

    else if(fGeneratorType==kPairwConj)
    {
      SetRadiusStartValue(tStartValuesPair[fCentralityType][1],fCentralityType);
      SetRadiusLimits(2.,8.,fCentralityType);

      SetScattParamStartValues(tStartValuesPair[fCentralityType][2],tStartValuesPair[fCentralityType][3],tStartValuesPair[fCentralityType][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesPair[fCentralityType][0]);
      SetLambdaParamLimits(0.1,0.8);
      if(!fShareLambdaParams) SetLambdaParamStartValue(tStartValuesConjPair[fCentralityType][0],true);
    }

    else assert(0);
    break;

  case kMB:
    if(fGeneratorType==kPair)
    {
      SetRadiusStartValues(tStartValuesPair[k0010][1],tStartValuesPair[k1030][1],tStartValuesPair[k3050][1]);
      SetRadiusLimits(2.,8.,2.,8.,2.,8.);

      SetScattParamStartValues(tStartValuesPair[k0010][2],tStartValuesPair[k0010][3],tStartValuesPair[k0010][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesPair[k0010][0],false,k0010);
      SetLambdaParamStartValue(tStartValuesPair[k1030][0],false,k1030);
      SetLambdaParamStartValue(tStartValuesPair[k3050][0],false,k3050);

      SetLambdaParamLimits(0.1,0.8,false,k0010);
      SetLambdaParamLimits(0.1,0.8,false,k1030);
      SetLambdaParamLimits(0.1,0.8,false,k3050);
    }

    else if(fGeneratorType==kConjPair)
    {
      SetRadiusStartValues(tStartValuesConjPair[k0010][1],tStartValuesConjPair[k1030][1],tStartValuesConjPair[k3050][1]);
      SetRadiusLimits(2.,8.,2.,8.,2.,8.);

      SetScattParamStartValues(tStartValuesConjPair[k0010][2],tStartValuesConjPair[k0010][3],tStartValuesConjPair[k0010][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesConjPair[k0010][0],false,k0010);
      SetLambdaParamStartValue(tStartValuesConjPair[k1030][0],false,k1030);
      SetLambdaParamStartValue(tStartValuesConjPair[k3050][0],false,k3050);

      SetLambdaParamLimits(0.1,0.8,false,k0010);
      SetLambdaParamLimits(0.1,0.8,false,k1030);
      SetLambdaParamLimits(0.1,0.8,false,k3050);
    }

    else if(fGeneratorType==kPairwConj)
    {
      SetRadiusStartValues(tStartValuesPair[k0010][1],tStartValuesPair[k1030][1],tStartValuesPair[k3050][1]);
      SetRadiusLimits(2.,8.,2.,8.,2.,8.);

      SetScattParamStartValues(tStartValuesPair[k0010][2],tStartValuesPair[k0010][3],tStartValuesPair[k0010][4]);
      SetScattParamLimits(0.,0.,0.,0.,0.,0.);

      SetLambdaParamStartValue(tStartValuesPair[k0010][0],false,k0010);
      SetLambdaParamStartValue(tStartValuesPair[k1030][0],false,k1030);
      SetLambdaParamStartValue(tStartValuesPair[k3050][0],false,k3050);

      SetLambdaParamLimits(0.1,0.8,false,k0010);
      SetLambdaParamLimits(0.1,0.8,false,k1030);
      SetLambdaParamLimits(0.1,0.8,false,k3050);

      if(!fShareLambdaParams)
      {
        SetLambdaParamStartValue(tStartValuesConjPair[k0010][0],true,k0010);
        SetLambdaParamStartValue(tStartValuesConjPair[k1030][0],true,k1030);
        SetLambdaParamStartValue(tStartValuesConjPair[k3050][0],true,k3050);

        SetLambdaParamLimits(0.1,0.8,true,k0010);
        SetLambdaParamLimits(0.1,0.8,true,k1030);
        SetLambdaParamLimits(0.1,0.8,true,k3050);
      }
    }

    else assert(0);
    break;

  default:
    cout << "ERROR: FitGenerator::SetDefaultSharedParameters:  Invalid fCentralityType = " << fCentralityType << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
void FitGenerator::SetAllParameters()
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


  //Always shared amongst all
  SetSharedParameter(kRef0,fScattFitParams[0].GetStartValue(),fScattFitParams[0].GetLowerBound(),fScattFitParams[0].GetUpperBound());
  SetSharedParameter(kImf0,fScattFitParams[1].GetStartValue(),fScattFitParams[1].GetLowerBound(),fScattFitParams[1].GetUpperBound());
  SetSharedParameter(kd0,fScattFitParams[2].GetStartValue(),fScattFitParams[2].GetLowerBound(),fScattFitParams[2].GetUpperBound());

  if(fNAnalyses==1)
  {
    SetSharedParameter(kLambda, fLambdaFitParams[0].GetStartValue(), 
                       fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());

    SetSharedParameter(kRadius, fRadiusFitParams[0].GetStartValue(),
                       fRadiusFitParams[0].GetLowerBound(), fRadiusFitParams[0].GetUpperBound());

    fFitParamsPerPad[0][0] = fLambdaFitParams[0];
    fFitParamsPerPad[0][1] = fRadiusFitParams[0];
  }
  else if(fNAnalyses==2)
  {
    if(fShareLambdaParams)
    {
      SetSharedParameter(kLambda, Share01, fLambdaFitParams[0].GetStartValue(), 
                         fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());

      for(int i=0; i<fNAnalyses; i++) fFitParamsPerPad[i][0] = fLambdaFitParams[0];
    }
    else
    {
      SetParameter(kLambda, 0, fLambdaFitParams[0].GetStartValue(),
                   fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());
      SetParameter(kLambda, 1, fLambdaFitParams[1].GetStartValue(),
                   fLambdaFitParams[1].GetLowerBound(), fLambdaFitParams[1].GetUpperBound());

      fFitParamsPerPad[0][0] = fLambdaFitParams[0];
      fFitParamsPerPad[1][0] = fLambdaFitParams[1];
    }

    SetSharedParameter(kRadius, Share01, fRadiusFitParams[0].GetStartValue(),
                       fRadiusFitParams[0].GetLowerBound(), fRadiusFitParams[0].GetUpperBound());

    for(int i=0; i<fNAnalyses; i++) fFitParamsPerPad[i][1] = fRadiusFitParams[0];
  }
  else if(fNAnalyses==3)
  {
    SetParameter(kLambda, 0, fLambdaFitParams[0].GetStartValue(),
                 fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());
    SetParameter(kLambda, 1, fLambdaFitParams[1].GetStartValue(),
                 fLambdaFitParams[1].GetLowerBound(), fLambdaFitParams[1].GetUpperBound());
    SetParameter(kLambda, 2, fLambdaFitParams[2].GetStartValue(),
                 fLambdaFitParams[2].GetLowerBound(), fLambdaFitParams[2].GetUpperBound());

    SetParameter(kRadius, 0, fRadiusFitParams[0].GetStartValue(),
                 fRadiusFitParams[0].GetLowerBound(), fRadiusFitParams[0].GetUpperBound());
    SetParameter(kRadius, 1, fRadiusFitParams[1].GetStartValue(),
                 fRadiusFitParams[1].GetLowerBound(), fRadiusFitParams[1].GetUpperBound());
    SetParameter(kRadius, 2, fRadiusFitParams[2].GetStartValue(),
                 fRadiusFitParams[2].GetLowerBound(), fRadiusFitParams[2].GetUpperBound());

    for(int i=0; i<fNAnalyses; i++)
    {
      fFitParamsPerPad[i][0] = fLambdaFitParams[i];
      fFitParamsPerPad[i][1] = fRadiusFitParams[i];
    }
  }
  else if(fNAnalyses==6)
  {
    SetSharedParameter(kRadius, Share01, fRadiusFitParams[k0010].GetStartValue(),
                       fRadiusFitParams[k0010].GetLowerBound(), fRadiusFitParams[k0010].GetUpperBound());
    SetSharedParameter(kRadius, Share23, fRadiusFitParams[k1030].GetStartValue(),
                       fRadiusFitParams[k1030].GetLowerBound(), fRadiusFitParams[k1030].GetUpperBound());
    SetSharedParameter(kRadius, Share45, fRadiusFitParams[k3050].GetStartValue(),
                       fRadiusFitParams[k3050].GetLowerBound(), fRadiusFitParams[k3050].GetUpperBound());

    for(int i=0; i<(fNAnalyses/2); i++)
    {
      fFitParamsPerPad[2*i][1] = fRadiusFitParams[i];
      fFitParamsPerPad[2*i+1][1] = fRadiusFitParams[i];
    }

    if(fShareLambdaParams)
    {
      SetSharedParameter(kLambda, Share01, fLambdaFitParams[0].GetStartValue(),
                         fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());
      SetSharedParameter(kLambda, Share23, fLambdaFitParams[1].GetStartValue(),
                         fLambdaFitParams[1].GetLowerBound(), fLambdaFitParams[1].GetUpperBound());
      SetSharedParameter(kLambda, Share45, fLambdaFitParams[2].GetStartValue(),
                         fLambdaFitParams[2].GetLowerBound(), fLambdaFitParams[2].GetUpperBound());

      for(int i=0; i<(fNAnalyses/2); i++)
      {
        fFitParamsPerPad[2*i][0] = fLambdaFitParams[i];
        fFitParamsPerPad[2*i+1][0] = fLambdaFitParams[i];
      }
    }
    else
    {
      SetParameter(kLambda, 0, fLambdaFitParams[0].GetStartValue(),
                   fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound());
      SetParameter(kLambda, 1, fLambdaFitParams[1].GetStartValue(),
                   fLambdaFitParams[1].GetLowerBound(), fLambdaFitParams[1].GetUpperBound());
      SetParameter(kLambda, 2, fLambdaFitParams[2].GetStartValue(),
                   fLambdaFitParams[2].GetLowerBound(), fLambdaFitParams[2].GetUpperBound());
      SetParameter(kLambda, 3, fLambdaFitParams[3].GetStartValue(),
                   fLambdaFitParams[3].GetLowerBound(), fLambdaFitParams[3].GetUpperBound());
      SetParameter(kLambda, 4, fLambdaFitParams[4].GetStartValue(),
                   fLambdaFitParams[4].GetLowerBound(), fLambdaFitParams[4].GetUpperBound());
      SetParameter(kLambda, 5, fLambdaFitParams[5].GetStartValue(),
                   fLambdaFitParams[5].GetLowerBound(), fLambdaFitParams[5].GetUpperBound());

      for(int i=0; i<fNAnalyses; i++) fFitParamsPerPad[i][0] = fLambdaFitParams[i];
    }
  }
  else
  {
    cout << "ERROR:  FitGenerator::SetAllParameters:: Incorrect fNAnalyses = " << fNAnalyses << endl;
    assert(0);
  }

  for(int i=0; i<fNAnalyses; i++)
  {
    fFitParamsPerPad[i][2] = fScattFitParams[0];
    fFitParamsPerPad[i][3] = fScattFitParams[1];
    fFitParamsPerPad[i][4] = fScattFitParams[2];
  }

}


//________________________________________________________________________________________________________________
void FitGenerator::DoFit(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, double aMaxFitKStar)
{
  SetAllParameters();

  fSharedAn->CreateMinuitParameters();

  fLednickyFitter = new LednickyFitter(fSharedAn,aMaxFitKStar);
  fLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN);
  fLednickyFitter->SetApplyMomResCorrection(aApplyMomResCorrection);
  fLednickyFitter->SetApplyNonFlatBackgroundCorrection(aApplyNonFlatBackgroundCorrection);
  GlobalFitter = fLednickyFitter;

  fLednickyFitter->DoFit();
}

//________________________________________________________________________________________________________________
void FitGenerator::WriteAllFitParameters(ostream &aOut)
{
  for(int iAn=0; iAn<fSharedAn->GetNFitPairAnalysis(); iAn++)
  {
    aOut << "______________________________" << endl;
    aOut << "AnalysisType = " << cAnalysisBaseTags[fSharedAn->GetFitPairAnalysis(iAn)->GetAnalysisType()] << endl;
    aOut << "CentralityType = " << cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(iAn)->GetCentralityType()] << endl << endl;
    fSharedAn->GetFitPairAnalysis(iAn)->WriteFitParameters(aOut);
    aOut << endl;
  }
}

//________________________________________________________________________________________________________________
vector<TString> FitGenerator::GetAllFitParametersTStringVector()
{
  vector<TString> tReturnVec(0);

  for(int iAn=0; iAn<fSharedAn->GetNFitPairAnalysis(); iAn++)
  {
    tReturnVec.push_back("______________________________");
    TString tAnLine = TString("AnalysisType = ") + TString(cAnalysisBaseTags[fSharedAn->GetFitPairAnalysis(iAn)->GetAnalysisType()]);
      tReturnVec.push_back(tAnLine);
    TString tCentLine = TString("CentralityType = ") + TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(iAn)->GetCentralityType()]);
      tReturnVec.push_back(tCentLine);
    tReturnVec.push_back(TString(""));

    vector<TString> tParamVec = fSharedAn->GetFitPairAnalysis(iAn)->GetFitParametersTStringVector();
    for(unsigned int iPar=0; iPar<tParamVec.size(); iPar++) tReturnVec.push_back(tParamVec[iPar]);
    tReturnVec.push_back(TString(""));
  }
  return tReturnVec;
}

//________________________________________________________________________________________________________________
void FitGenerator::FindGoodInitialValues(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection)
{
  SetAllParameters();

  fSharedAn->CreateMinuitParameters();

  fLednickyFitter = new LednickyFitter(fSharedAn);
  fLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN);
  fLednickyFitter->SetApplyMomResCorrection(aApplyMomResCorrection);
  fLednickyFitter->SetApplyNonFlatBackgroundCorrection(aApplyNonFlatBackgroundCorrection);
  GlobalFitter = fLednickyFitter;

  vector<double> tValues = fLednickyFitter->FindGoodInitialValues();

  for(unsigned int i=0; i<tValues.size(); i++) cout << "i = " << i << ": " << tValues[i] << endl;
}


//________________________________________________________________________________________________________________
void FitGenerator::SetSaveLocationBase(TString aBase, TString aSaveNameModifier)
{
  fSaveLocationBase=aBase;
  if(!aSaveNameModifier.IsNull()) fSaveNameModifier = aSaveNameModifier;
}

//________________________________________________________________________________________________________________
void FitGenerator::ExistsSaveLocationBase()
{
  if(!fSaveLocationBase.IsNull()) return;

  cout << "fSaveLocationBase is Null!!!!!" << endl;
  cout << "Create? (0=No 1=Yes)" << endl;
  int tResponse;
  cin >> tResponse;
  if(!tResponse) return;

  cout << "Enter base:" << endl;
  cin >> fSaveLocationBase;
  if(fSaveLocationBase[fSaveLocationBase.Length()] != '/') fSaveLocationBase += TString("/");
  return;

}


