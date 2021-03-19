#include "CompareFittingMethodswSysErrs.h"
#include "CanvasPartition.h"

//_________________________________________________________________________________________________________________________________
TCanvas* CompareImF0vsReF0(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly, bool aDrawCircleStamps)
{
  CentralityType tCentType = k0010;  //Doesn't matter which centrality chosen, because all share same scattering parameters

  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareImF0vsReF0wSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();
  tReturnCan->SetTopMargin(0.02);
  tReturnCan->SetBottomMargin(0.175);
  tReturnCan->SetLeftMargin(0.175);

  TPad* tPadReF0vsImF0 = new TPad(TString::Format("tPadReF0vsImF0%s", tCanName.Data()), TString::Format("tPadReF0vsImF0%s", tCanName.Data()), 
                                  0.0, 0.0, 0.8, 1.0);
  tPadReF0vsImF0->SetRightMargin(0.02);
  tPadReF0vsImF0->SetTopMargin(0.02);
  tPadReF0vsImF0->SetBottomMargin(0.175);
  tPadReF0vsImF0->SetLeftMargin(0.175);
  tPadReF0vsImF0->SetTicks(1,1);  
  tPadReF0vsImF0->Draw();

  TPad* tPadD0 = new TPad(TString::Format("tPadD0%s", tCanName.Data()), TString::Format("tPadD0%s", tCanName.Data()), 
                          0.8, 0.0, 1.0, 1.0);
  tPadD0->SetRightMargin(0.6);
  tPadD0->SetLeftMargin(0.);
  tPadD0->SetTopMargin(0.02);
  tPadD0->SetBottomMargin(0.175);
  tPadD0->SetTicks(1,1);  
  tPadD0->Draw();

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupReF0vsImF0AndD0Axes(tPadReF0vsImF0, tPadD0);

  //------------------------------------------------------
  double tStartX = -1.;
  if(aDrawPredictions) tStartX = -1.35;
  double tStartY = 1.4;
  double tIncrementX = 0.075;
  double tIncrementY = 0.10;
  double tTextSize = 0.0575;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  //------------------------------------------------------
  double tD0XOffset = 0.5;

  //I don't want to repeat entries in the legend
  //  For instance, without the "used" check, "3 Res., Poly Bgd" would be printed for LamKchP and LamKchM (and LamK0 if it is included)
  vector<TString> tUsedDescriptors(0);
  vector<int> tUsedMarkerStyles(0);
  vector<AnalysisType> tAnTypes(0);

  int iTex = 0;
  int iD0Inc = 0;
  double tIncrementSize = 1./(aFitValWriterInfo.size()+1);
  TString aSystematicsFileLocation;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if     (aFitValWriterInfo[iAn].analysisType==kLamKchP || aFitValWriterInfo[iAn].analysisType==kALamKchM
         || aFitValWriterInfo[iAn].analysisType==kLamKchM || aFitValWriterInfo[iAn].analysisType==kALamKchP) aSystematicsFileLocation = aSystematicsFileLocation_LamKch;
    else if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) aSystematicsFileLocation = aSystematicsFileLocation_LamK0;
    else assert(0);

    FitValuesWriterwSysErrs::DrawImF0vsReF0Graph(tPadReF0vsImF0, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epzsame", "e2same", aDrawStatOnly);
    FitValuesWriterwSysErrs::DrawD0Graph(tPadD0, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, (iD0Inc+1)*tIncrementSize, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epzsame", "e2same", aDrawStatOnly);
    tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);

    if(!DescriptorAlreadyIncluded(tUsedDescriptors, tUsedMarkerStyles, aFitValWriterInfo[iAn].legendDescriptor, aFitValWriterInfo[iAn].markerStyle) && !aSuppressDescs)
    {
      tPadReF0vsImF0->cd();

      if(aFitValWriterInfo[iAn].legendDescriptor.Contains("Suppress Markers")) bSuppressMarkers=true;
      else bSuppressMarkers=false;

      tLegDesc = StripSuppressMarkersFlat(aFitValWriterInfo[iAn].legendDescriptor);

      if(tLegDesc.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tLegDesc);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
      if(!bSuppressMarkers)
      {
        tMarker->SetMarkerStyle(aFitValWriterInfo[iAn].markerStyle);
        tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      }
      iTex++;

    }
    iD0Inc++;
  }


  //------------------------------------------------------
  if(aDrawPredictions)
  {
    tPadReF0vsImF0->cd();

    int tPredColor1 = kCyan+1;
    int tPredColor2 = kMagenta;

    TGraphAsymmErrors *tGr_0607100_Set1 = new TGraphAsymmErrors(1);
      tGr_0607100_Set1->SetPoint(0, 0.17, 0.34);
      tGr_0607100_Set1->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
//      tGr_0607100_Set1->SetMarkerStyle(39);
      tGr_0607100_Set1->SetMarkerStyle(34);
      tGr_0607100_Set1->SetMarkerSize(1.5);
      tGr_0607100_Set1->SetMarkerColor(tPredColor1);
      tGr_0607100_Set1->SetLineColor(tPredColor1);
      tGr_0607100_Set1->Draw("pzsame");

    TGraphAsymmErrors *tGr_0607100_Set2 = new TGraphAsymmErrors(1);
      tGr_0607100_Set2->SetPoint(0, 0.09, 0.34);
      tGr_0607100_Set2->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
//      tGr_0607100_Set2->SetMarkerStyle(37);
      tGr_0607100_Set2->SetMarkerStyle(28);
      tGr_0607100_Set2->SetMarkerSize(1.5);
      tGr_0607100_Set2->SetMarkerColor(tPredColor1);
      tGr_0607100_Set2->SetLineColor(tPredColor1);
      tGr_0607100_Set2->Draw("pzsame");

  //-----------

    TGraphAsymmErrors *tGr_PhysRevD_KLam = new TGraphAsymmErrors(1);
      tGr_PhysRevD_KLam->SetPoint(0, 0.19, 0.14);
      tGr_PhysRevD_KLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
//      tGr_PhysRevD_KLam->SetMarkerStyle(29);
      tGr_PhysRevD_KLam->SetMarkerStyle(47);
      tGr_PhysRevD_KLam->SetMarkerSize(1.5);
      tGr_PhysRevD_KLam->SetMarkerColor(tPredColor2);
      tGr_PhysRevD_KLam->SetLineColor(tPredColor2);
      tGr_PhysRevD_KLam->Draw("pzsame");

    TGraphAsymmErrors *tGr_PhysRevD_AKLam = new TGraphAsymmErrors(1);
      tGr_PhysRevD_AKLam->SetPoint(0, 0.04, 0.18);
      tGr_PhysRevD_AKLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
//      tGr_PhysRevD_AKLam->SetMarkerStyle(30);
      tGr_PhysRevD_AKLam->SetMarkerStyle(46);
      tGr_PhysRevD_AKLam->SetMarkerSize(1.5);
      tGr_PhysRevD_AKLam->SetMarkerColor(tPredColor2);
      tGr_PhysRevD_AKLam->SetLineColor(tPredColor2);
      tGr_PhysRevD_AKLam->Draw("pzsame");

    TLegend* tLegPredictions = new TLegend(0.525, 0.625, 0.85, 0.95);
      tLegPredictions->SetLineWidth(0);
      tLegPredictions->SetFillStyle(0);
/*
      tLegPredictions->AddEntry(tGr_0607100_Set1, "[A] Set 1: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set2, "[A] Set 2: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[B] K#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[B] #bar{K}#Lambda", "p");
*/
      //tLegPredictions->AddEntry(tGr_0607100_Set1, "[10] Set 1: K#Lambda = #bar{K}#Lambda", "p");
      //tLegPredictions->AddEntry(tGr_0607100_Set2, "[10] Set 2: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set1, "[10] I:  K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set2, "[10] II: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[11] K#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[11] #bar{K}#Lambda", "p");
    tLegPredictions->SetTextSize(0.0625);
    tLegPredictions->Draw();
  }

  //------------------------------------------------------

  double tStartXStamp = -1.75;
  if(aDrawPredictions) tStartXStamp = -1.85;
  double tStartYStamp = 1.4;
  double tIncrementXStamp = 0.05;
  double tIncrementYStamp = 0.10;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  if(aDrawCircleStamps) tMarkerStyleStamp = 20;
  if(!aSuppressAnStamps) DrawAnalysisStamps((TPad*)tPadReF0vsImF0, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------ For Phys Rev C final
  TLatex* tLaText;

  double tXLett_LaTex=0.2125;
  double tYLett_LaTex=0.24;
  bool tIsNDC_LaTex=true;    
  
  int tTextAlign_LaTex = 11;
  double tLineWidth_LaTex=2;
  int tTextFont_LaTex = 62;
  double tTextSize_LaTex = 0.085;
  double tScaleFactor_LaTex = 1.0;

  tLaText = CanvasPartition::BuildTLatex(TString("(a)"), tXLett_LaTex, tYLett_LaTex, tTextAlign_LaTex, tLineWidth_LaTex, tTextFont_LaTex, tTextSize_LaTex, tScaleFactor_LaTex, tIsNDC_LaTex);
  tPadReF0vsImF0->cd();
  tLaText->Draw();

  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareLambdavsRadius(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, CentralityType aCentType, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly)
{
  TString tCanBaseName = TString::Format("CompareLambdavsRadiuswSys%s%s", aCanNameMod.Data(), cCentralityTags[aCentType]);
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //------------------------------------------------------
  double tStartX = 5.8;
  double tStartY = 0.50;
  double tIncrementX = 0.14;
  double tIncrementY = 0.11;
  double tTextSize = 0.03;

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan, 2.5, 8., 0.4, 1.49);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  //------------------------------------------------------
  //I don't want to repeat entries in the legend
  //  For instance, without the "used" check, "3 Res., Poly Bgd" would be printed for LamKchP and LamKchM (and LamK0 if it is included)
  vector<TString> tUsedDescriptors(0);
  vector<int> tUsedMarkerStyles(0);
  vector<AnalysisType> tAnTypes(0);

  //Figure out if we have only LamKchP and LamKchM separately, only LamKchP and LamKchM combined, or both
  bool tLamKchCombined=false, tLamKchSeparate=false;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) continue;
    if(aFitValWriterInfo[iAn].lamKchCombined) tLamKchCombined=true;
    else tLamKchSeparate=true;
  }

  int iTex = 0;
  TString aSystematicsFileLocation;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if     (aFitValWriterInfo[iAn].analysisType==kLamKchP || aFitValWriterInfo[iAn].analysisType==kALamKchM
         || aFitValWriterInfo[iAn].analysisType==kLamKchM || aFitValWriterInfo[iAn].analysisType==kALamKchP) aSystematicsFileLocation = aSystematicsFileLocation_LamKch;
    else if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) aSystematicsFileLocation = aSystematicsFileLocation_LamK0;
    else assert(0);

    if(!aFitValWriterInfo[iAn].lamKchCombined || aFitValWriterInfo[iAn].analysisType==kLamK0)
    {
      FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
      tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
    }
    else
    {
      //If LamKch are combined, draw only LamKchP, and draw it with purple.  Also alter the legend entry to be LamKpm
      if(aFitValWriterInfo[iAn].analysisType==kLamKchP)
      {
        int tColor = kViolet;
        FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, tColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
        tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
      }
    }

    if(!DescriptorAlreadyIncluded(tUsedDescriptors, tUsedMarkerStyles, aFitValWriterInfo[iAn].legendDescriptor, aFitValWriterInfo[iAn].markerStyle) && !aSuppressDescs)
    {
      if(aFitValWriterInfo[iAn].legendDescriptor.Contains("Suppress Markers")) bSuppressMarkers=true;
      else bSuppressMarkers=false;

      tLegDesc = StripSuppressMarkersFlat(aFitValWriterInfo[iAn].legendDescriptor);

      if(tLegDesc.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tLegDesc);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
      if(!bSuppressMarkers)
      {
        tMarker->SetMarkerStyle(aFitValWriterInfo[iAn].markerStyle);
        tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      }
      iTex++;
    }

  }
  //------------------------------------------------------

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(0.5, 2.3, cPrettyCentralityTags[aCentType]);

  //------------------------------------------------------

  double tStartXStamp = 0.7;
  double tStartYStamp = 2.1;
  double tIncrementXStamp = 0.125;
  double tIncrementYStamp = 0.15;
  double tSecondColumnShiftX = 1.0;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  int tConjMarkerStyleStamp = 25;
  if(!aSuppressAnStamps) DrawAnalysisAndConjStamps((TPad*)tReturnCan, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tSecondColumnShiftX, tTextSizeStamp, tMarkerStyleStamp, tConjMarkerStyleStamp, tLamKchCombined, tLamKchSeparate);

  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareLambdavsRadiusTweak(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, CentralityType aCentType, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly)
{
  TString tCanBaseName = TString::Format("CompareLambdavsRadiuswSys%s%s", aCanNameMod.Data(), cCentralityTags[aCentType]);
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //------------------------------------------------------
  double tStartX = 5.8;
  double tStartY = 0.50;
  double tIncrementX = 0.14;
  double tIncrementY = 0.11;
  double tTextSize = 0.0575;

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan, 2.5, 8., 0.4, 1.49);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  //------------------------------------------------------
  //I don't want to repeat entries in the legend
  //  For instance, without the "used" check, "3 Res., Poly Bgd" would be printed for LamKchP and LamKchM (and LamK0 if it is included)
  vector<TString> tUsedDescriptors(0);
  vector<int> tUsedMarkerStyles(0);
  vector<AnalysisType> tAnTypes(0);

  //Figure out if we have only LamKchP and LamKchM separately, only LamKchP and LamKchM combined, or both
  bool tLamKchCombined=false, tLamKchSeparate=false;
  bool tAllLamKCombined=false, tAllLamKCombinedDrawn=false;
  vector<TString> tAllLamKCombined_FitInfoTStringVec(0);
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if(aFitValWriterInfo[iAn].allLamKCombined) {tAllLamKCombined=true; continue;}
    if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) continue;

    if(aFitValWriterInfo[iAn].lamKchCombined) tLamKchCombined=true;
    else tLamKchSeparate=true;
  }

  int iTex = 0;
  TString aSystematicsFileLocation;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if     (aFitValWriterInfo[iAn].analysisType==kLamKchP || aFitValWriterInfo[iAn].analysisType==kALamKchM
         || aFitValWriterInfo[iAn].analysisType==kLamKchM || aFitValWriterInfo[iAn].analysisType==kALamKchP) aSystematicsFileLocation = aSystematicsFileLocation_LamKch;
    else if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) aSystematicsFileLocation = aSystematicsFileLocation_LamK0;
    else assert(0);

    if(aFitValWriterInfo[iAn].allLamKCombined)
    {
      int tColor = kOrange+1;
      tAllLamKCombinedDrawn = false;
      for(int i=0; i<tAllLamKCombined_FitInfoTStringVec.size(); i++) if(tAllLamKCombined_FitInfoTStringVec[i].EqualTo(aFitValWriterInfo[iAn].fitInfoTString)) tAllLamKCombinedDrawn=true;
      if(!tAllLamKCombinedDrawn)
      {
        FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, tColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
        tAllLamKCombined_FitInfoTStringVec.push_back(aFitValWriterInfo[iAn].fitInfoTString);
      }
    }
    else if(!aFitValWriterInfo[iAn].lamKchCombined || aFitValWriterInfo[iAn].analysisType==kLamK0)
    {
      FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
      tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
    }
    else
    {
      //If LamKch are combined, draw only LamKchP, and draw it with purple.  Also alter the legend entry to be LamKpm
      if(aFitValWriterInfo[iAn].analysisType==kLamKchP)
      {
        int tColor = kViolet;
        FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, tColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
        tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
      }
    }

    if(!DescriptorAlreadyIncluded(tUsedDescriptors, tUsedMarkerStyles, aFitValWriterInfo[iAn].legendDescriptor, aFitValWriterInfo[iAn].markerStyle) && !aSuppressDescs)
    {
      if(aFitValWriterInfo[iAn].legendDescriptor.Contains("Suppress Markers")) bSuppressMarkers=true;
      else bSuppressMarkers=false;

      tLegDesc = StripSuppressMarkersFlat(aFitValWriterInfo[iAn].legendDescriptor);

      if(tLegDesc.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tLegDesc);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
      if(!bSuppressMarkers)
      {
        tMarker->SetMarkerStyle(aFitValWriterInfo[iAn].markerStyle);
        tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      }
      iTex++;
    }

  }
  //------------------------------------------------------

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(0.5, 2.3, cPrettyCentralityTags[aCentType]);

  //------------------------------------------------------

  double tStartXStamp = 0.7;
  double tStartYStamp = 2.1;
  double tIncrementXStamp = 0.125;
  double tIncrementYStamp = 0.15;
  double tSecondColumnShiftX = 1.0;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  int tConjMarkerStyleStamp = 25;
  if(!aSuppressAnStamps) DrawAnalysisAndConjStamps((TPad*)tReturnCan, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tSecondColumnShiftX, tTextSizeStamp, tMarkerStyleStamp, tConjMarkerStyleStamp, tLamKchCombined, tLamKchSeparate, tAllLamKCombined);

  return tReturnCan;
}



//TODO Written super quickly, so real janky, should probably rewrite
//_________________________________________________________________________________________________________________________________
TCanvas* CompareLambdavsRadiusAll(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly)
{
  TString tCanBaseName = TString::Format("CompareLambdavsRadiusAllwSys%s", aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();
  tReturnCan->SetTopMargin(0.02);
  tReturnCan->SetBottomMargin(0.175);
  tReturnCan->SetRightMargin(0.02);
  tReturnCan->SetLeftMargin(0.175);
  tReturnCan->SetTicks(1,1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //------------------------------------------------------
  double tStartX = 3.2;
  double tStartY = 1.40;
  double tIncrementX = 0.20;
  double tIncrementY = 0.1;
  double tTextSize = 0.070;

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan, 2.5, 8., 0.4, 1.49);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.75;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  vector<int> tMarkerStyles{20, 21, 29};
  //------------------------------------------------------

  int iTex = 0;
  TString aSystematicsFileLocation;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  vector<int> tColors{kRed, kGreen+2, kBlue};
  aSystematicsFileLocation = aSystematicsFileLocation_LamKch;  //Until new sys. generated, just use old ones from LamKch
  for(int i=0; i<3; i++)
  {
    FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[0].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[0].fitInfoTString, aFitValWriterInfo[0].analysisType, static_cast<CentralityType>(i), tColors[i], tMarkerStyles[i], aFitValWriterInfo[0].markerSize, "epzsame", "e2same", aDrawStatOnly);

    tLegDesc = cPrettyCentralityTags[i];

    tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
    tMarker->SetMarkerStyle(tMarkerStyles[i]);
    tMarker->SetMarkerColor(tColors[i]);
    tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);

    iTex++;
  }

  //------------------------------------------------------
  tTex->SetTextSize(0.075);
  tTex->DrawLatex(3.40, 0.51, TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));

//  tTex->DrawLatex(3.0, 0.5, TString("ALICE Preliminary"));
//  tTex->DrawLatex(5.3, 0.5, TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));
  //------------------------------------------------------
  //------ For Phys Rev C final
  TLatex* tLaText;

  double tXLett_LaTex=0.2125;
  double tYLett_LaTex=0.24;
  bool tIsNDC_LaTex=true;    
  
  int tTextAlign_LaTex = 11;
  double tLineWidth_LaTex=2;
  int tTextFont_LaTex = 62;
  double tTextSize_LaTex = 0.085;
  double tScaleFactor_LaTex = 1.0;

  tLaText = CanvasPartition::BuildTLatex(TString("(b)"), tXLett_LaTex, tYLett_LaTex, tTextAlign_LaTex, tLineWidth_LaTex, tTextFont_LaTex, tTextSize_LaTex, tScaleFactor_LaTex, tIsNDC_LaTex);
  tReturnCan->cd();
  tLaText->Draw();

  return tReturnCan;
}



//_________________________________________________________________________________________________________________________________
TCanvas* CompareAll(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aDrawStatOnly)
{
  TCanvas *tReturnCan, *tCanImF0vsReF0, *tCanLamvsR0010, *tCanLamvsR1030, *tCanLamvsR3050;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, aDrawPredictions, tSubCanNameMod, false, false, aDrawStatOnly);
  tCanLamvsR0010 = CompareLambdavsRadius(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k0010, tSubCanNameMod, true, false, aDrawStatOnly);
  tCanLamvsR1030 = CompareLambdavsRadius(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k1030, tSubCanNameMod, true, true, aDrawStatOnly);
  tCanLamvsR3050 = CompareLambdavsRadius(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k3050, tSubCanNameMod, true, true, aDrawStatOnly);

  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareAllScattParamswSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  tReturnCan = new TCanvas(tCanName, tCanName, 1400, 1000);
  tReturnCan->Divide(2, 2, 0.001, 0.001);
  tReturnCan->cd(1);
  tCanImF0vsReF0->DrawClonePad();
  tReturnCan->cd(2);
  tCanLamvsR0010->DrawClonePad();
  tReturnCan->cd(3);
  tCanLamvsR1030->DrawClonePad();
  tReturnCan->cd(4);
  tCanLamvsR3050->DrawClonePad();

  //--------------------
  delete tCanImF0vsReF0;
  delete tCanLamvsR0010;
  delete tCanLamvsR1030;
  delete tCanLamvsR3050;
  //--------------------
  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareAllTweak(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aDrawStatOnly)
{
  TCanvas *tReturnCan, *tCanImF0vsReF0, *tCanLamvsR0010, *tCanLamvsR1030, *tCanLamvsR3050;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, aDrawPredictions, tSubCanNameMod, false, false, aDrawStatOnly);
  tCanLamvsR0010 = CompareLambdavsRadiusTweak(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k0010, tSubCanNameMod, true, false, aDrawStatOnly);
  tCanLamvsR1030 = CompareLambdavsRadiusTweak(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k1030, tSubCanNameMod, true, true, aDrawStatOnly);
  tCanLamvsR3050 = CompareLambdavsRadiusTweak(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, k3050, tSubCanNameMod, true, true, aDrawStatOnly);

  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareAllScattParamswSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  tReturnCan = new TCanvas(tCanName, tCanName, 1400, 1000);
  tReturnCan->Divide(2, 2, 0.001, 0.001);
  tReturnCan->cd(1);
  tCanImF0vsReF0->DrawClonePad();
  tReturnCan->cd(2);
  tCanLamvsR0010->DrawClonePad();
  tReturnCan->cd(3);
  tCanLamvsR1030->DrawClonePad();
  tReturnCan->cd(4);
  tCanLamvsR3050->DrawClonePad();

  //--------------------
  delete tCanImF0vsReF0;
  delete tCanLamvsR0010;
  delete tCanLamvsR1030;
  delete tCanLamvsR3050;
  //--------------------
  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareAll2Panel(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aDrawStatOnly, bool aDrawVertical)
{
  TCanvas *tReturnCan, *tCanImF0vsReF0, *tCanLamvsR;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, aDrawPredictions, tSubCanNameMod, false, true, aDrawStatOnly, true);
  tCanLamvsR = CompareLambdavsRadiusAll(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, tSubCanNameMod, true, false, aDrawStatOnly);

  assert(aFitValWriterInfo.size()==3);
//  vector<AnalysisType> tAnTypes{kLamKchP, kLamKchM, kLamK0};
  vector<int> tMarkerStyles{aFitValWriterInfo[0].markerStyle, aFitValWriterInfo[1].markerStyle, aFitValWriterInfo[2].markerStyle};
    double tStartXStamp = -1.75;
    if(aDrawPredictions) tStartXStamp = -1.8;
    double tStartYStamp = 1.35;
    double tIncrementXStamp = 0.075;
    double tIncrementYStamp = 0.175;
    double tTextSizeStamp = 0.060;
  TString tPadName = TString::Format("tPadReF0vsImF0%s", tCanImF0vsReF0->GetName());
  TPad* tTestPad = (TPad*)tCanImF0vsReF0->GetPrimitive(tPadName);
//  DrawAnalysisStamps(tTestPad, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyles);

  TString tSysTypeText1 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kALamKchM]);
  TString tSysTypeText2 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamKchM], cAnalysisRootTags[kALamKchP]);
  TString tSysTypeText3 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamK0], cAnalysisRootTags[kALamK0]);
  vector<TString> tSysTexts{tSysTypeText1, tSysTypeText2, tSysTypeText3};
  vector<int> tColors{kRed+1, kBlue+1, kBlack};
  DrawAnalysisStamps(tTestPad, tSysTexts, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyles, tColors);

  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareAll2PanelScattParamswSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;
  if(aDrawVertical) tCanName += TString("_Vertical");

  if(!aDrawVertical) 
  {
    tReturnCan = new TCanvas(tCanName, tCanName, 1400, 500);
    tReturnCan->Divide(2, 1, 0.01, 0.001);
  }
  else
  {
    tReturnCan = new TCanvas(tCanName, tCanName, 700, 1000);
    tReturnCan->Divide(1, 2, 0.001, 0.001);
  }
  tReturnCan->cd(1);
  tCanImF0vsReF0->DrawClonePad();
  tReturnCan->cd(2);
  tCanLamvsR->DrawClonePad();


  //--------------------
  delete tCanImF0vsReF0;
  delete tCanLamvsR;
  //--------------------
  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
vector<TCanvas*> CompareAll2Panel_separate(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aDrawStatOnly)
{
  TCanvas *tCanImF0vsReF0, *tCanLamvsR;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, aDrawPredictions, tSubCanNameMod, false, true, aDrawStatOnly, true);
  tCanLamvsR = CompareLambdavsRadiusAll(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, tSubCanNameMod, true, false, aDrawStatOnly);

  assert(aFitValWriterInfo.size()==3);
//  vector<AnalysisType> tAnTypes{kLamKchP, kLamKchM, kLamK0};
  vector<int> tMarkerStyles{aFitValWriterInfo[0].markerStyle, aFitValWriterInfo[1].markerStyle, aFitValWriterInfo[2].markerStyle};
    double tStartXStamp = -1.75;
    if(aDrawPredictions) tStartXStamp = -1.8;
    double tStartYStamp = 1.35;
    double tIncrementXStamp = 0.075;
    double tIncrementYStamp = 0.20;
    double tTextSizeStamp = 0.075;
  TString tPadName = TString::Format("tPadReF0vsImF0%s", tCanImF0vsReF0->GetName());
  TPad* tTestPad = (TPad*)tCanImF0vsReF0->GetPrimitive(tPadName);
//  DrawAnalysisStamps(tTestPad, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyles);

  TString tSysTypeText1 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kALamKchM]);
  TString tSysTypeText2 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamKchM], cAnalysisRootTags[kALamKchP]);
  TString tSysTypeText3 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[kLamK0], cAnalysisRootTags[kALamK0]);
  vector<TString> tSysTexts{tSysTypeText1, tSysTypeText2, tSysTypeText3};
  vector<int> tColors{kRed+1, kBlue+1, kBlack};
  DrawAnalysisStamps(tTestPad, tSysTexts, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyles, tColors);
  //--------------------
  vector<TCanvas*> tReturnVec{tCanImF0vsReF0, tCanLamvsR};
  return tReturnVec;
}

