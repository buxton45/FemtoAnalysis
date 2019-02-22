#include "CompareFittingMethodswSysErrs.h"


//_________________________________________________________________________________________________________________________________
TCanvas* CompareImF0vsReF0(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly)
{
  CentralityType tCentType = k0010;  //Doesn't matter which centrality chosen, because all share same scattering parameters

  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareImF0vsReF0wSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();

  TPad* tPadReF0vsImF0 = new TPad(TString::Format("tPadReF0vsImF0%s", tCanName.Data()), TString::Format("tPadReF0vsImF0%s", tCanName.Data()), 
                                  0.0, 0.0, 0.8, 1.0);
  tPadReF0vsImF0->SetRightMargin(0.01);
  tPadReF0vsImF0->Draw();

  TPad* tPadD0 = new TPad(TString::Format("tPadD0%s", tCanName.Data()), TString::Format("tPadD0%s", tCanName.Data()), 
                          0.8, 0.0, 1.0, 1.0);
  tPadD0->SetRightMargin(0.4);
  tPadD0->SetLeftMargin(0.);
  tPadD0->Draw();

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupReF0vsImF0AndD0Axes(tPadReF0vsImF0, tPadD0);

  //------------------------------------------------------
  double tStartX = -1.;
  double tStartY = 1.4;
  double tIncrementX = 0.075;
  double tIncrementY = 0.10;
  double tTextSize = 0.03;

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

    FitValuesWriterwSysErrs::DrawImF0vsReF0Graph(tPadReF0vsImF0, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
    FitValuesWriterwSysErrs::DrawD0Graph(tPadD0, aFitValWriterInfo[iAn].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, (iD0Inc+1)*tIncrementSize, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, "epsame", "e2same", aDrawStatOnly);
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

    int tPredColor1 = kCyan;
    int tPredColor2 = kMagenta;

    TGraphAsymmErrors *tGr_0607100_Set1 = new TGraphAsymmErrors(1);
      tGr_0607100_Set1->SetPoint(0, 0.17, 0.34);
      tGr_0607100_Set1->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
      tGr_0607100_Set1->SetMarkerStyle(39);
      tGr_0607100_Set1->SetMarkerSize(1.5);
      tGr_0607100_Set1->SetMarkerColor(tPredColor1);
      tGr_0607100_Set1->SetLineColor(tPredColor1);
      tGr_0607100_Set1->Draw("pzsame");

    TGraphAsymmErrors *tGr_0607100_Set2 = new TGraphAsymmErrors(1);
      tGr_0607100_Set2->SetPoint(0, 0.09, 0.34);
      tGr_0607100_Set2->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
      tGr_0607100_Set2->SetMarkerStyle(37);
      tGr_0607100_Set2->SetMarkerSize(1.5);
      tGr_0607100_Set2->SetMarkerColor(tPredColor1);
      tGr_0607100_Set2->SetLineColor(tPredColor1);
      tGr_0607100_Set2->Draw("pzsame");

  //-----------

    TGraphAsymmErrors *tGr_PhysRevD_KLam = new TGraphAsymmErrors(1);
      tGr_PhysRevD_KLam->SetPoint(0, 0.19, 0.14);
      tGr_PhysRevD_KLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
      tGr_PhysRevD_KLam->SetMarkerStyle(29);
      tGr_PhysRevD_KLam->SetMarkerSize(1.5);
      tGr_PhysRevD_KLam->SetMarkerColor(tPredColor2);
      tGr_PhysRevD_KLam->SetLineColor(tPredColor2);
      tGr_PhysRevD_KLam->Draw("pzsame");

    TGraphAsymmErrors *tGr_PhysRevD_AKLam = new TGraphAsymmErrors(1);
      tGr_PhysRevD_AKLam->SetPoint(0, 0.04, 0.18);
      tGr_PhysRevD_AKLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
      tGr_PhysRevD_AKLam->SetMarkerStyle(30);
      tGr_PhysRevD_AKLam->SetMarkerSize(1.5);
      tGr_PhysRevD_AKLam->SetMarkerColor(tPredColor2);
      tGr_PhysRevD_AKLam->SetLineColor(tPredColor2);
      tGr_PhysRevD_AKLam->Draw("pzsame");

    TLegend* tLegPredictions = new TLegend(0.825, 0.725, 0.975, 0.875);
      tLegPredictions->SetLineWidth(0);
      tLegPredictions->AddEntry(tGr_0607100_Set1, "[1] Set 1", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set2, "[1] Set 2", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[2] K#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[2] #bar{K}#Lambda", "p");
    tLegPredictions->Draw();
  }

  //------------------------------------------------------

  double tStartXStamp = -1.75;
  double tStartYStamp = 1.4;
  double tIncrementXStamp = 0.05;
  double tIncrementYStamp = 0.10;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  if(!aSuppressAnStamps) DrawAnalysisStamps((TPad*)tPadReF0vsImF0, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);



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

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan);

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


//TODO Written super quickly, so real janky, should probably rewrite
//_________________________________________________________________________________________________________________________________
TCanvas* CompareLambdavsRadiusAll(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps, bool aDrawStatOnly)
{
  TString tCanBaseName = TString::Format("CompareLambdavsRadiusAllwSys%s", aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  TCanvas* tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //------------------------------------------------------
  double tStartX = 1.0;
  double tStartY = 2.3;
  double tIncrementX = 0.20;
  double tIncrementY = 0.25;
  double tTextSize = 0.05;

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan);

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

  int iTex = 0;
  TString aSystematicsFileLocation;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  vector<int> tColors{kRed, kGreen+2, kBlue};
  aSystematicsFileLocation = aSystematicsFileLocation_LamKch;  //Until new sys. generated, just use old ones from LamKch
  for(int i=0; i<3; i++)
  {
    FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph((TPad*)tReturnCan, aFitValWriterInfo[0].masterFileLocation, aSystematicsFileLocation, aFitValWriterInfo[0].fitInfoTString, aFitValWriterInfo[0].analysisType, static_cast<CentralityType>(i), tColors[i], aFitValWriterInfo[0].markerStyle, aFitValWriterInfo[0].markerSize, "epsame", "e2same", aDrawStatOnly);

    tLegDesc = cPrettyCentralityTags[i];

    tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
    tMarker->SetMarkerStyle(aFitValWriterInfo[0].markerStyle);
    tMarker->SetMarkerColor(tColors[i]);
    tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);

    iTex++;
  }

  //------------------------------------------------------

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
//  tTex->DrawLatex(0.5, 2.3, cPrettyCentralityTags[aCentType]);

  //------------------------------------------------------

  double tStartXStamp = 0.7;
  double tStartYStamp = 2.1;
  double tIncrementXStamp = 0.125;
  double tIncrementYStamp = 0.15;
  double tSecondColumnShiftX = 1.0;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  int tConjMarkerStyleStamp = 25;

  //--------------------------------------
/*
  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  tMarker->SetMarkerStyle(aMarkerStyle);
  tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cPrettyCentralityTags[0]);
  tMarker->SetMarkerColor(kGreen);
  tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  iTex++;

  tMarker->SetMarkerStyle(aMarkerStyle);
  tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cPrettyCentralityTags[0]);
  tMarker->SetMarkerColor(kGreen);
  tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  iTex++;
*/

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
TCanvas* CompareAll2Panel(vector<FitValWriterInfo> &aFitValWriterInfo, TString aSystematicsFileLocation_LamKch, TString aSystematicsFileLocation_LamK0, bool aDrawPredictions, TString aCanNameMod, bool aDrawStatOnly)
{
  TCanvas *tReturnCan, *tCanImF0vsReF0, *tCanLamvsR;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, aDrawPredictions, tSubCanNameMod, false, false, aDrawStatOnly);
  tCanLamvsR = CompareLambdavsRadiusAll(aFitValWriterInfo, aSystematicsFileLocation_LamKch, aSystematicsFileLocation_LamK0, tSubCanNameMod, true, false, aDrawStatOnly);


  vector<TString> twPredVec{"", "_wPredictions"};
  TString tCanBaseName = TString::Format("CompareAll2PanelScattParamswSys%s%s", twPredVec[aDrawPredictions].Data(), aCanNameMod.Data());
  TString tCanName = tCanBaseName;

  tReturnCan = new TCanvas(tCanName, tCanName, 1400, 500);
  tReturnCan->Divide(2, 1, 0.001, 0.001);
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

