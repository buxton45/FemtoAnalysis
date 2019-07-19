#include "CompareFittingMethods.h"

//_________________________________________________________________________________________________________________________________
void DrawAnalysisStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize, int aMarkerStyle)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(aMarkerStyle);

  bool bIncLamKchP=false;
  bool bIncLamKchM=false;
  bool bIncLamK0=false;
  for(unsigned int i=0; i<aAnTypes.size(); i++)
  {
    if     (aAnTypes[i]==kLamKchP) bIncLamKchP=true;
    else if(aAnTypes[i]==kLamKchM) bIncLamKchM=true;
    else if(aAnTypes[i]==kLamK0) bIncLamK0=true;
  }



  int iTex = 0;

  if(bIncLamKchP)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchP]);
    tMarker->SetMarkerColor(kRed+1);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }

  if(bIncLamKchM)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchM]);
    tMarker->SetMarkerColor(kBlue+1);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }

  if(bIncLamK0)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamK0]);
    tMarker->SetMarkerColor(kBlack);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }
}

//_________________________________________________________________________________________________________________________________
void DrawAnalysisStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize, vector<int> &aMarkerStyles)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  bool bIncLamKchP=false;
  bool bIncLamKchM=false;
  bool bIncLamK0=false;
  for(unsigned int i=0; i<aAnTypes.size(); i++)
  {
    if     (aAnTypes[i]==kLamKchP) bIncLamKchP=true;
    else if(aAnTypes[i]==kLamKchM) bIncLamKchM=true;
    else if(aAnTypes[i]==kLamK0) bIncLamK0=true;
  }



  int iTex = 0;

  if(bIncLamKchP)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchP]);
    tMarker->SetMarkerColor(kRed+1);
    tMarker->SetMarkerStyle(aMarkerStyles[iTex]);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }

  if(bIncLamKchM)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchM]);
    tMarker->SetMarkerColor(kBlue+1);
    tMarker->SetMarkerStyle(aMarkerStyles[iTex]);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }

  if(bIncLamK0)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamK0]);
    tMarker->SetMarkerColor(kBlack);
    tMarker->SetMarkerStyle(aMarkerStyles[iTex]);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }
}

//_________________________________________________________________________________________________________________________________
void DrawAnalysisStamps(TPad* aPad, vector<TString> &aAnTexts, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize, vector<int> &aMarkerStyles, vector<int> &aMarkerColors)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  assert(aAnTexts.size()==aMarkerStyles.size());
  assert(aAnTexts.size()==aMarkerColors.size());

  int iTex = 0;
  for(int i=0; i<aAnTexts.size(); i++)
  {
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, aAnTexts[i]);
    tMarker->SetMarkerColor(aMarkerColors[i]);
    tMarker->SetMarkerStyle(aMarkerStyles[i]);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
    iTex++;
  }
}

//_________________________________________________________________________________________________________________________________
void DrawAnalysisAndConjStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aSecondColumnShiftX, double aTextSize, int aMarkerStyle, int aConjMarkerStyle, bool aLamKchCombined, bool aLamKchSeparate, bool aAllLamKCombined)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  bool bIncLamKchP=false, bIncALamKchM=false;
  bool bIncLamKchM=false, bIncALamKchP=false;
  bool bIncLamK0=false, bIncALamK0=false;
  for(unsigned int i=0; i<aAnTypes.size(); i++)
  {
    if     (aAnTypes[i]==kLamKchP) bIncLamKchP=true;
    else if(aAnTypes[i]==kALamKchM) bIncALamKchM=true;

    else if(aAnTypes[i]==kLamKchM) bIncLamKchM=true;
    else if(aAnTypes[i]==kALamKchP) bIncALamKchP=true;

    else if(aAnTypes[i]==kLamK0) bIncLamK0=true;
    else if(aAnTypes[i]==kALamK0) bIncALamK0=true;
  }

  if(aLamKchCombined && !aLamKchSeparate)
  {
    bIncLamKchP = false;
    bIncLamKchM = false;
  }


  int iTex = 0;
  //----------
  if(aAllLamKCombined)
  {
    if(!aLamKchSeparate)
    {
      bIncLamKchP = false;
      bIncLamKchM = false;
    }

    tMarker->SetMarkerStyle(aMarkerStyle);
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, "All #LambdaK");
    tMarker->SetMarkerColor(kOrange+1);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);

    iTex++;
  }
  //----------
  if(aLamKchCombined)
  {
    tMarker->SetMarkerStyle(aMarkerStyle);
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, "#LambdaK^{#pm}");
    tMarker->SetMarkerColor(kViolet);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);

    iTex++;
  }
  //----------
  if(bIncLamKchP)
  {
    tMarker->SetMarkerStyle(aMarkerStyle);
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchP]);
    tMarker->SetMarkerColor(kRed+1);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncALamKchM)
  {
    tMarker->SetMarkerStyle(aConjMarkerStyle);
    tTex->DrawLatex(aSecondColumnShiftX+aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kALamKchM]);
    tMarker->SetMarkerColor(kRed+1);
    tMarker->DrawMarker(aSecondColumnShiftX+aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncLamKchP || bIncALamKchM) iTex++;

  //----------

  if(bIncLamKchM)
  {
    tMarker->SetMarkerStyle(aMarkerStyle);
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchM]);
    tMarker->SetMarkerColor(kBlue+1);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncALamKchP)
  {
    tMarker->SetMarkerStyle(aConjMarkerStyle);
    tTex->DrawLatex(aSecondColumnShiftX+aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kALamKchP]);
    tMarker->SetMarkerColor(kBlue+1);
    tMarker->DrawMarker(aSecondColumnShiftX+aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncLamKchM || bIncALamKchP) iTex++;

  //----------
  if(bIncLamK0)
  {
    tMarker->SetMarkerStyle(aMarkerStyle);
    tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamK0]);
    tMarker->SetMarkerColor(kBlack);
    tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncALamK0)
  {
    tMarker->SetMarkerStyle(aConjMarkerStyle);
    tTex->DrawLatex(aSecondColumnShiftX+aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kALamK0]);
    tMarker->SetMarkerColor(kBlack);
    tMarker->DrawMarker(aSecondColumnShiftX+aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  }
  if(bIncLamK0 || bIncALamK0) iTex++;
}

//_________________________________________________________________________________________________________________________________
void DrawFixedRadiiStamps(TPad* aPad, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize, int aMarkerStyle)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(aMarkerStyle);

  int iTex = 0;

  tTex->DrawLatex(aStartX, aStartY-(iTex-0.75)*aIncrementY, "Fixed Radii");

  tMarker->SetMarkerColor(kPink+10);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;

  tMarker->SetMarkerColor(kAzure+10);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;

  tMarker->SetMarkerColor(kGray+1);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;
}


//_________________________________________________________________________________________________________________________________
void SetupReF0vsImF0Axes(TPad* aPad, double aMinReF0, double aMaxReF0, double aMinImF0, double aMaxImF0)
{
  aPad->cd();
  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinReF0, aMaxReF0);
  tTrash->GetXaxis()->SetRangeUser(aMinReF0, aMaxReF0);
  tTrash->GetYaxis()->SetRangeUser(aMinImF0, aMaxImF0);

  tTrash->GetXaxis()->SetTitle("Re[#it{f}_{0}]");
  tTrash->GetYaxis()->SetTitle("Im[#it{f}_{0}]");

  tTrash->DrawCopy("axis");
  tTrash->Delete();
}


//_________________________________________________________________________________________________________________________________
void SetupReF0vsImF0AndD0Axes(TPad* aPadReF0vsImF0, TPad* aPadD0, 
                              double aMinReF0, double aMaxReF0, double aMinImF0, double aMaxImF0,
                              double aMinD0, double aMaxD0)
{
  aPadReF0vsImF0->cd();
  TH1D* tTrash1 = new TH1D("tTrash1", "tTrash1", 10, aMinReF0, aMaxReF0);
  tTrash1->GetXaxis()->SetRangeUser(aMinReF0, aMaxReF0);
  tTrash1->GetYaxis()->SetRangeUser(aMinImF0, aMaxImF0);

  tTrash1->GetXaxis()->SetTitle("#Rgothic#it{f}_{0} (fm)");
  tTrash1->GetXaxis()->SetTitleOffset(1.0);
  tTrash1->GetXaxis()->SetTitleSize(0.06);
  tTrash1->GetXaxis()->SetLabelSize(1.25*tTrash1->GetXaxis()->GetLabelSize());

  tTrash1->GetYaxis()->SetTitle("#Jgothic#it{f}_{0} (fm)");
  tTrash1->GetYaxis()->SetTitleOffset(1.0);
  tTrash1->GetYaxis()->SetTitleSize(0.06);
  tTrash1->GetYaxis()->SetLabelSize(1.25*tTrash1->GetYaxis()->GetLabelSize());

  tTrash1->DrawCopy("axis");

  //------------------------
  aPadD0->cd();
  TH1D* tTrash2 = new TH1D("tTrash2", "tTrash2", 1, 0., 1.);
  tTrash2->GetXaxis()->SetRangeUser(0., 1.);
  tTrash2->GetYaxis()->SetRangeUser(aMinD0, aMaxD0);

  double tScale = aPadReF0vsImF0->GetAbsWNDC()/aPadD0->GetAbsWNDC();
  tTrash2->GetYaxis()->SetLabelSize(tScale*tTrash1->GetYaxis()->GetLabelSize());
  tTrash2->GetYaxis()->SetLabelOffset(0.05);

  tTrash2->GetYaxis()->SetTitle("#it{d}_{0} (fm)");
  tTrash2->GetYaxis()->SetTitleSize(tScale*tTrash1->GetYaxis()->GetTitleSize());
  tTrash2->GetYaxis()->SetTitleOffset(0.90);
  tTrash2->GetYaxis()->RotateTitle(true);
  tTrash2->GetYaxis()->SetTickLength(0.025*tScale);

  tTrash2->GetXaxis()->SetLabelSize(0.0);
  tTrash2->GetXaxis()->SetTickLength(0.0);

  tTrash2->DrawCopy("Y+ axis");
  //------------------------
  tTrash1->Delete();
  tTrash2->Delete();
}


//_________________________________________________________________________________________________________________________________
void SetupRadiusvsLambdaAxes(TPad* aPad, double aMinR, double aMaxR, double aMinLam, double aMaxLam)
{
  aPad->cd();
  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinR, aMaxR);
  tTrash->GetXaxis()->SetRangeUser(aMinR, aMaxR);
  tTrash->GetYaxis()->SetRangeUser(aMinLam, aMaxLam);

  tTrash->GetXaxis()->SetTitle("#it{R}_{inv} (fm)");
  tTrash->GetXaxis()->SetTitleOffset(1.0);
  tTrash->GetXaxis()->SetTitleSize(0.06);
  tTrash->GetXaxis()->SetLabelSize(1.25*tTrash->GetXaxis()->GetLabelSize());

  tTrash->GetYaxis()->SetTitle("#it{#lambda}_{Fit}");
  tTrash->GetYaxis()->SetTitleOffset(0.65);
  tTrash->GetYaxis()->SetTitleSize(0.07);
  tTrash->GetYaxis()->SetLabelSize(1.25*tTrash->GetYaxis()->GetLabelSize()); 

  tTrash->DrawCopy("axis");
  tTrash->Delete();
}

//_________________________________________________________________________________________________________________________________
bool MarkerStylesConsistent(int aMarkerStyle1, int aMarkerStyle2)
{
  if(aMarkerStyle1==aMarkerStyle1) return true;

  //"Conjugate" symbols are also considered consistent
  bool tAreConsistent = false;

  if     ( (aMarkerStyle1==20 && aMarkerStyle2==24) || (aMarkerStyle1==24 && aMarkerStyle2==20) ) tAreConsistent=true;
  else if( (aMarkerStyle1==21 && aMarkerStyle2==25) || (aMarkerStyle1==25 && aMarkerStyle2==21) ) tAreConsistent=true;
  else if( (aMarkerStyle1==22 && aMarkerStyle2==26) || (aMarkerStyle1==26 && aMarkerStyle2==22) ) tAreConsistent=true;
  else if( (aMarkerStyle1==23 && aMarkerStyle2==32) || (aMarkerStyle1==32 && aMarkerStyle2==23) ) tAreConsistent=true;
  else if( (aMarkerStyle1==29 && aMarkerStyle2==30) || (aMarkerStyle1==30 && aMarkerStyle2==29) ) tAreConsistent=true;
  else if( (aMarkerStyle1==33 && aMarkerStyle2==27) || (aMarkerStyle1==27 && aMarkerStyle2==33) ) tAreConsistent=true;
  else if( (aMarkerStyle1==34 && aMarkerStyle2==28) || (aMarkerStyle1==28 && aMarkerStyle2==34) ) tAreConsistent=true;

  return tAreConsistent;
}

//_________________________________________________________________________________________________________________________________
bool DescriptorAlreadyIncluded(vector<TString> &aUsedDescriptors, vector<int> &aUsedMarkerStyles, TString aNewDescriptor, int aNewMarkerStyle)
{
  assert(aUsedDescriptors.size()==aUsedMarkerStyles.size());
  bool tAlreadyIncluded = false;
  for(unsigned int i=0; i<aUsedDescriptors.size(); i++)
  {
    if(aUsedDescriptors[i].EqualTo(aNewDescriptor))
    {
      tAlreadyIncluded = true;
      assert(MarkerStylesConsistent(aUsedMarkerStyles[i], aNewMarkerStyle));  //ensure marker styles are consistent
    }
  }
  if(!tAlreadyIncluded)
  {
    aUsedDescriptors.push_back(aNewDescriptor);
    aUsedMarkerStyles.push_back(aNewMarkerStyle);
  }
  return tAlreadyIncluded;
}

//_________________________________________________________________________________________________________________________________
TString StripSuppressMarkersFlat(TString aString)
{
  if(!aString.Contains("Suppress Markers")) return aString;

  TObjArray* tContents = aString.Tokenize("(");
  return ((TObjString*)tContents->At(0))->String().Strip(TString::kBoth, ' ');
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareImF0vsReF0(vector<FitValWriterInfo> &aFitValWriterInfo, bool aDrawPredictions, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps)
{
  CentralityType tCentType = k0010;  //Doesn't matter which centrality chosen, because all share same scattering parameters

  TString tCanBaseName = TString::Format("CompareImF0vsReF0%s", aCanNameMod.Data());
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
  if(aDrawPredictions) tStartX = -1.35;
  double tStartY = 1.4;
  double tIncrementX = 0.075;
  double tIncrementY = 0.10;
  double tTextSize = 0.035;

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
  TString tDrawOption = "epsame";

  double tD0XOffset = 0.5;

  //I don't want to repeat entries in the legend
  //  For instance, without the "used" check, "3 Res., Poly Bgd" would be printed for LamKchP and LamKchM (and LamK0 if it is included)
  vector<TString> tUsedDescriptors(0);
  vector<int> tUsedMarkerStyles(0);
  vector<AnalysisType> tAnTypes(0);

  int iTex = 0;
  int iD0Inc = 0;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    FitValuesWriter::DrawImF0vsReF0GraphStat(tPadReF0vsImF0, aFitValWriterInfo[iAn].masterFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, tDrawOption);
    FitValuesWriter::DrawD0GraphStat(tPadD0, aFitValWriterInfo[iAn].masterFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, tCentType, tD0XOffset, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, tDrawOption);
    tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);

    if(!DescriptorAlreadyIncluded(tUsedDescriptors, tUsedMarkerStyles, aFitValWriterInfo[iAn].legendDescriptor, aFitValWriterInfo[iAn].markerStyle) && !aSuppressDescs)
    {
      tPadReF0vsImF0->cd();

      if(aFitValWriterInfo[iAn].legendDescriptor.Contains("Suppress Markers")) bSuppressMarkers=true;
      else bSuppressMarkers=false;

      tLegDesc = StripSuppressMarkersFlat(aFitValWriterInfo[iAn].legendDescriptor);

      if(tLegDesc.Contains("#it{d}_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tLegDesc);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tLegDesc);
      if(!bSuppressMarkers)
      {
        tMarker->SetMarkerStyle(aFitValWriterInfo[iAn].markerStyle);
        tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      }
      iTex++;
      iD0Inc++;
    }
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

    TLegend* tLegPredictions = new TLegend(0.60, 0.725, 0.975, 0.875);
      tLegPredictions->SetLineWidth(0);
      tLegPredictions->AddEntry(tGr_0607100_Set1, "[A] Set 1: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set2, "[A] Set 2: K#Lambda = #bar{K}#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[B] K#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[B] #bar{K}#Lambda", "p");
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
  if(!aSuppressAnStamps) DrawAnalysisStamps((TPad*)tPadReF0vsImF0, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);



  return tReturnCan;
}

//_________________________________________________________________________________________________________________________________
TCanvas* CompareLambdavsRadius(vector<FitValWriterInfo> &aFitValWriterInfo, CentralityType aCentType, TString aCanNameMod, bool aSuppressDescs, bool aSuppressAnStamps)
{
  TString tCanBaseName = TString::Format("CompareLambdavsRadius%s%s", aCanNameMod.Data(), cCentralityTags[aCentType]);
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
  TString tDrawOption = "epsame";

  //I don't want to repeat entries in the legend
  //  For instance, without the "used" check, "3 Res., Poly Bgd" would be printed for LamKchP and LamKchM (and LamK0 if it is included)
  vector<TString> tUsedDescriptors(0);
  vector<int> tUsedMarkerStyles(0);
  vector<AnalysisType> tAnTypes(0);

  //Figure out if we have only LamKchP and LamKchM separately, only LamKchP and LamKchM combined, or both
  bool tLamKchCombined=false, tLamKchSeparate=false;
  bool tAllLamKCombined=false, tAllLamKCombinedDrawn=false;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if(aFitValWriterInfo[iAn].allLamKCombined) {tAllLamKCombined=true; continue;}
    if(aFitValWriterInfo[iAn].analysisType==kLamK0 || aFitValWriterInfo[iAn].analysisType==kALamK0) continue;

    if(aFitValWriterInfo[iAn].lamKchCombined) tLamKchCombined=true;
    else tLamKchSeparate=true;
  }

  int iTex = 0;
  bool bSuppressMarkers = false;
  TString tLegDesc;
  for(unsigned int iAn=0; iAn<aFitValWriterInfo.size(); iAn++)
  {
    if(aFitValWriterInfo[iAn].allLamKCombined)
    {
      int tColor = kOrange+1;
      if(!tAllLamKCombinedDrawn)
      {
        FitValuesWriter::DrawLambdavsRadiusGraphStat((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, tColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, tDrawOption);
        tAllLamKCombinedDrawn=true;
      }
    }
    else if(!aFitValWriterInfo[iAn].lamKchCombined || aFitValWriterInfo[iAn].analysisType==kLamK0)
    {
      FitValuesWriter::DrawLambdavsRadiusGraphStat((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, aFitValWriterInfo[iAn].markerColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, tDrawOption);
      tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
    }
    else
    {
      //If LamKch are combined, draw only LamKchP, and draw it with purple.  Also alter the legend entry to be LamKpm
      if(aFitValWriterInfo[iAn].analysisType==kLamKchP)
      {
        int tColor = kViolet;
        FitValuesWriter::DrawLambdavsRadiusGraphStat((TPad*)tReturnCan, aFitValWriterInfo[iAn].masterFileLocation, aFitValWriterInfo[iAn].fitInfoTString, aFitValWriterInfo[iAn].analysisType, aCentType, tColor, aFitValWriterInfo[iAn].markerStyle, aFitValWriterInfo[iAn].markerSize, tDrawOption);
        tAnTypes.push_back(aFitValWriterInfo[iAn].analysisType);
      }
    }





    if(!DescriptorAlreadyIncluded(tUsedDescriptors, tUsedMarkerStyles, aFitValWriterInfo[iAn].legendDescriptor, aFitValWriterInfo[iAn].markerStyle) && !aSuppressDescs)
    {
      if(aFitValWriterInfo[iAn].legendDescriptor.Contains("Suppress Markers")) bSuppressMarkers=true;
      else bSuppressMarkers=false;

      tLegDesc = StripSuppressMarkersFlat(aFitValWriterInfo[iAn].legendDescriptor);

      if(tLegDesc.Contains("#it{d}_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tLegDesc);
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
  double tIncrementYStamp = 0.125;
  double tSecondColumnShiftX = 1.0;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  int tConjMarkerStyleStamp = 25;
  if(!aSuppressAnStamps) DrawAnalysisAndConjStamps((TPad*)tReturnCan, tAnTypes, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tSecondColumnShiftX, tTextSizeStamp, tMarkerStyleStamp, tConjMarkerStyleStamp, tLamKchCombined, tLamKchSeparate, tAllLamKCombined);

  return tReturnCan;
}



//_________________________________________________________________________________________________________________________________
TCanvas* CompareAll(vector<FitValWriterInfo> &aFitValWriterInfo, bool aDrawPredictions, TString aCanNameMod)
{
  TCanvas *tReturnCan, *tCanImF0vsReF0, *tCanLamvsR0010, *tCanLamvsR1030, *tCanLamvsR3050;

  TString tSubCanNameMod = TString::Format("%sForAll", aCanNameMod.Data());
  tCanImF0vsReF0 = CompareImF0vsReF0(aFitValWriterInfo, aDrawPredictions, tSubCanNameMod, false, false);
  tCanLamvsR0010 = CompareLambdavsRadius(aFitValWriterInfo, k0010, tSubCanNameMod, true, false);
  tCanLamvsR1030 = CompareLambdavsRadius(aFitValWriterInfo, k1030, tSubCanNameMod, true, true);
  tCanLamvsR3050 = CompareLambdavsRadius(aFitValWriterInfo, k3050, tSubCanNameMod, true, true);


  TString tCanBaseName = TString::Format("CompareAllScattParams%s", aCanNameMod.Data());
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
