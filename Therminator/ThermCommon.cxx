/* ThermCommon.cxx */

#include "ThermCommon.h"


//_________________________________________________________________________________________
TH1D* Get1dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH1D *ReturnHisto = (TH1D*)f1.Get(HistoName);

  TH1D *ReturnHistoClone = (TH1D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TH2D* Get2dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH2D *ReturnHisto = (TH2D*)f1.Get(HistoName);

  TH2D *ReturnHistoClone = (TH2D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
TH3D* Get3dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH3D *ReturnHisto = (TH3D*)f1.Get(HistoName);

  TH3D *ReturnHistoClone = (TH3D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


const char* tXAxisLabels_LamKchP[13] = {cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kResSig0KchP], cAnalysisRootTags[kResXi0KchP], cAnalysisRootTags[kResXiCKchP], cAnalysisRootTags[kResSigStPKchP], cAnalysisRootTags[kResSigStMKchP], cAnalysisRootTags[kResSigSt0KchP], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Other", "Fake"};
const char* tXAxisLabels_ALamKchM[13] = {cAnalysisRootTags[kALamKchM], cAnalysisRootTags[kResASig0KchM], cAnalysisRootTags[kResAXi0KchM], cAnalysisRootTags[kResAXiCKchM], cAnalysisRootTags[kResASigStMKchM], cAnalysisRootTags[kResASigStPKchM], cAnalysisRootTags[kResASigSt0KchM], cAnalysisRootTags[kResALamAKSt0], cAnalysisRootTags[kResASig0AKSt0], cAnalysisRootTags[kResAXi0AKSt0], cAnalysisRootTags[kResAXiCAKSt0], "Other", "Fake"};

const char* tXAxisLabels_LamKchM[13] = {cAnalysisRootTags[kLamKchM], cAnalysisRootTags[kResSig0KchM], cAnalysisRootTags[kResXi0KchM], cAnalysisRootTags[kResXiCKchM], cAnalysisRootTags[kResSigStPKchM], cAnalysisRootTags[kResSigStMKchM], cAnalysisRootTags[kResSigSt0KchM], cAnalysisRootTags[kResLamAKSt0], cAnalysisRootTags[kResSig0AKSt0], cAnalysisRootTags[kResXi0AKSt0], cAnalysisRootTags[kResXiCAKSt0], "Other", "Fake"};
const char* tXAxisLabels_ALamKchP[13] = {cAnalysisRootTags[kALamKchP], cAnalysisRootTags[kResASig0KchP], cAnalysisRootTags[kResAXi0KchP], cAnalysisRootTags[kResAXiCKchP], cAnalysisRootTags[kResASigStMKchP], cAnalysisRootTags[kResASigStPKchP], cAnalysisRootTags[kResASigSt0KchP], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Other", "Fake"}; 

const char* tXAxisLabels_LamK0[13] = {cAnalysisRootTags[kLamK0], cAnalysisRootTags[kResSig0K0], cAnalysisRootTags[kResXi0K0], cAnalysisRootTags[kResXiCK0], cAnalysisRootTags[kResSigStPK0], cAnalysisRootTags[kResSigStMK0], cAnalysisRootTags[kResSigSt0K0], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Other", "Fake"};
const char* tXAxisLabels_ALamK0[13] = {cAnalysisRootTags[kALamK0], cAnalysisRootTags[kResASig0K0], cAnalysisRootTags[kResAXi0K0], cAnalysisRootTags[kResAXiCK0], cAnalysisRootTags[kResASigStMK0], cAnalysisRootTags[kResASigStPK0], cAnalysisRootTags[kResASigSt0K0], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Other", "Fake"}; 
//________________________________________________________________________________________________________________
void SetXAxisLabels(AnalysisType aAnType, TH1D* aHist)
{
  const char** tLabels;

  switch(aAnType) {
  case kLamKchP:
    tLabels = tXAxisLabels_LamKchP;
    break;

  case kALamKchM:
    tLabels = tXAxisLabels_ALamKchM;
    break;

  case kLamKchM:
    tLabels = tXAxisLabels_LamKchM;
    break;

  case kALamKchP:
    tLabels = tXAxisLabels_ALamKchP;
    break;

  case kLamK0:
    tLabels = tXAxisLabels_LamK0;
    break;

  case kALamK0:
    tLabels = tXAxisLabels_ALamK0;
    break;

  default:
    cout << "ERROR: SetXAxisLabels: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  for(int i=1; i<=13; i++) aHist->GetXaxis()->SetBinLabel(i, tLabels[i-1]);
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
void PrintLambdaValues(TPad* aPad, TH1D* aHisto)
{
  aPad->cd();
  TPaveText* returnText = new TPaveText(0.65,0.25,0.85,0.85,"NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->SetTextFont(63);
    returnText->SetTextSize(10);

  returnText->AddText("Estimated #lambda Values");

  double tTotal = 0.;
  for(int i=1; i<=12; i++) tTotal += aHisto->GetBinContent(i);
  for(int i=1; i<=12; i++) returnText->AddText(TString(aHisto->GetXaxis()->GetBinLabel(i)) + TString::Format(" = %0.3f", aHisto->GetBinContent(i)/tTotal));

  returnText->Draw();
}


//________________________________________________________________________________________________________________
void DrawPairFractions(TPad* aPad, TH1D* aHisto, bool aSave, TString aSaveName)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  double tNCounts = 0.;
  for(int i=1; i<=12; i++) tNCounts += aHisto->GetBinContent(i);
  double tNFakes = 0.05*tNCounts;
  aHisto->SetBinContent(13, tNFakes);

  aHisto->GetXaxis()->SetTitle("Parent System");
  aHisto->GetYaxis()->SetTitle("Counts");

  aHisto->GetXaxis()->SetTitleOffset(1.25);
  aHisto->GetYaxis()->SetTitleOffset(1.5);

  aHisto->Draw();

  PrintLambdaValues(aPad,aHisto);

  if(aSave) aPad->SaveAs(aSaveName+TString(".pdf"));
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
vector<int> GetParentsPidVector(ParticlePDGType aType)
{
  vector<int> tFathers;

  switch(aType) {
  case kPDGLam:
  case kPDGALam:
    tFathers = cAllLambdaFathers;
    break;

  case kPDGKchP:
  case kPDGKchM:
    tFathers = cAllKchFathers;
    break;

  case kPDGK0:
    tFathers = cAllK0ShortFathers;
    break;

  case kPDGProt:
  case kPDGAntiProt:
    tFathers = cAllProtonFathers;
    break;

  default:
    cout << "ERROR: GetParentsPidVector: aType = " << aType << " is not appropriate" << endl << endl;
    assert(0);
  }

  return tFathers;
}

//________________________________________________________________________________________________________________
void SetParentPidBinLabels(TAxis* aAxis, ParticlePDGType aType)
{
  vector<int> tFathers = GetParentsPidVector(aType);
  for(unsigned int i=0; i<tFathers.size(); i++) aAxis->SetBinLabel(i+1, GetParticleName(tFathers[i]));
}

//________________________________________________________________________________________________________________
void DrawParentsMatrixBackground(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix)
{
  vector<int> tBinsXToSetToZero;
  vector<int> tBinsYToSetToZero;
  switch(aAnType) {
  case kLamKchP:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{61, 63};
    break;

  case kALamKchM:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{53, 55};
    break;

  case kLamKchM:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{53, 55};
    break;

  case kALamKchP:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{61, 63};
    break;

  case kLamK0:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{42, 45};
    break;

  case kALamK0:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{42, 45};
    break;

  default:
    cout << "ERROR: DrawParentsMatrixBackground: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  //---------------------------------
  for(unsigned int i=0; i<tBinsXToSetToZero.size(); i++)
  {
    for(unsigned int j=0; j<tBinsYToSetToZero.size(); j++)
    {
      aMatrix->SetBinContent(tBinsXToSetToZero[i], tBinsYToSetToZero[j], 0.);
    }
  }

  aPad->cd();
  aPad->SetRightMargin(0.15);
  gStyle->SetOptStat(0);

  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  aMatrix->LabelsOption("v", "X");

  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  aMatrix->Draw("colz");
}


//________________________________________________________________________________________________________________
void DrawOnlyPairsInOthers(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, double aMaxDecayLength)
{
  vector<int> tParentCollection1, tParentCollection2;
  switch(aAnType) {
  case kLamKchP:
  case kALamKchM:
  case kLamKchM:
  case kALamKchP:
    tParentCollection1 = cAllLambdaFathers;
    tParentCollection2 = cAllKchFathers;
    break;

  case kLamK0:
  case kALamK0:
    tParentCollection1 = cAllLambdaFathers;
    tParentCollection2 = cAllK0ShortFathers;
    break;


  default:
    cout << "ERROR: DrawOnlyPairsInOthers: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  for(unsigned int i=0; i<tParentCollection1.size(); i++)
  {
    for(unsigned int j=0; j<tParentCollection2.size(); j++)
    {
      if(!IncludeInOthers(tParentCollection1[i], tParentCollection2[j], aMaxDecayLength)) aMatrix->SetBinContent(i+1, j+1, 0.);
    }
  }

  aPad->cd();
  aPad->SetRightMargin(0.15);
  gStyle->SetOptStat(0);

  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  aMatrix->LabelsOption("v", "X");

  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  aMatrix->Draw("colz");
}

//________________________________________________________________________________________________________________
void DrawParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aZoomROI, bool aSetLogZ, bool aSave, TString aSaveName)
{
  aPad->cd();
  aPad->SetRightMargin(0.15);
  aPad->SetLogz(aSetLogZ);
  gStyle->SetOptStat(0);

  TString tReturnName;
  if(aZoomROI) tReturnName = TString("Parents Matrix: ");
  else tReturnName = TString("Parents Matrix (Full): ");
  tReturnName += TString(cAnalysisRootTags[aAnType]);
  aMatrix->SetTitle(tReturnName);

//  aMatrix->GetXaxis()->SetTitle("Lambda Parent ID");
  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  if(aZoomROI)
  {
    if(aAnType==kLamKchP || aAnType==kLamKchM || aAnType==kLamK0) aMatrix->GetXaxis()->SetRange(50,65);
    else if(aAnType==kALamKchP || aAnType==kALamKchM || aAnType==kALamK0) aMatrix->GetXaxis()->SetRange(35,50);
    else assert(0);
    aMatrix->GetXaxis()->SetLabelSize(0.03);
  }
  aMatrix->LabelsOption("v", "X");

//  aMatrix->GetYaxis()->SetTitle("Kch Parent ID");
  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  if(aZoomROI)
  {
    if(aAnType==kLamKchP || aAnType==kALamKchP) aMatrix->GetYaxis()->SetRange(56,66);
    else if(aAnType==kLamKchM || aAnType==kALamKchM) aMatrix->GetYaxis()->SetRange(50,60);
    else if(aAnType==kLamK0 || aAnType==kALamK0) aMatrix->GetYaxis()->SetRange(38,48);
    else assert(0);
    aMatrix->GetYaxis()->SetLabelSize(0.04);
  }
  aMatrix->Draw("colz");

  if(aSave)
  {
    TString tSaveName = aSaveName;
    if(aSetLogZ) tSaveName += TString("_LogZ");
    if(!aZoomROI) tSaveName += TString("_UnZoomed");
    tSaveName += TString(".pdf");

    aPad->SaveAs(tSaveName);
  }
}

//________________________________________________________________________________________________________________
TH2D* BuildCondensedParentsMatrix(TH2D* aMatrix, TString aReturnName)
{
  TH2D* tCondensedMatrix = new TH2D(aReturnName, aReturnName, 
                                    aMatrix->GetNbinsX(), aMatrix->GetXaxis()->GetBinLowEdge(1), aMatrix->GetXaxis()->GetBinUpEdge(aMatrix->GetNbinsX()), 
                                    aMatrix->GetNbinsY(), aMatrix->GetYaxis()->GetBinLowEdge(1), aMatrix->GetYaxis()->GetBinUpEdge(aMatrix->GetNbinsY()));
  //-------------------------------------------------
  vector<int> tColumnsToSkip(0);
  vector<int> tRowsToSkip(0);

  double tCounts = 0.;
  for(int i=1; i<=aMatrix->GetNbinsX(); i++)
  {
    tCounts = 0.;
    for(int j=1; j<=aMatrix->GetNbinsY(); j++)
    {
      tCounts += aMatrix->GetBinContent(i,j);
    }
    if(tCounts==0.) tColumnsToSkip.push_back(i);
  }

  for(int j=1; j<=aMatrix->GetNbinsY(); j++)
  {
    tCounts = 0.;
    for(int i=1; i<=aMatrix->GetNbinsX(); i++)
    {
      tCounts += aMatrix->GetBinContent(i,j);
    }
    if(tCounts==0.) tRowsToSkip.push_back(j);
  }

  //-------------------------------------------------

  bool bSkipX=false;
  bool bSkipY=false;

  int tXbinTracker = 0;
  int tYbinTracker = 0;

  for(int i=1; i<=aMatrix->GetNbinsX(); i++)
  {
    bSkipX=false;
    for(int a=0; a<(int)tColumnsToSkip.size(); a++)
    {
      if(i==tColumnsToSkip[a]) bSkipX=true;
    }
    if(!bSkipX)
    {
      tXbinTracker++;
      tYbinTracker=0;
      for(int j=1; j<=aMatrix->GetNbinsY(); j++)
      {
        bSkipY=false;
        for(int b=0; b<(int)tRowsToSkip.size(); b++)
        {
          if(j==tRowsToSkip[b]) bSkipY = true;
        }
        if(!bSkipY)
        {
          tYbinTracker++;
          tCondensedMatrix->SetBinContent(tXbinTracker, tYbinTracker, aMatrix->GetBinContent(i,j));
          tCondensedMatrix->GetXaxis()->SetBinLabel(tXbinTracker, aMatrix->GetXaxis()->GetBinLabel(i));
          tCondensedMatrix->GetYaxis()->SetBinLabel(tYbinTracker, aMatrix->GetYaxis()->GetBinLabel(j));
        }
      }
    }

  }

  tCondensedMatrix->GetXaxis()->SetRange(1,tXbinTracker);
  tCondensedMatrix->GetXaxis()->SetLabelSize(0.02);

  tCondensedMatrix->GetYaxis()->SetRange(1,tYbinTracker);
  tCondensedMatrix->GetYaxis()->SetLabelSize(0.02);

  tCondensedMatrix->LabelsOption("v", "X");

  return tCondensedMatrix;
}

//________________________________________________________________________________________________________________
void DrawCondensedParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aSetLogZ, bool aSave, TString aSaveName)
{
  aPad->cd();
  aPad->SetRightMargin(0.15);
  aPad->SetTopMargin(0.075);
  aPad->SetLogz(aSetLogZ);
//  gStyle->SetOptStat(0);

  TString tReturnName = TString("Parents Matrix: ") + TString(cAnalysisRootTags[aAnType]);
  TH2D* tCondensedMatrix = BuildCondensedParentsMatrix(aMatrix, tReturnName);

  tCondensedMatrix->Draw("colz");

  if(aSave)
  {
    TString tSaveName = aSaveName;
    if(aSetLogZ) tSaveName += TString("_LogZ");
    tSaveName += TString(".pdf");

    aPad->SaveAs(tSaveName);
  }
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
TH2D* BuildCondensed2dRadiiVsBeta(TH2D* a2dHist, TString aReturnName)
{
  TH2D* tCondensed = new TH2D(aReturnName, aReturnName, 
                              a2dHist->GetNbinsX(), a2dHist->GetXaxis()->GetBinLowEdge(1), a2dHist->GetXaxis()->GetBinUpEdge(a2dHist->GetNbinsX()), 
                              a2dHist->GetNbinsY(), a2dHist->GetYaxis()->GetBinLowEdge(1), a2dHist->GetYaxis()->GetBinUpEdge(a2dHist->GetNbinsY()));
  //-------------------------------------------------
  vector<int> tColumnsToSkip(0);

  double tCounts = 0.;
  for(int i=1; i<=a2dHist->GetNbinsX(); i++)
  {
    tCounts = 0.;
    for(int j=1; j<=a2dHist->GetNbinsY(); j++)
    {
      tCounts += a2dHist->GetBinContent(i,j);
    }
    if(tCounts==0.) tColumnsToSkip.push_back(i);
  }

  //-------------------------------------------------

  bool bSkipX=false;

  int tXbinTracker = 0;

  for(int i=1; i<=a2dHist->GetNbinsX(); i++)
  {
    bSkipX=false;
    for(int a=0; a<(int)tColumnsToSkip.size(); a++)
    {
      if(i==tColumnsToSkip[a]) bSkipX=true;
    }
    if(!bSkipX)
    {
      tXbinTracker++;
      for(int j=1; j<=a2dHist->GetNbinsY(); j++)
      {
        tCondensed->SetBinContent(tXbinTracker, j, a2dHist->GetBinContent(i,j));
        tCondensed->GetXaxis()->SetBinLabel(tXbinTracker, a2dHist->GetXaxis()->GetBinLabel(i));
      }
    }
  }

  tCondensed->GetXaxis()->SetRange(1,tXbinTracker);
  tCondensed->GetXaxis()->SetLabelSize(0.02);

  tCondensed->GetYaxis()->SetLabelSize(0.02);

  tCondensed->LabelsOption("v", "X");

  return tCondensed;
}



//________________________________________________________________________________________________________________
void DrawCondensed2dRadiiVsPid(ParticlePDGType aType, TPad* aPad, TH2D* a2dHist, bool aSetLogZ, bool aSave, TString aSaveName)
{
  aPad->cd();
  aPad->SetRightMargin(0.15);
  aPad->SetTopMargin(0.075);
  aPad->SetLogz(aSetLogZ);
//  gStyle->SetOptStat(0);

  TString tReturnName = TString("Radii Vs PID: ") + TString(GetPDGRootName(aType));
  SetParentPidBinLabels(a2dHist->GetXaxis(), aType);
  TH2D* tCondensed = BuildCondensed2dRadiiVsBeta(a2dHist, tReturnName);

  tCondensed->Draw("colz");

  if(aSave)
  {
    TString tSaveName = aSaveName;
    if(aSetLogZ) tSaveName += TString("_LogZ");
    tSaveName += TString(".pdf");

    aPad->SaveAs(tSaveName);
  }

}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
TH1D* BuildCondensed1dParentsHistogram(TH1D* a1dHist, TString aReturnName)
{
  TH1D* tCondensed = new TH1D(aReturnName, aReturnName, 
                              a1dHist->GetNbinsX(), a1dHist->GetXaxis()->GetBinLowEdge(1), a1dHist->GetXaxis()->GetBinUpEdge(a1dHist->GetNbinsX()));
  //-------------------------------------------------
  vector<int> tColumnsToSkip(0);

  for(int i=1; i<=a1dHist->GetNbinsX(); i++)
  {
    if(a1dHist->GetBinContent(i)==0.) tColumnsToSkip.push_back(i);
  }

  //-------------------------------------------------

  bool bSkipX=false;

  int tXbinTracker = 0;

  for(int i=1; i<=a1dHist->GetNbinsX(); i++)
  {
    bSkipX=false;
    for(int a=0; a<(int)tColumnsToSkip.size(); a++)
    {
      if(i==tColumnsToSkip[a]) bSkipX=true;
    }
    if(!bSkipX)
    {
      tXbinTracker++;
      tCondensed->SetBinContent(tXbinTracker, a1dHist->GetBinContent(i));
      tCondensed->GetXaxis()->SetBinLabel(tXbinTracker, a1dHist->GetXaxis()->GetBinLabel(i));
    }
  }

  tCondensed->GetXaxis()->SetRange(1,tXbinTracker);
  tCondensed->GetXaxis()->SetLabelSize(0.02);

  tCondensed->GetYaxis()->SetLabelSize(0.02);

  tCondensed->LabelsOption("v", "X");

  return tCondensed;
}


//________________________________________________________________________________________________________________
void DrawCondensed1dParentsHistogram(ParticlePDGType aType, TPad* aPad, TH1D* a1dHist, bool aSave, TString aSaveName)
{
  aPad->cd();
//  gStyle->SetOptStat(0);

  TString tReturnName = TString::Format("%s Parents", GetPDGRootName(aType));
  SetParentPidBinLabels(a1dHist->GetXaxis(), aType);
  TH1D* tCondensed = BuildCondensed1dParentsHistogram(a1dHist, tReturnName);

  tCondensed->Draw();

  if(aSave)
  {
    TString tSaveName = aSaveName;
    tSaveName += TString(".pdf");

    aPad->SaveAs(tSaveName);
  }

}


