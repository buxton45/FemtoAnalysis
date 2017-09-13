#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

//________________________________________________________________________________________________________________
void DrawProject3dMatrixTo1d(TPad* aPad, TH3D* a3dMatrix)
{
  aPad->cd();
  aPad->SetLogy(true);
  gStyle->SetOptStat(1111);
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  TString tName = TString(t3dMatrix->GetName()) + TString("_pz");
  TH1D* t1dHisto = t3dMatrix->ProjectionZ(tName, 1, t3dMatrix->GetNbinsX(), 1, t3dMatrix->GetNbinsY());

  t1dHisto->GetXaxis()->SetTitle("Radius (fm)");
  t1dHisto->GetYaxis()->SetTitle("Counts");

  t1dHisto->DrawCopy();

cout << "t1dHisto->GetMean() = " << t1dHisto->GetMean() << endl;
}


//________________________________________________________________________________________________________________
void Draw2dRadiiVsBeta(TPad* aPad, TH3D* a3dMatrix, bool aSetLogZ=true)
{
  aPad->cd();
  aPad->SetLogz(aSetLogZ);
  gStyle->SetOptStat(1111);
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  t3dMatrix->GetYaxis()->SetRange(1, t3dMatrix->GetNbinsY());
  t3dMatrix->GetZaxis()->SetRange(1, t3dMatrix->GetNbinsZ());

  TH2D* t2dHisto = (TH2D*)t3dMatrix->Project3D("zy");

  t2dHisto->GetXaxis()->SetTitle("#Beta");
  t2dHisto->GetYaxis()->SetTitle("Radius");

  t2dHisto->DrawCopy("colz");
}

//________________________________________________________________________________________________________________
TH2D* Get2dRadiiVsPid(TH3D* a3dMatrix)
{
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  t3dMatrix->GetYaxis()->SetRange(1, t3dMatrix->GetNbinsY());
  t3dMatrix->GetZaxis()->SetRange(1, t3dMatrix->GetNbinsZ());

  TH2D* t2dHisto = (TH2D*)t3dMatrix->Project3D("zx");

  t2dHisto->GetXaxis()->SetTitle("PID");
  t2dHisto->GetYaxis()->SetTitle("Radius");

  return t2dHisto;
}

//________________________________________________________________________________________________________________
void Draw2dRadiiVsPid(TPad* aPad, TH3D* a3dMatrix, bool aSetLogZ=true)
{
  aPad->cd();
  aPad->SetLogz(aSetLogZ);
  gStyle->SetOptStat(1111);
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  TH2D* t2dHisto = Get2dRadiiVsPid(t3dMatrix);

  t2dHisto->DrawCopy("colz");
}

//________________________________________________________________________________________________________________
TH2D* GetProject3dMatrixTo2dRadiiVsPidPrimaryOnly(TH3D* a3dMatrix, ParticlePDGType aType, double aMaxDecayLength=-1., bool aGetCondensed=true)
{
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();
  TH2D* t2dRadiiVsPid = Get2dRadiiVsPid(t3dMatrix);
  SetParentPidBinLabels(t2dRadiiVsPid->GetXaxis(), aType);

  TString tNamePart = TString::Format("%s2dRadiiVsPidPrimOnly_%0.3f", GetPDGRootName(aType), aMaxDecayLength);
  TH2D* tPartiallyCondensed = BuildCondensed2dRadiiVsBeta(t2dRadiiVsPid, tNamePart);
  for(int i=1; i<=tPartiallyCondensed->GetNbinsX(); i++)
  {
    TString tParentName = tPartiallyCondensed->GetXaxis()->GetBinLabel(i);
    if(tParentName.IsNull() || aMaxDecayLength<0.) continue;
    if(GetParticleDecayLength(tParentName) > aMaxDecayLength &&
       GetParticlePid(tParentName) != aType)
    {
      for(int j=1; j<=tPartiallyCondensed->GetNbinsY(); j++)
      {
        tPartiallyCondensed->SetBinContent(i,j,0.);
      }
    }
    
  }

  if(aGetCondensed)
  {
    TString tName = TString::Format("%sCondensed2dRadiiVsPidPrimOnly_%0.3f", GetPDGRootName(aType), aMaxDecayLength);
    TH2D* tCondensed = BuildCondensed2dRadiiVsBeta(tPartiallyCondensed, tName);

    return tCondensed;
  }
  else return tPartiallyCondensed;
}

//________________________________________________________________________________________________________________
void DrawProject3dMatrixTo2dRadiiVsPidPrimaryOnly(TPad* aPad, TH3D* a3dMatrix, ParticlePDGType aType, double aMaxDecayLength=-1., bool aDrawCondensed=false, bool aSetLogZ=true)
{
  aPad->cd();
  aPad->SetLogz(aSetLogZ);
  gStyle->SetOptStat(101);
  gStyle->SetOptTitle(0);

  aPad->SetTopMargin(0.05);
  aPad->SetBottomMargin(0.12);

  TH2D* t2dRadiiVsPid = GetProject3dMatrixTo2dRadiiVsPidPrimaryOnly(a3dMatrix, aType, aMaxDecayLength, aDrawCondensed);

  t2dRadiiVsPid->GetXaxis()->SetTitle("Parent");
  t2dRadiiVsPid->GetYaxis()->SetTitle("Radius (fm)");

  t2dRadiiVsPid->GetXaxis()->SetTitleOffset(1.7);

  t2dRadiiVsPid->DrawCopy("colz");
}


//________________________________________________________________________________________________________________
void DrawProject3dMatrixTo1dPrimaryOnly(TPad* aPad, TH3D* a3dMatrix, ParticlePDGType aType, double aMaxDecayLength=-1.)
{
  aPad->cd();
  aPad->SetLogy(true);
  gStyle->SetOptStat(1111);

  TH2D* tCondensed = GetProject3dMatrixTo2dRadiiVsPidPrimaryOnly(a3dMatrix, aType, aMaxDecayLength);

  TString tName2 = TString::Format("%sRadiiPrimOnly_%0.3f", GetPDGRootName(aType), aMaxDecayLength);
  TH1D* tRadii = tCondensed->ProjectionY(tName2.Data(), 1, tCondensed->GetNbinsX());
    tRadii->SetName(tName2);
    tRadii->SetTitle(tName2);

  tRadii->GetXaxis()->SetTitle("Radius (fm)");
  tRadii->GetYaxis()->SetTitle("Counts");

  tRadii->DrawCopy();

}

//________________________________________________________________________________________________________________
void DrawMultipleProject3dMatrixTo1dPrimaryOnly(TPad* aPad, TH3D* a3dMatrix, ParticlePDGType aType, td1dVec &aMaxDecayLengthVec)
{
  aPad->cd();
  aPad->SetLogy(true);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TLegend* tLeg = new TLegend(0.50, 0.50, 0.85, 0.85);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  TH2D* tCondensed;
  TH1D* tRadii;
  for(unsigned int i=0; i<aMaxDecayLengthVec.size(); i++)
  {
    tCondensed = GetProject3dMatrixTo2dRadiiVsPidPrimaryOnly(a3dMatrix, aType, aMaxDecayLengthVec[i]);

    TString tName2 = TString::Format("%sRadiiPrimOnly_%0.3f", GetPDGRootName(aType), aMaxDecayLengthVec[i]);
    tRadii = tCondensed->ProjectionY(tName2.Data(), 1, tCondensed->GetNbinsX());
      tRadii->SetName(tName2);
      tRadii->SetTitle(tName2);
      tRadii->SetMarkerStyle(20);
      tRadii->SetMarkerColor(i+1);
      tRadii->SetLineColor(i+1);
      tRadii->SetLineStyle(1);
      tRadii->SetLineWidth(2);

    tRadii->GetXaxis()->SetTitle("Radius (fm)");
    tRadii->GetYaxis()->SetTitle("Counts");

    if(i==0) tRadii->Draw();
    else tRadii->Draw("same");

    if(aMaxDecayLengthVec[i] < 0.) tLeg->AddEntry(tRadii, TString::Format("No #bar{#tau_{PDG}} cut (#bar{R} = %0.2f)", tRadii->GetMean()), "l");
    else tLeg->AddEntry(tRadii, TString::Format("#bar{#tau_{PDG}} < %0.1f (#bar{R} = %0.2f)", aMaxDecayLengthVec[i], tRadii->GetMean()), "l");
  }
  tLeg->Draw();

}



//________________________________________________________________________________________________________________
TH1D* Get1dRadiiForParticularParent(TH3D* a3dMatrix, int aType, int aParentType)
{
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  vector<int> tFathers = GetParentsPidVector(static_cast<ParticlePDGType>(aType));
  int tBin = -1;
  for(unsigned int i=0; i<tFathers.size(); i++) if(tFathers[i] == aParentType) tBin = i+1;
  assert(tBin > -1);

  TString tReturnName = TString::Format("%s from %s Radii (fm)", GetPDGRootName(static_cast<ParticlePDGType>(aType)), GetParticleName(aParentType).Data());
  TH1D* tReturnHist = t3dMatrix->ProjectionZ(tReturnName.Data(), tBin, tBin, 1, t3dMatrix->GetNbinsY());
    tReturnHist->SetName(tReturnName);
    tReturnHist->SetTitle(tReturnName);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
void DrawRadiiForParticularParent(TPad* aPad, TH3D* a3dMatrix, int aType, int aParentType)
{
  gStyle->SetOptStat(1111);
  TH3D* t3dMatrix = (TH3D*)a3dMatrix->Clone();

  TH1D* tHistToDraw = Get1dRadiiForParticularParent(t3dMatrix, aType, aParentType);
  aPad->cd();
  tHistToDraw->DrawCopy();
}

//________________________________________________________________________________________________________________
void DrawRadiiForMultipleParents(TCanvas* aCan, TH3D* a3dMatrix, int aType, vector<int> &aParentTypes, int aNx, int aNy)
{
  assert(aNx*aNy >= (int)aParentTypes.size());

  aCan->Divide(aNx,aNy);
  for(unsigned int i=0; i<aParentTypes.size(); i++) DrawRadiiForParticularParent((TPad*)aCan->cd(i+1), a3dMatrix, aType, aParentTypes[i]);
}


vector<int> tMostAbundantParents_Lam {3122, 3212, 3322, 3312, 3224, 3214, 3114, 3334};
vector<int> tMostAbundantParents_ALam {-3122, -3212, -3322, -3312, -3224, -3214, -3114, -3334};
vector<int> tMostAbundantParents_KchP {321, 313, 323, 333};
vector<int> tMostAbundantParents_KchM {-321, -313, -323, 333};
vector<int> tMostAbundantParents_K0 {311, 323, 313, 333};
vector<int> tMostAbundantParents_Prot {3122, 2212, 2224, 2214, 3222, 2114};
vector<int> tMostAbundantParents_AProt {-3122, -2212, -2224, -2214, -3222, -2114};
//________________________________________________________________________________________________________________
void DrawAll(TString aFileLocationSingleParticleAnalyses, ParticlePDGType aType, double aMaxDecayLength=-1, bool aSave=false)
{
  gStyle->SetOptStat(1111);

  TString tName3dHist = TString::Format("%s3dRadii", GetPDGRootName(aType));
  TH3D* t3dRadii = Get3dHisto(aFileLocationSingleParticleAnalyses, tName3dHist);

  TCanvas* tCan_1dRadii = new TCanvas(TString::Format("tCan_%sRadii", GetPDGRootName(aType)),
                                      TString::Format("tCan_%sRadii", GetPDGRootName(aType)));
  DrawProject3dMatrixTo1d((TPad*)tCan_1dRadii, t3dRadii);


  TCanvas* tCan_1dRadiiPrimOnly = new TCanvas(TString::Format("tCan_%sRadiiPrimOnly_MaxDecay%0.2f", GetPDGRootName(aType), aMaxDecayLength),
                                              TString::Format("tCan_%sRadiiPrimOnly_MaxDecay%0.2f", GetPDGRootName(aType), aMaxDecayLength));
  DrawProject3dMatrixTo1dPrimaryOnly((TPad*)tCan_1dRadiiPrimOnly, t3dRadii, aType, aMaxDecayLength);

  td1dVec tMaxDecayLengthVec{-1., 5.5, 5., 3.};
  TCanvas* tCan_Multiple1dRadiiPrimOnly = new TCanvas(TString::Format("tCan_%sRadiiPrimOnly", GetPDGRootName(aType)),
                                              TString::Format("tCan_%sRadiiPrimOnly", GetPDGRootName(aType)));
  DrawMultipleProject3dMatrixTo1dPrimaryOnly((TPad*)tCan_Multiple1dRadiiPrimOnly, t3dRadii, aType, tMaxDecayLengthVec);


  TCanvas* tCan_2dRadiiVsPidPrimOnly = new TCanvas(TString::Format("tCan_%s2dRadiiVsPidPrimOnly_MaxDecay%0.2f", GetPDGRootName(aType), aMaxDecayLength),
                                                   TString::Format("tCan_%s2dRadiiVsPidPrimOnly_MaxDecay%0.2f", GetPDGRootName(aType), aMaxDecayLength));
  DrawProject3dMatrixTo2dRadiiVsPidPrimaryOnly((TPad*)tCan_2dRadiiVsPidPrimOnly, t3dRadii, aType, aMaxDecayLength);


  TCanvas* tCan_2dRadiiVsBeta = new TCanvas(TString::Format("tCan_%s2dRadiiVsBeta", GetPDGRootName(aType)),
                                            TString::Format("tCan_%s2dRadiiVsBeta", GetPDGRootName(aType)));
  Draw2dRadiiVsBeta((TPad*)tCan_2dRadiiVsBeta, t3dRadii);

/*
  TCanvas* tCan_2dRadiiVsPid = new TCanvas(TString::Format("tCan_%s2dRadiiVsPid", GetPDGRootName(aType)),
                                           TString::Format("tCan_%s2dRadiiVsPid", GetPDGRootName(aType)));
  Draw2dRadiiVsPid((TPad*)tCan_2dRadiiVsPid, t3dRadii);
*/

  TCanvas* tCan_Condensed2dRadiiVsPid = new TCanvas(TString::Format("tCan_%sCondensed2dRadiiVsPid", GetPDGRootName(aType)),
                                                    TString::Format("tCan_%sCondensed2dRadiiVsPid", GetPDGRootName(aType)));
  TH2D* tCondensedRadiiVsPid = Get2dRadiiVsPid(t3dRadii);
  DrawCondensed2dRadiiVsPid(aType, (TPad*)tCan_Condensed2dRadiiVsPid, tCondensedRadiiVsPid);

  vector<int> tMostAbundantParents;
  int tNx = 2;
  int tNy = -1;
  TString tSubDirectory;
  switch(aType) {
  case kPDGLam:
    tMostAbundantParents = tMostAbundantParents_Lam;
    tNy = 4;
    tSubDirectory = "Lam/";
    break;

  case kPDGALam:
    tMostAbundantParents = tMostAbundantParents_ALam;
    tNy = 4;
    tSubDirectory = "ALam/";
    break;

  case kPDGKchP:
    tMostAbundantParents = tMostAbundantParents_KchP;
    tNy = 2;
    tSubDirectory = "KchP/";
    break;

  case kPDGKchM:
    tMostAbundantParents = tMostAbundantParents_KchM;
    tNy = 2;
    tSubDirectory = "KchM/";
    break;

  case kPDGK0:
    tMostAbundantParents = tMostAbundantParents_K0;
    tNy = 2;
    tSubDirectory = "K0/";
    break;

  case kPDGProt:
    tMostAbundantParents = tMostAbundantParents_Prot;
    tNy = 3;
    tSubDirectory = "Prot/";
    break;

  case kPDGAntiProt:
    tMostAbundantParents = tMostAbundantParents_AProt;
    tNy = 3;
    tSubDirectory = "AProt/";
    break;

  default:
    cout << "ERROR: DrawAll: aType = " << aType << " is not appropriate" << endl << endl;
    assert(0);
  }

  TCanvas* tCan_RadiiForMultipleParents = new TCanvas(TString::Format("tCan_%sRadiiForMultipleParents", GetPDGRootName(aType)), 
                                                      TString::Format("tCan_%sRadiiForMultipleParents", GetPDGRootName(aType)));
  DrawRadiiForMultipleParents(tCan_RadiiForMultipleParents, t3dRadii, aType, tMostAbundantParents, tNx, tNy);

  TString tNameParents = TString::Format("%sParents", GetPDGRootName(aType));
  TH1D* tParents = Get1dHisto(aFileLocationSingleParticleAnalyses, tNameParents);
  TCanvas* tCan_Parents = new TCanvas(TString::Format("tCan_%sParents", GetPDGRootName(aType)), 
                                      TString::Format("tCan_%sParents", GetPDGRootName(aType)));
  DrawCondensed1dParentsHistogram(aType, (TPad*)tCan_Parents, tParents);

  //---------------------------------------
  if(aSave)
  {
    TString tSaveDirectory = "/home/jesse/Analysis/Presentations/AliFemto/20170913/Figures/";
    TString tFileType = ".eps";
//    TString tFileType = ".pdf";

    tCan_1dRadii->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_1dRadii->GetName()) + tFileType);
    tCan_1dRadiiPrimOnly->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_1dRadiiPrimOnly->GetName()) + tFileType);
    tCan_Multiple1dRadiiPrimOnly->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_Multiple1dRadiiPrimOnly->GetName()) + tFileType);
    tCan_2dRadiiVsPidPrimOnly->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_2dRadiiVsPidPrimOnly->GetName()) + tFileType);
    tCan_2dRadiiVsBeta->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_2dRadiiVsBeta->GetName()) + tFileType);
    tCan_Condensed2dRadiiVsPid->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_Condensed2dRadiiVsPid->GetName()) + tFileType);
    tCan_RadiiForMultipleParents->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_RadiiForMultipleParents->GetName()) + tFileType);
    tCan_Parents->SaveAs(tSaveDirectory + tSubDirectory + TString(tCan_Parents->GetName()) + tFileType);
  }
}
//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------


  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationSingleParticleAnalyses = tDirectory + "SingleParticleAnalysesv2.root";

/*
  TString tDirectory = "~/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocationSingleParticleAnalyses = tDirectory + "testSingleParticleAnalysesv2.root";
*/
  ParticlePDGType tType = kPDGLam;
  bool bSave = false;

//  double tMaxDecayLength = -1.;
//  double tMaxDecayLength = 3.0;
  double tMaxDecayLength = 5.0;
//  double tMaxDecayLength = 5.5;

  DrawAll(tFileLocationSingleParticleAnalyses, tType, tMaxDecayLength, bSave);

/*
  vector<double> tMaxDecayLengthVec {-1., 3.0, 5.0, 5.5};
  for(unsigned int i=0; i<tMaxDecayLengthVec.size(); i++) DrawAll(tFileLocationSingleParticleAnalyses, tType, tMaxDecayLengthVec[i], bSave);
*/

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




