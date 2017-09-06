#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

//________________________________________________________________________________________________________________
void DrawProject3dMatrixTo1d(TPad* aPad, TH3D* a3dMatrix)
{
  aPad->cd();
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
void DrawAll(TString aFileLocationSingleParticleAnalyses, ParticlePDGType aType)
{
  gStyle->SetOptStat(1111);

  TString tName3dHist = TString::Format("%s3dRadii", GetPDGRootName(aType));
  TH3D* t3dRadii = Get3dHisto(aFileLocationSingleParticleAnalyses, tName3dHist);

  TCanvas* tCan_1dRadii = new TCanvas(TString::Format("tCan_%sRadii", GetPDGRootName(aType)),
                                      TString::Format("tCan_%sRadii", GetPDGRootName(aType)));
  DrawProject3dMatrixTo1d((TPad*)tCan_1dRadii, t3dRadii);


  TCanvas* tCan_2dRadiiVsBeta = new TCanvas(TString::Format("tCan_%s2dRadiiVsBeta", GetPDGRootName(aType)),
                                            TString::Format("tCan_%s2dRadiiVsBeta", GetPDGRootName(aType)));
  Draw2dRadiiVsBeta((TPad*)tCan_2dRadiiVsBeta, t3dRadii);


  TCanvas* tCan_2dRadiiVsPid = new TCanvas(TString::Format("tCan_%s2dRadiiVsPid", GetPDGRootName(aType)),
                                           TString::Format("tCan_%s2dRadiiVsPid", GetPDGRootName(aType)));
  Draw2dRadiiVsPid((TPad*)tCan_2dRadiiVsPid, t3dRadii);

  TCanvas* tCan_Condensed2dRadiiVsPid = new TCanvas(TString::Format("tCan_%sCondensed2dRadiiVsPid", GetPDGRootName(aType)),
                                                    TString::Format("tCan_%sCondensed2dRadiiVsPid", GetPDGRootName(aType)));
  TH2D* tCondensedRadiiVsPid = Get2dRadiiVsPid(t3dRadii);
  DrawCondensed2dRadiiVsPid(kPDGProt, (TPad*)tCan_Condensed2dRadiiVsPid, tCondensedRadiiVsPid);

  vector<int> tMostAbundantParents;
  int tNx = 2;
  int tNy = -1;
  switch(aType) {
  case kPDGLam:
    tMostAbundantParents = tMostAbundantParents_Lam;
    tNy = 4;
    break;

  case kPDGALam:
    tMostAbundantParents = tMostAbundantParents_ALam;
    tNy = 4;
    break;

  case kPDGKchP:
    tMostAbundantParents = tMostAbundantParents_KchP;
    tNy = 2;
    break;

  case kPDGKchM:
    tMostAbundantParents = tMostAbundantParents_KchM;
    tNy = 2;
    break;

  case kPDGK0:
    tMostAbundantParents = tMostAbundantParents_K0;
    tNy = 2;
    break;

  case kPDGProt:
    tMostAbundantParents = tMostAbundantParents_Prot;
    tNy = 3;
    break;

  case kPDGAntiProt:
    tMostAbundantParents = tMostAbundantParents_AProt;
    tNy = 3;
    break;

  default:
    cout << "ERROR: DrawAll: aType = " << aType << " is not appropriate" << endl << endl;
    assert(0);
  }

  TCanvas *tCan_RadiiForMultipleParents = new TCanvas("tCan_RadiiForMultipleParents", "tCan_RadiiForMultipleParents");
  DrawRadiiForMultipleParents(tCan_RadiiForMultipleParents, t3dRadii, aType, tMostAbundantParents, tNx, tNy);

  TString tNameParents = TString::Format("%sParents", GetPDGRootName(aType));
  TH1D* tParents = Get1dHisto(aFileLocationSingleParticleAnalyses, tNameParents);
  TCanvas* tCan_Parents = new TCanvas(TString::Format("tCan_%sParents", GetPDGRootName(aType)), 
                                      TString::Format("tCan_%sParents", GetPDGRootName(aType)));
  DrawCondensed1dParentsHistogram(aType, (TPad*)tCan_Parents, tParents);
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

/*
  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";
*/

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocationSingleParticleAnalyses = tDirectory + "testSingleParticleAnalysesv2.root";

  DrawAll(tFileLocationSingleParticleAnalyses, kPDGProt);

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




