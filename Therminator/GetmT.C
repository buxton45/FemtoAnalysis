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
#include "ThermPairAnalysis.h"


//________________________________________________________________________________________________________________
void PrintTotalmT(AnalysisType aAnType, TString tFileLocation, double aMaxKStar=0.30001)
{
  TString tPairKStarVsmTName = TString::Format("PairKStarVsmT%s", cAnalysisBaseTags[aAnType]);
  TH2D* tPairKStarVsmT = Get2dHisto(tFileLocation, tPairKStarVsmTName); 

  tPairKStarVsmT->GetXaxis()->SetRangeUser(0., aMaxKStar);

  cout << "Total mT for " << cAnalysisBaseTags[aAnType] << " = " << tPairKStarVsmT->GetMean(2) << endl;
}


//________________________________________________________________________________________________________________
double GetmT(ParticlePDGType aType1, ParticlePDGType aType2, TH3D* aPairmT3d)
{
  int tPartIndex1 = GetParticleIndexInPidInfo(aType1) + 1;
  int tPartIndex2 = GetParticleIndexInPidInfo(aType2) + 1;

  TH1D* tTempHist = aPairmT3d->ProjectionZ(TString::Format("tTempHist%i%i", aType1, aType2),
                                           tPartIndex1, tPartIndex1,
                                           tPartIndex2, tPartIndex2, "e");

  double tReturnmT = tTempHist->GetMean();
  tTempHist->Delete();
  
  return tReturnmT;
}

//________________________________________________________________________________________________________________
void PrintmT(ParticlePDGType aType1, ParticlePDGType aType2, TH3D* aPairmT3d)
{
  cout << "mT for pair: " << GetPDGRootName(aType1) << GetPDGRootName(aType2) << " = " << GetmT(aType1, aType2, aPairmT3d) << endl;
}

//________________________________________________________________________________________________________________
void PrintAllmT(AnalysisType aAnType, TString tFileLocation)
{
  TString tPairmT3dName = TString::Format("PairmT3d%s", cAnalysisBaseTags[aAnType]);
  TH3D* tPairmT3d = Get3dHisto(tFileLocation, tPairmT3dName);

  vector<vector<ParticlePDGType> > tPDGTypes;

  if(aAnType==kLamKchP)
  {
    tPDGTypes = {{kPDGLam, kPDGKchP}, 
                 {kPDGSigma, kPDGKchP}, {kPDGXi0, kPDGKchP}, {kPDGXiC, kPDGKchP}, {kPDGOmega, kPDGKchP}, 
                 {kPDGSigStP, kPDGKchP}, {kPDGSigStM, kPDGKchP}, {kPDGSigSt0, kPDGKchP}, 
                 {kPDGLam, kPDGKSt0}, {kPDGSigma, kPDGKSt0}, {kPDGXi0, kPDGKSt0}, {kPDGXiC, kPDGKSt0}};
  }
  else if(aAnType==kALamKchM)
  {
    tPDGTypes = {{kPDGALam, kPDGKchM}, 
                 {kPDGASigma, kPDGKchM}, {kPDGAXi0, kPDGKchM}, {kPDGAXiC, kPDGKchM}, {kPDGAOmega, kPDGKchM}, 
                 {kPDGASigStM, kPDGKchM}, {kPDGASigStP, kPDGKchM}, {kPDGASigSt0, kPDGKchM}, 
                 {kPDGALam, kPDGAKSt0}, {kPDGASigma, kPDGAKSt0}, {kPDGAXi0, kPDGAKSt0}, {kPDGAXiC, kPDGAKSt0}};
  }
  //------------------------------------------------------------------------
  else if(aAnType==kLamKchM)
  {
    tPDGTypes = {{kPDGLam, kPDGKchM}, 
                 {kPDGSigma, kPDGKchM}, {kPDGXi0, kPDGKchM}, {kPDGXiC, kPDGKchM}, {kPDGOmega, kPDGKchM},  
                 {kPDGSigStP, kPDGKchM}, {kPDGSigStM, kPDGKchM}, {kPDGSigSt0, kPDGKchM}, 
                 {kPDGLam, kPDGAKSt0}, {kPDGSigma, kPDGAKSt0}, {kPDGXi0, kPDGAKSt0}, {kPDGXiC, kPDGAKSt0}};
  }
  else if(aAnType==kALamKchP)
  {
    tPDGTypes = {{kPDGALam, kPDGKchP}, 
                 {kPDGASigma, kPDGKchP}, {kPDGAXi0, kPDGKchP}, {kPDGAXiC, kPDGKchP}, {kPDGAOmega, kPDGKchP},  
                 {kPDGASigStM, kPDGKchP}, {kPDGASigStP, kPDGKchP}, {kPDGASigSt0, kPDGKchP}, 
                 {kPDGALam, kPDGKSt0}, {kPDGASigma, kPDGKSt0}, {kPDGAXi0, kPDGKSt0}, {kPDGAXiC, kPDGKSt0}};
  }
  //------------------------------------------------------------------------
  else if(aAnType==kLamK0)
  {
    tPDGTypes = {{kPDGLam, kPDGK0}, 
                 {kPDGSigma, kPDGK0}, {kPDGXi0, kPDGK0}, {kPDGXiC, kPDGK0}, {kPDGOmega, kPDGK0},  
                 {kPDGSigStP, kPDGK0}, {kPDGSigStM, kPDGK0}, {kPDGSigSt0, kPDGK0}, 
                 {kPDGLam, kPDGKSt0}, {kPDGSigma, kPDGKSt0}, {kPDGXi0, kPDGKSt0}, {kPDGXiC, kPDGKSt0}};
  }
  else if(aAnType==kALamK0)
  {
    tPDGTypes = {{kPDGALam, kPDGK0}, 
                 {kPDGASigma, kPDGK0}, {kPDGAXi0, kPDGK0}, {kPDGAXiC, kPDGK0}, {kPDGAOmega, kPDGK0},  
                 {kPDGASigStM, kPDGK0}, {kPDGASigStP, kPDGK0}, {kPDGASigSt0, kPDGK0}, 
                 {kPDGALam, kPDGKSt0}, {kPDGASigma, kPDGKSt0}, {kPDGAXi0, kPDGKSt0}, {kPDGAXiC, kPDGKSt0}};
  }

  for(unsigned int i=0; i<tPDGTypes.size(); i++) PrintmT(tPDGTypes[i][0], tPDGTypes[i][1], tPairmT3d);

  tPairmT3d->Delete();
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

  int tImpactParam = 2;

  TString tFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions.root", tImpactParam);
//  TString tFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testCorrelationFunctions.root";

//  AnalysisType tAnType = kLamKchP;
//  PrintAllmT(tAnType, tFileName);

  vector<AnalysisType> tAnTypes{kLamKchP, kALamKchM,
                                kLamKchM, kALamKchP,
                                kLamK0, kALamK0};

  for(unsigned int i=0; i<tAnTypes.size(); i++)
  {
    cout << "----------------------------------------" << endl;
    PrintTotalmT(tAnTypes[i], tFileName, 0.30001);
    PrintAllmT(tAnTypes[i], tFileName);
    cout << "----------------------------------------" << endl << endl;
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}














