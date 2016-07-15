///////////////////////////////////////////////////////////////////////////
// Cf2dHeavy:                                                            //
//                                                                       //
//  A collection of Cf2dLite*, all to be combined via a weighted average //
//  This will contain a vector<Cf2dLite*>                                //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef CF2DHEAVY_H
#define CF2DHEAVY_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"

using std::cout;
using std::endl;
using std::vector;

#include "CfHeavy.h"
class CfHeavy;

#include "Cf2dLite.h"
class Cf2dLite;

class Cf2dHeavy {

public:


  Cf2dHeavy(TString aDaughterHeavyCfsBaseName, vector<Cf2dLite*> &aCf2dLiteCollection, double aMinNorm, double aMaxNorm);
  virtual ~Cf2dHeavy();

  void CombineCfs(int aRebinFactor=1);
  void Rebin(int aRebinFactor);

  CfHeavy* GetDaughterHeavyCf(int aDaughterHeavyCf);

  //inline
  vector<CfHeavy*> GetAllDaughterHeavyCfs();


private:

  vector<Cf2dLite*> fCf2dLiteCollection;
  int fCollectionSize;
  int fNDaughterHeavyCfs;

  vector<CfHeavy*> fDaughterHeavyCfs;
  TString fDaughterHeavyCfsBaseName;
  double fMinNorm, fMaxNorm;





#ifdef __ROOT__
  ClassDef(Cf2dHeavy, 1)
#endif
};

inline vector<CfHeavy*> Cf2dHeavy::GetAllDaughterHeavyCfs() {return fDaughterHeavyCfs;}








#endif











