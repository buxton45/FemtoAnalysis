/*
 *  myAliFemtoKStarCorrFctnMC.h
 *
 *  Based off AliFemtoQinvCorrFctn and Kubera's KStarCF
 */

#ifndef MYALIFEMTOKSTARCORRFCTNMC_H
#define MYALIFEMTOKSTARCORRFCTNMC_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

#include "AliFemtoModelHiddenInfo.h"

class myAliFemtoKStarCorrFctnMC : public AliFemtoCorrFctn {
public:
  myAliFemtoKStarCorrFctnMC(const char* title, const int& nbins, const float& KStarLo, const float& KStarHi);
  myAliFemtoKStarCorrFctnMC(const myAliFemtoKStarCorrFctnMC& aCorrFctn);
  virtual ~myAliFemtoKStarCorrFctnMC();

  myAliFemtoKStarCorrFctnMC& operator=(const myAliFemtoKStarCorrFctnMC& aCorrFctn);

  double GetKStarTrue(AliFemtoPair* aPair);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  TH1D* NumeratorTrue();
  TH1D* DenominatorTrue();

  virtual TList* GetOutputList();
  void Write();

private:
  TH1D* fNumeratorTrue;	//numerator using momenta before propagation through detectors
  TH1D* fDenominatorTrue; //denominator using momenta before propagation through detectors


#ifdef __ROOT__
  ClassDef(myAliFemtoKStarCorrFctnMC, 1)
#endif
};


inline  TH1D* myAliFemtoKStarCorrFctnMC::NumeratorTrue(){return fNumeratorTrue;}
inline  TH1D* myAliFemtoKStarCorrFctnMC::DenominatorTrue(){return fDenominatorTrue;}


#endif
