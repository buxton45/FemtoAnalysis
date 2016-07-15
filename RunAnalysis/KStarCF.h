/*
 *  KStarCF.h
 *
 */

#ifndef KSTARCF_H
#define KSTARCF_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#ifndef __CINT__

#include "AliFemtoPair.h"

#include <AliFemtoBasicTrackCut.h>
#include <AliFemtoTrack.h>
#include <AliFemtoBasicEventCut.h>
#include <AliFemtoSimpleAnalysis.h>
#include <AliFemtoCutMonitorEventMult.h>
#include "AliFemtoV0TrackCut.h"
#include <AliFemtoAvgSepCorrFctn.h>
#include <AliFemtoESDTrackCut.h>
#include <AliFemtoCutMonitorParticleYPt.h>

#include <AliFemtoCorrFctn.h>

#endif

//#include <AliFemtoCutMonitorEventMult.h>


#include <AliFemtoCorrFctn.h>


//#include "AliAODInputHandler.h"
//#include "AliAnalysisManager.h"



class KStarCF : public AliFemtoCorrFctn {
public:
  KStarCF();
  KStarCF(char *title, const int& nbins, const float KStarLo, const float KStarHi, AliFemtoAnalysis *analysis = NULL);

  /** Pure Virtual **/
  virtual AliFemtoString Report();
  virtual void Finish();
  virtual TList* GetOutputList();


  /** Corr Functions **/
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

protected:
  TH1D *_numerator;
  TH1D *_denominator;

  TH1D *_minv;
  TH1D *_qinv;

  TH1D *_minv_m;
  TH1D *_qinv_m;

#ifdef __ROOT__
  ClassDef(KStarCF, 2)
#endif
};

#endif /*KSTARCF_H*/

