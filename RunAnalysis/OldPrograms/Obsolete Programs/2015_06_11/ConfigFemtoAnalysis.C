//
//  ConfigFemtoAnalysis.cxx
//
//
//  Returns a pointer to the created AliFemtoManager


#if !defined(__CINT__) || defined(__MAKECINT__)
#include "AliFemtoManager.h"
#include "AliFemtoEventReaderESDChain.h"
#include "AliFemtoEventReaderAODChain.h"
#include "AliFemtoShareQualityTPCEntranceSepPairCut.h"
#include "AliFemtoQinvCorrFctn.h"
#include "AliFemtoShareQualityCorrFctn.h"
#include "AliFemtoTPCInnerCorrFctn.h"

#include <AliFemtoTrack.h>
#include "AliFemtoV0TrackCut.h"
#include <AliFemtoAvgSepCorrFctn.h>

#include "KStarCF.h"


#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoV0PairCut.h"
#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoV0TrackCut.h"

#include "myAnalysisConstructor.h"
#endif

AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  gROOT->LoadMacro("myAliFemtoV0TrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn.cxx+g");
  gROOT->LoadMacro("KStarCF.cxx+g");
  gROOT->LoadMacro("myAnalysisConstructor.cxx+g");

  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kTRUE);

  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
    //Point to the data source - the reader
    mgr->SetEventReader(rdr);


  //Setup the analyses
  myAnalysisConstructor *analy_LamK0 = new myAnalysisConstructor("LamK0",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut1 = analy_LamK0->CreateEventCut();
    myAliFemtoV0TrackCut *LamCut1 = analy_LamK0->CreateLambdaCut();
    myAliFemtoV0TrackCut *K0Cut1 = analy_LamK0->CreateK0ShortCut();
    AliFemtoV0PairCut *PairCut1 = analy_LamK0->CreateV0PairCut();
    myAliFemtoKStarCorrFctn *KStarCF1 = analy_LamK0->CreateKStarCorrFctn("LamK0KStarCF1",75,0.,0.4);
    analy_LamK0->SetAnalysis(EvCut1,LamCut1,K0Cut1,PairCut1,KStarCF1);

  myAnalysisConstructor *analy_ALamK0 = new myAnalysisConstructor("ALamK0",8, -8.0, 8.0, 4, 0, 100);
    AliFemtoBasicEventCut *EvCut2 = analy_ALamK0->CreateEventCut();
    myAliFemtoV0TrackCut *ALamCut2 = analy_ALamK0->CreateAntiLambdaCut();
    myAliFemtoV0TrackCut *K0Cut2 = analy_ALamK0->CreateK0ShortCut();
    AliFemtoV0PairCut *PairCut2 = analy_ALamK0->CreateV0PairCut();
    myAliFemtoKStarCorrFctn *KStarCF2 = analy_ALamK0->CreateKStarCorrFctn("ALamK0KStarCF2",75,0.,0.4);
    analy_ALamK0->SetAnalysis(EvCut2,ALamCut2,K0Cut2,PairCut2,KStarCF2);

  // Add the analyses to the manager
  mgr->AddAnalysis(analy_LamK0);
  mgr->AddAnalysis(analy_ALamK0);

  return mgr;
}


