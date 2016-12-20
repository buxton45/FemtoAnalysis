//
//  ConfigFemtoAnalysis.cxx
//
//
//  Returns a pointer to the created AliFemtoManager


#if !defined(__CINT__) || defined(__MAKECINT__)
#include "AliFemtoManager.h"
#include "AliFemtoEventReaderESDChain.h"
#include "AliFemtoEventReaderAODChain.h"
#include "AliFemtoSimpleAnalysis.h"
#include "AliFemtoBasicEventCut.h"
#include "AliFemtoESDTrackCut.h"
#include "AliFemtoAODTrackCut.h"
#include "AliFemtoCutMonitorParticleYPt.h"
#include "AliFemtoShareQualityTPCEntranceSepPairCut.h"
#include "AliFemtoQinvCorrFctn.h"
#include "AliFemtoShareQualityCorrFctn.h"
#include "AliFemtoTPCInnerCorrFctn.h"
#include "AliFemtoVertexMultAnalysis.h"

#include <AliFemtoBasicTrackCut.h>
#include <AliFemtoTrack.h>
#include <AliFemtoCutMonitorEventMult.h>
#include "AliFemtoV0TrackCut.h"
#include "myAliFemtoV0TrackCut.h"
#include <AliFemtoAvgSepCorrFctn.h>
#include "AliFemtoV0PairCut.h"

#include "myAliFemtoKStarCorrFctn.h"
#include "KStarCF.h"
#include "myAliFemtoVertexMultAnalysis.h"
#endif



AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  const int chargePip = 1;
  const int chargePim = -1;

  gROOT->LoadMacro("myAliFemtoV0TrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn.cxx+g");
  gROOT->LoadMacro("KStarCF.cxx+g");
  gROOT->LoadMacro("myAliFemtoVertexMultAnalysis.cxx+g");

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
  myAliFemtoVertexMultAnalysis *analy_K0Lam = new myAliFemtoVertexMultAnalysis("K0Lam",8, -8.0, 8.0, 4, 0, 100);
    //AliFemtoVertexMultAnalysis(unsigned int binsVertex=10, double minVertex=-100., double maxVertex=+100., unsigned int binsMult=10, double minMult=-1.e9, double maxMult=+1.e9);
    analy_K0Lam->SetNumEventsToMix(5);
    analy_K0Lam->SetMinSizePartCollection(1);
    analy_K0Lam->SetVerboseMode(kFALSE);

  myAliFemtoVertexMultAnalysis *analy_K0ALam = new myAliFemtoVertexMultAnalysis("K0ALam",8, -8.0, 8.0, 4, 0, 100);
    //AliFemtoVertexMultAnalysis(unsigned int binsVertex=10, double minVertex=-100., double maxVertex=+100., unsigned int binsMult=10, double minMult=-1.e9, double maxMult=+1.e9);
    analy_K0ALam->SetNumEventsToMix(5);
    analy_K0ALam->SetMinSizePartCollection(1);
    analy_K0ALam->SetVerboseMode(kFALSE);

  //The event selector
  AliFemtoBasicEventCut* mec = new AliFemtoBasicEventCut();
    //Accept events with the given multiplicity
    mec->SetEventMult(0,100000);
    //and z-vertex distance to the center of the TPC
    mec->SetVertZPos(-10.0,10.0);

  //--------------------------------------------------------------------Lambdas----------------------------------------------------------------//
  // V0 Track Cut (1 = lambda)
  myAliFemtoV0TrackCut* v0cut1 = new myAliFemtoV0TrackCut();
    v0cut1->SetParticleType(0);  //  0=lambda -> daughters = proton(+) and pi-
    v0cut1->SetMass(LambdaMass);
    v0cut1->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for lambda's
    v0cut1->SetRemoveMisidentified(kTRUE);
    v0cut1->SetInvMassMisidentified(KaonMass-0.003677,KaonMass+0.003677);  //m_inv criteria to remove all lambda candidates fulfilling K0short hypothesis
    v0cut1->SetMisIDHisto("MisIDLambdas",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetCalculatePurity(kTRUE);
    v0cut1->SetLooseInvMassCut(LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetUseLooseInvMassCut(kTRUE);
    v0cut1->SetPurityHisto("LambdaPurity",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    v0cut1->SetPt(0.2, 5.0);
    v0cut1->SetOnFlyStatus(kFALSE);
    v0cut1->SetMaxDcaV0(0.5); //  DCA of V0 to primary vertex must be less than 0.5 cm
    v0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    v0cut1->SetMaxV0DecayLength(60.0);
    //-----
    v0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    v0cut1->SetPtPosDaughter(0.5,4.0); //0.5 for protons
    v0cut1->SetPtNegDaughter(0.16,4.0); //0.16 for pions
    v0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    v0cut1->SetNdofDaughters(4.0); //4.0
    v0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut1->SetMaxDcaV0Daughters(0.4); //DCA of v0 daughters at decay vertex
    v0cut1->SetMinDaughtersToPrimVertex(0.1,0.3);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    v0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_Lam_Pass"),new AliFemtoCutMonitorV0("_Lam_Fail"));


  // V0 Track Cut (2 = anti-lambda)
  myAliFemtoV0TrackCut* v0cut2 = new myAliFemtoV0TrackCut();
    v0cut2->SetParticleType(1);  //1=anti-lambda -> daughters = anti-proton(-) and pi+
    v0cut2->SetMass(LambdaMass);
    v0cut2->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for anti-lambda's
    v0cut2->SetRemoveMisidentified(kTRUE);
    v0cut2->SetInvMassMisidentified(KaonMass-0.003677,KaonMass+0.003677);  //m_inv criteria to remove all anti-lambda candidates fulfilling K0short hypothesis
    v0cut2->SetMisIDHisto("MisIDAntiLambdas",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetCalculatePurity(kTRUE);
    v0cut2->SetLooseInvMassCut(LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetUseLooseInvMassCut(kTRUE);
    v0cut2->SetPurityHisto("AntiLambdaPurity",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetEta(0.8);
    v0cut2->SetPt(0.5,5.0);
    v0cut2->SetOnFlyStatus(kFALSE); //kTRUE
    v0cut2->SetMaxDcaV0(0.5);
    v0cut2->SetMaxCosPointingAngle(0.9993); //0.99 - Jai
    v0cut2->SetMaxV0DecayLength(60.0);
    //-----
    v0cut2->SetEtaDaughters(0.8);
    v0cut2->SetPtPosDaughter(0.16,4.0); //0.16 for pions
    v0cut2->SetPtNegDaughter(0.3,4.0);  //0.3 for anti-protons
    v0cut2->SetTPCnclsDaughters(80);
    v0cut2->SetNdofDaughters(4.0); //4.0
    v0cut2->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut2->SetMaxDcaV0Daughters(0.4); //1.5 Jai, 0.6
    v0cut2->SetMinDaughtersToPrimVertex(0.3,0.1);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    v0cut2->AddCutMonitor(new AliFemtoCutMonitorV0("_ALam_Pass"),new AliFemtoCutMonitorV0("_ALam_Fail"));


  //--------------------------------------------------------------------Kaons----------------------------------------------------------------//
  myAliFemtoV0TrackCut* k0cut1 = new myAliFemtoV0TrackCut();
    k0cut1->SetParticleType(2);  //  2=K0Short -> daughters = pi+ and pi-
    k0cut1->SetMass(KaonMass);
    k0cut1->SetInvariantMassK0Short(KaonMass-0.013677,KaonMass+0.020323);  //m_inv criteria for K0shorts
    k0cut1->SetRemoveMisidentified(kTRUE);
    k0cut1->SetInvMassMisidentified(LambdaMass-0.005683,LambdaMass+0.005683);  //m_inv criteria to remove all K0short candidates fulfilling (anti-)lambda hypothesis
    k0cut1->SetMisIDHisto("MisIDK0Short1",100,KaonMass-0.070,KaonMass+0.070);
    k0cut1->SetCalculatePurity(kTRUE);
    k0cut1->SetLooseInvMassCut(KaonMass-0.070,KaonMass+0.070);
    k0cut1->SetUseLooseInvMassCut(kTRUE);
    k0cut1->SetPurityHisto("K0ShortPurity1",100,KaonMass-0.070,KaonMass+0.070);
    k0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    k0cut1->SetPt(0.2, 5.0);
    k0cut1->SetOnFlyStatus(kFALSE);
    k0cut1->SetMaxDcaV0(0.3); //  DCA of V0 to primary vertex
    k0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    k0cut1->SetMaxV0DecayLength(30.0);
    //-----
    k0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    k0cut1->SetPtPosDaughter(0.15,4.0); //
    k0cut1->SetPtNegDaughter(0.15,4.0); //
    k0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    k0cut1->SetNdofDaughters(4.0); //4.0
    k0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    k0cut1->SetMaxDcaV0Daughters(0.3); //DCA of v0 daughters at decay vertex
    k0cut1->SetMinDaughtersToPrimVertex(0.3,0.3);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    k0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_K01_Pass"),new AliFemtoCutMonitorV0("_K01_Fail"));

  myAliFemtoV0TrackCut* k0cut2 = new myAliFemtoV0TrackCut();
    k0cut2->SetParticleType(2);  //  2=K0Short -> daughters = pi+ and pi-
    k0cut2->SetMass(KaonMass);
    k0cut2->SetInvariantMassK0Short(KaonMass-0.013677,KaonMass+0.020323);  //m_inv criteria for K0shorts
    k0cut2->SetRemoveMisidentified(kTRUE);
    k0cut2->SetInvMassMisidentified(LambdaMass-0.005683,LambdaMass+0.005683);  //m_inv criteria to remove all K0short candidates fulfilling (anti-)lambda hypothesis
    k0cut2->SetMisIDHisto("MisIDK0Short2",100,KaonMass-0.070,KaonMass+0.070);
    k0cut2->SetCalculatePurity(kTRUE);
    k0cut2->SetLooseInvMassCut(KaonMass-0.070,KaonMass+0.070);
    k0cut2->SetUseLooseInvMassCut(kTRUE);
    k0cut2->SetPurityHisto("K0ShortPurity2",100,KaonMass-0.070,KaonMass+0.070);
    k0cut2->SetEta(0.8); //|eta|<0.8 for V0s
    k0cut2->SetPt(0.2, 5.0);
    k0cut2->SetOnFlyStatus(kFALSE);
    k0cut2->SetMaxDcaV0(0.3); //  DCA of V0 to primary vertex
    k0cut2->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    k0cut2->SetMaxV0DecayLength(30.0);
    //-----
    k0cut2->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    k0cut2->SetPtPosDaughter(0.15,4.0); //
    k0cut2->SetPtNegDaughter(0.15,4.0); //
    k0cut2->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    k0cut2->SetNdofDaughters(4.0); //4.0
    k0cut2->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    k0cut2->SetMaxDcaV0Daughters(0.3); //DCA of v0 daughters at decay vertex
    k0cut2->SetMinDaughtersToPrimVertex(0.3,0.3);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    k0cut2->AddCutMonitor(new AliFemtoCutMonitorV0("_K02_Pass"),new AliFemtoCutMonitorV0("_K02_Fail"));


  // Pair selector
  //	For two tracks (ex. pi+ pi-) use, for example, AliFemtoShareQualityTPCEntranceSepPairCut or AliFemtoShareQualityPairCut
  //	For one track and one v0 (ex. pi+ lambda) use AliFemtoV0TrackPairCut
  //	For two v0s (ex. K0Short lambda) use AliFemtoV0PairCut

  AliFemtoV0PairCut *v0pc1 = new AliFemtoV0PairCut();  //K0Short-lambda
//  v0pc1->SetV0Max(0.25);
//  v0pc1->SetShareFractionMax(0.05)  //how do I implement this in AliFemtoV0PairCut?
  v0pc1->SetRemoveSameLabel(kTRUE);
  v0pc1->SetTPCExitSepMinimum(-1.0);  //Default is 0, but for some reason distExitPos(Neg) always end up as 0?

  AliFemtoDummyPairCut *dumbpaircut = new AliFemtoDummyPairCut();


  // Add the cuts to the analyses
  analy_K0Lam->SetEventCut(mec);
  analy_K0Lam->SetFirstParticleCut(k0cut1);
  analy_K0Lam->SetSecondParticleCut(v0cut1);
  analy_K0Lam->SetPairCut(v0pc1);
//  analy_K0Lam->SetPairCut(dumbpaircut);

  analy_K0ALam->SetEventCut(mec);
  analy_K0ALam->SetFirstParticleCut(k0cut2);
  analy_K0ALam->SetSecondParticleCut(v0cut2);
  analy_K0ALam->SetPairCut(v0pc1);
//  analy_K0ALam->SetPairCut(dumbpaircut);


  // Setup correlation functions
  myAliFemtoKStarCorrFctn *K0LamKStarcf = new myAliFemtoKStarCorrFctn("K0LamKStarcf",75,0.0,0.4);
  myAliFemtoKStarCorrFctn *K0ALamKStarcf = new myAliFemtoKStarCorrFctn("K0ALamKStarcf",75,0.0,0.4);

  // add the correlation functions to the analyses
  analy_K0Lam->AddCorrFctn(K0LamKStarcf);

  analy_K0ALam->AddCorrFctn(K0ALamKStarcf);

  // Add the analyses to the manager
  mgr->AddAnalysis(analy_K0Lam);
  mgr->AddAnalysis(analy_K0ALam);

  return mgr;
}


