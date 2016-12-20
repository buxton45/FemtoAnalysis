
#include "myAnalysisConstructor.h"
#include "TObjArray.h"
#include "AliESDtrack.h"
#ifdef __ROOT__
ClassImp(myAnalysisConstructor)
#endif

static const double PionMass = 0.13956995,
                    KaonMass = 0.493677,
                  ProtonMass = 0.938272013,
                  LambdaMass = 1.115683;

//____________________________
myAnalysisConstructor::myAnalysisConstructor() : 
  AliFemtoVertexMultAnalysis(), fOutputName("Analysis"), fMultHist(0)
{
  SetVerboseMode(kFALSE);
  fMultHist = new TH1F("MultHist","MultHist",30,0,3000);
}

//____________________________
myAnalysisConstructor::myAnalysisConstructor(const char* name) : 
  AliFemtoVertexMultAnalysis(), fOutputName(name), fMultHist(0)
{
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);

  char buffer[50];
  sprintf(buffer, "MultHist_%s",name);
  fMultHist = new TH1F(buffer,buffer,30,0,3000);
}

//____________________________
myAnalysisConstructor::myAnalysisConstructor(const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult) : 
  AliFemtoVertexMultAnalysis(binsVertex,minVertex,maxVertex,binsMult,minMult,maxMult), fOutputName(name), fMultHist(0)
{
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);

  char buffer[50];
  sprintf(buffer, "MultHist_%s",name);
  fMultHist = new TH1F(buffer,buffer,30,0,3000);
}

//____________________________
/*
myAnalysisConstructor::myAnalysisConstructor(const myAnalysisConstructor& a) :
  AliFemtoVertexMultAnalysis()
{

}
*/
//____________________________
/*
myAnalysisConstructor& myAnalysisConstructor::operator=(const myAnalysisConstructor& TheOriginalAnalysis)
{

}
*/
//____________________________
myAnalysisConstructor::~myAnalysisConstructor()
{

}

//____________________________
void myAnalysisConstructor::ProcessEvent(const AliFemtoEvent* hbtEvent)
{
  double multiplicity = hbtEvent->UncorrectedNumberOfPrimaries();
  fMultHist->Fill(multiplicity);
  AliFemtoVertexMultAnalysis::ProcessEvent(hbtEvent);
}

//____________________________
TList* myAnalysisConstructor::GetOutputList()
{
  TList *olist = new TList();
  TObjArray *temp = new TObjArray();
  olist->SetName(fOutputName);
  temp->SetName(fOutputName);

  TList *tOutputList = AliFemtoSimpleAnalysis::GetOutputList(); 
  myAliFemtoV0TrackCut* p1cut = dynamic_cast <myAliFemtoV0TrackCut*> (fFirstParticleCut);
  if(p1cut)
    {
      tOutputList->Add(p1cut->GetPurityHisto());
      tOutputList->Add(p1cut->GetMisIDHisto());
    }
  myAliFemtoV0TrackCut* p2cut = dynamic_cast <myAliFemtoV0TrackCut*> (fSecondParticleCut);
  if(p2cut)
    {
      tOutputList->Add(p2cut->GetPurityHisto());
      tOutputList->Add(p2cut->GetMisIDHisto());
    }

  tOutputList->Add(GetMultHist());

  /*
  myAliFemtoV0TrackCut *p1cut = (myAliFemtoV0TrackCut*) fFirstParticleCut;
  myAliFemtoV0TrackCut *p2cut = (myAliFemtoV0TrackCut*) fSecondParticleCut;
  tOutputList->Add(p1cut->GetPurityHisto());
  tOutputList->Add(p1cut->GetMisIDHisto());
  tOutputList->Add(p2cut->GetPurityHisto());
  tOutputList->Add(p2cut->GetMisIDHisto());
  */

  TListIter next(tOutputList);
  while (TObject *obj = next())
  {
    temp->Add(obj);
  }

  olist->Add(temp);    
  return olist;
}

//____________________________
AliFemtoBasicEventCut* myAnalysisConstructor::CreateBasicEventCut()
{
  AliFemtoBasicEventCut* mec = new AliFemtoBasicEventCut();
    //Accept events with the given multiplicity
    mec->SetEventMult(0,100000);
    //and z-vertex distance to the center of the TPC
    mec->SetVertZPos(-10.0,10.0);

    mec->AddCutMonitor(new AliFemtoCutMonitorEventMult("_EvPass"), new AliFemtoCutMonitorEventMult("_EvFail"));

  return mec;
}

//____________________________
TH1F *myAnalysisConstructor::GetMultHist()
{
  return fMultHist;
}

//____________________________
AliFemtoEventCutEstimators* myAnalysisConstructor::CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh)
{
  AliFemtoEventCutEstimators* EvCutEst = new AliFemtoEventCutEstimators();
    EvCutEst->SetCentEst1Range(aCentLow,aCentHigh);
    EvCutEst->SetVertZPos(-8.0,8.0);

    EvCutEst->AddCutMonitor(new AliFemtoCutMonitorEventMult("_EvPass"), new AliFemtoCutMonitorEventMult("_EvFail"));

  return EvCutEst;
}

//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateLambdaCut()
{
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

  return v0cut1;
}

//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateAntiLambdaCut()
{
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

  return v0cut2;
}

//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateK0ShortCut()
{
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

    k0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_K0_Pass"),new AliFemtoCutMonitorV0("_K0_Fail"));

  return k0cut1;
}

//____________________________
AliFemtoESDTrackCut* myAnalysisConstructor::CreateKchCut(const int aCharge)
{
  AliFemtoESDTrackCut* kaontc1 = new AliFemtoESDTrackCut();
    kaontc1->SetPidProbPion(0.0,0.1);
    kaontc1->SetPidProbMuon(0.0,0.8);
    kaontc1->SetPidProbKaon(0.2,1.001);
    kaontc1->SetPidProbProton(0.0,0.1);
    kaontc1->SetMostProbableKaon();
    kaontc1->SetCharge(aCharge);
  // so we set the correct mass
    kaontc1->SetMass(KaonMass);
  // we select low pt
    kaontc1->SetPt(0.1,2.0);
    kaontc1->SetEta(-0.8,0.8);
//    kaontc1->SetStatus(AliESDtrack::kTPCrefit|AliESDtrack::kITSrefit);  //This cuts out all particles when used in conjunction with SetFilterBit(7)
    kaontc1->SetminTPCncls(80);
    kaontc1->SetRemoveKinks(kTRUE);
    kaontc1->SetLabel(kFALSE);
    kaontc1->SetMaxITSChiNdof(3.0);
    kaontc1->SetMaxTPCChiNdof(2.0);
    kaontc1->SetMaxSigmaToVertex(3.0);
    kaontc1->SetMaxImpactXY(2.4);
    kaontc1->SetMaxImpactZ(3.2);
  //Cut monitor
  char pass[20];
  char fail[20];
  if(aCharge == 1)
  {
    sprintf(pass, "_KchP_Pass");
    sprintf(fail, "_KchP_Fail");
  }
  else
  {
    sprintf(pass, "_KchM_Pass");
    sprintf(fail, "_KchM_Fail");
  }
  AliFemtoCutMonitorParticleYPt *cutPass = new AliFemtoCutMonitorParticleYPt(pass, 0.13957);
  AliFemtoCutMonitorParticleYPt *cutFail = new AliFemtoCutMonitorParticleYPt(fail, 0.13957);
  kaontc1->AddCutMonitor(cutPass, cutFail);

  return kaontc1;
}

//____________________________
AliFemtoESDTrackCut* myAnalysisConstructor::CreatePiCut(const int aCharge)
{
  AliFemtoESDTrackCut* piontc1 = new AliFemtoESDTrackCut();
    piontc1->SetPidProbPion(0.2,1.001);
    piontc1->SetPidProbMuon(0.0,0.8);
    piontc1->SetPidProbKaon(0.0,0.1);
    piontc1->SetPidProbProton(0.0,0.1);
    piontc1->SetMostProbablePion();
    piontc1->SetCharge(aCharge);
  // so we set the correct mass
    piontc1->SetMass(PionMass);
  // we select low pt
    piontc1->SetPt(0.1,2.0);
    piontc1->SetEta(-0.8,0.8);
//    piontc1->SetStatus(AliESDtrack::kTPCrefit|AliESDtrack::kITSrefit);    //This cuts out all pions when used in conjunction with SetFilterBit(7)
    piontc1->SetminTPCncls(80);
    piontc1->SetRemoveKinks(kTRUE);
    piontc1->SetLabel(kFALSE);
    piontc1->SetMaxITSChiNdof(3.0);
    piontc1->SetMaxTPCChiNdof(2.0);
    piontc1->SetMaxSigmaToVertex(3.0);
    piontc1->SetMaxImpactXY(2.4);
    piontc1->SetMaxImpactZ(3.2);
  //Cut monitor
  char pass[20];
  char fail[20];
  if(aCharge == 1)
  {
    sprintf(pass, "_PiP_Pass");
    sprintf(fail, "_PiP_Fail");
  }
  else
  {
    sprintf(pass, "_PiM_Pass");
    sprintf(fail, "_PiM_Fail");
  }
  AliFemtoCutMonitorParticleYPt *cutPass = new AliFemtoCutMonitorParticleYPt(pass, 0.13957);
  AliFemtoCutMonitorParticleYPt *cutFail = new AliFemtoCutMonitorParticleYPt(fail, 0.13957);
  piontc1->AddCutMonitor(cutPass, cutFail);

  return piontc1;
}

//____________________________
AliFemtoV0PairCut* myAnalysisConstructor::CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg)
{
  AliFemtoV0PairCut *v0pc1 = new AliFemtoV0PairCut();  //K0Short-lambda
//  v0pc1->SetV0Max(0.25);
//  v0pc1->SetShareFractionMax(0.05)  //how do I implement this in AliFemtoV0PairCut?
  v0pc1->SetRemoveSameLabel(kTRUE);
  v0pc1->SetTPCExitSepMinimum(-1.0);  //Default is 0, but for some reason distExitPos(Neg) always end up as 0?

  v0pc1->SetMinAvgSeparation(0,aMinAvgSepPosPos);
  v0pc1->SetMinAvgSeparation(1,aMinAvgSepPosNeg);
  v0pc1->SetMinAvgSeparation(2,aMinAvgSepNegPos);
  v0pc1->SetMinAvgSeparation(3,aMinAvgSepNegNeg);

  return v0pc1;
}

//____________________________
AliFemtoV0TrackPairCut* myAnalysisConstructor::CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg)
{
  AliFemtoV0TrackPairCut *v0TrackPairCut1 = new AliFemtoV0TrackPairCut();
    v0TrackPairCut1->SetShareQualityMax(1.0);
    v0TrackPairCut1->SetShareFractionMax(1.0);
    v0TrackPairCut1->SetTPCOnly(kTRUE);
    v0TrackPairCut1->SetDataType(AliFemtoPairCut::kAOD);
    v0TrackPairCut1->SetTPCEntranceSepMinimum(0.00001);
    v0TrackPairCut1->SetTPCExitSepMinimum(-1.);
//    v0TrackPairCut1->SetKstarCut(0.04,AliFemtoV0TrackPairCut::kAntiLambda,AliFemtoV0TrackPairCut::kAntiProton); //1 - antilambda, 3 - antiproton
//    v0TrackPairCut1->SetMinAvgSeparation(0,0); //0 - track-pos, 1 - track-neg
//    v0TrackPairCut1->SetMinAvgSeparation(1,11);
    v0TrackPairCut1->SetRemoveSameLabel(kTRUE);

  v0TrackPairCut1->SetMinAvgSeparation(0,aMinAvgSepTrackPos);
  v0TrackPairCut1->SetMinAvgSeparation(1,aMinAvgSepTrackNeg);

  return v0TrackPairCut1;
}

//____________________________
myAliFemtoKStarCorrFctn* myAnalysisConstructor::CreateKStarCorrFctn(const char* name, unsigned int bins, double min, double max)
{
  myAliFemtoKStarCorrFctn *cf = new myAliFemtoKStarCorrFctn(name,bins,min,max);
  return cf;
}

//____________________________
myAliFemtoAvgSepCorrFctn* myAnalysisConstructor::CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max)
{
  myAliFemtoAvgSepCorrFctn *cf = new myAliFemtoAvgSepCorrFctn(name,bins,min,max);
  return cf;
}

//____________________________
myAliFemtoSepCorrFctns* myAnalysisConstructor::CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY)
{
  myAliFemtoSepCorrFctns *cf = new myAliFemtoSepCorrFctns(name,binsX,minX,maxX,binsY,minY,maxY);
  return cf;
}

//____________________________
void myAnalysisConstructor::SetAnalysis(AliFemtoBasicEventCut* aEventCut, myAliFemtoV0TrackCut* aPartCut1, myAliFemtoV0TrackCut* aPartCut2, AliFemtoV0PairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);
  AddCorrFctn(aCF);
}

//____________________________
void myAnalysisConstructor::SetAnalysis(AliFemtoEventCutEstimators* aEventCut, myAliFemtoV0TrackCut* aPartCut1, myAliFemtoV0TrackCut* aPartCut2, AliFemtoV0PairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);
  AddCorrFctn(aCF);
}

//____________________________
void myAnalysisConstructor::SetAnalysis(AliFemtoBasicEventCut* aEventCut, myAliFemtoV0TrackCut* aPartCut1, AliFemtoESDTrackCut* aPartCut2, AliFemtoV0TrackPairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);
  AddCorrFctn(aCF);
}

//____________________________
void myAnalysisConstructor::SetAnalysis(AliFemtoEventCutEstimators* aEventCut, myAliFemtoV0TrackCut* aPartCut1, AliFemtoESDTrackCut* aPartCut2, AliFemtoV0TrackPairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);
  AddCorrFctn(aCF);
}

