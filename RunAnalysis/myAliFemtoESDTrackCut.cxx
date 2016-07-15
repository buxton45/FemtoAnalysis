////////////////////////////////////////////////////////////////////////////
//                                                                        //
// myAliFemtoESDTrackCut:                                                 //
//                                                                        //
////////////////////////////////////////////////////////////////////////////


#include "myAliFemtoESDTrackCut.h"

#ifdef __ROOT__ 
ClassImp(myAliFemtoESDTrackCut)
#endif




//________________________________________________________________________________________________________________
myAliFemtoESDTrackCut::myAliFemtoESDTrackCut() :
  AliFemtoESDTrackCut(),
  fPionRejection(false)
{

}

//________________________________________________________________________________________________________________
myAliFemtoESDTrackCut::myAliFemtoESDTrackCut(const myAliFemtoESDTrackCut& aCut):
  AliFemtoESDTrackCut(aCut),
  fPionRejection(aCut.fPionRejection)
{

}


//________________________________________________________________________________________________________________
myAliFemtoESDTrackCut& myAliFemtoESDTrackCut::operator=(const myAliFemtoESDTrackCut& aCut)
{
  if(this == &aCut) {return *this;}

  AliFemtoESDTrackCut::operator=(aCut);

  fPionRejection = aCut.fPionRejection;
}


//________________________________________________________________________________________________________________
myAliFemtoESDTrackCut::~myAliFemtoESDTrackCut()
{
  cout << "myAliFemtoESDTrackCut object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
bool myAliFemtoESDTrackCut::IsKaonNSigma(float mom, float nsigmaTPCK, float nsigmaTOFK)
{
  if (fNsigmaTPCTOF) {
    if (mom > 0.5) {
      //        if (TMath::Hypot( nsigmaTOFP, nsigmaTPCP )/TMath::Sqrt(2) < 3.0)
      if (TMath::Hypot( nsigmaTOFK, nsigmaTPCK ) < fNsigma)
	return true;
    }
    else {
      if (TMath::Abs(nsigmaTPCK) < fNsigma)
	return true;
    }
  }

  else
  {
    if(mom < 0.5)
    {
      if(TMath::Abs(nsigmaTPCK) < 2.0) {return true;}
    }
    
    else if( (mom > 0.5) && (mom < 1.5) )
    {
      if( (nsigmaTOFK < -999) || (TMath::Abs(nsigmaTPCK) >= 3) ) {return false;}
      else
      {
        if( (mom < 0.8) && (TMath::Abs(nsigmaTOFK) < 2.0) ) {return true;}
        else if( (mom < 1.0) && (TMath::Abs(nsigmaTOFK) < 1.5) ) {return true;}
        else if( (mom < 1.5) && (TMath::Abs(nsigmaTOFK) < 1.0) ) {return true;}
      }
    }
  }
  return false;

}

//________________________________________________________________________________________________________________
bool myAliFemtoESDTrackCut::IsElectron(float nsigmaTPCE)
{
  if(TMath::Abs(nsigmaTPCE) < 3) return true;
  else return false;
}

//________________________________________________________________________________________________________________
bool myAliFemtoESDTrackCut::Pass(const AliFemtoTrack* track)
{
  //cout<<"AliFemtoESDTrackCut::Pass"<<endl;

  // test the particle and return
  // true if it meets all the criteria
  // false if it doesn't meet at least one of the criteria
  float tMost[5];

  //cout<<"AliFemtoESD  cut"<<endl;
  //cout<<fPidProbPion[0]<<" < pi ="<<track->PidProbPion()<<" <"<<fPidProbPion[1]<<endl;
  if (fStatus!=0)
    {
      //cout<<" status "<<track->Label()<<" "<<track->Flags()<<" "<<track->TPCnclsF()<<" "<<track->ITSncls()<<endl;
      if ((track->Flags()&fStatus)!=fStatus)
	{
	  //	  cout<<track->Flags()<<" "<<fStatus<<" no go through status"<<endl;
	  return false;
	}

    }
  if (fRemoveKinks) {
    if ((track->KinkIndex(0)) || (track->KinkIndex(1)) || (track->KinkIndex(2)))
      return false;
  }
  if (fRemoveITSFake) {
    if (track->ITSncls() < 0)
      return false;
  }
  if (fminTPCclsF>track->TPCnclsF())
    {
      //cout<<" No go because TPC Number of ClsF"<<fminTPCclsF<< " "<<track->TPCnclsF()<<endl;
      return false;
    }
  if (fminTPCncls>track->TPCncls())
    {
      //cout<<" No go because TPC Number of ClsF"<<fminTPCclsF<< " "<<track->TPCnclsF()<<endl;
      return false;
    }
  if (fminITScls>track->ITSncls())
    {
      //cout<<" No go because ITS Number of Cls"<<fminITScls<< " "<<track->ITSncls()<<endl;
      return false;
    }

  if (fMaxImpactXY < TMath::Abs(track->ImpactD()))
    return false;

  if (fMinImpactXY > TMath::Abs(track->ImpactD()))
    return false;

  if (fMaxImpactZ < TMath::Abs(track->ImpactZ()))
    return false;

  if (fMaxSigmaToVertex < track->SigmaToVertex()) {
    return false;
  }

  if (track->ITSncls() > 0)
    if ((track->ITSchi2()/track->ITSncls()) > fMaxITSchiNdof) {
      return false;
    }

  if (track->TPCncls() > 0)
    if ((track->TPCchi2()/track->TPCncls()) > fMaxTPCchiNdof) {
      return false;
    }
  //ITS cluster requirenments
  for (Int_t i = 0; i < 3; i++)
    if(!CheckITSClusterRequirement(fCutClusterRequirementITS[i], track->HasPointOnITSLayer(i*2), track->HasPointOnITSLayer(i*2+1)))
      return false;

  if (fLabel)
    {
      //cout<<"labels"<<endl;
      if(track->Label()<0)
	{
	  fNTracksFailed++;
	  //   cout<<"No Go Through the cut"<<endl;
	  //  cout<<fLabel<<" Label="<<track->Label()<<endl;
	  return false;
	}
    }
  if (fCharge!=0)
    {
      //cout<<"AliFemtoESD  cut ch "<<endl;
      //cout<<fCharge<<" Charge="<<track->Charge()<<endl;
      if (track->Charge()!= fCharge)
	{
	  fNTracksFailed++;
	  //  cout<<"No Go Through the cut"<<endl;
	  // cout<<fCharge<<" Charge="<<track->Charge()<<endl;
	  return false;
	}
    }




  Bool_t tTPCPidIn = (track->Flags()&AliFemtoTrack::kTPCpid)>0;
  Bool_t tITSPidIn = (track->Flags()&AliFemtoTrack::kITSpid)>0;
  Bool_t tTOFPidIn = (track->Flags()&AliFemtoTrack::kTOFpid)>0;

  if(fMinPforTOFpid > 0 && track->P().Mag() > fMinPforTOFpid &&
     track->P().Mag() < fMaxPforTOFpid && !tTOFPidIn)
    {
      fNTracksFailed++;
      return false;
    }

  if(fMinPforTPCpid > 0 && track->P().Mag() > fMinPforTPCpid &&
     track->P().Mag() < fMaxPforTPCpid && !tTPCPidIn)
    {
      fNTracksFailed++;
      return false;
    }

  if(fMinPforITSpid > 0 && track->P().Mag() > fMinPforITSpid &&
     track->P().Mag() < fMaxPforITSpid && !tITSPidIn)
    {
      fNTracksFailed++;
      return false;
    }


  float tEnergy = ::sqrt(track->P().Mag2()+fMass*fMass);
  float tRapidity = 0;
  if(tEnergy-track->P().z()!=0 && (tEnergy+track->P().z())/(tEnergy-track->P().z())>0)
    tRapidity = 0.5*::log((tEnergy+track->P().z())/(tEnergy-track->P().z()));
  float tPt = ::sqrt((track->P().x())*(track->P().x())+(track->P().y())*(track->P().y()));
  float tEta = track->P().PseudoRapidity();

  if (fMaxImpactXYPtOff < 999.0) {
    if ((fMaxImpactXYPtOff + fMaxImpactXYPtNrm*TMath::Power(tPt, fMaxImpactXYPtPow)) < TMath::Abs(track->ImpactD())) {
      fNTracksFailed++;
      return false;
    }
  }

  if ((tRapidity<fRapidity[0])||(tRapidity>fRapidity[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fRapidity[0]<<" < Rapidity ="<<tRapidity<<" <"<<fRapidity[1]<<endl;
      return false;
    }
  if ((tEta<fEta[0])||(tEta>fEta[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fEta[0]<<" < Eta ="<<tEta<<" <"<<fEta[1]<<endl;
      return false;
    }
  if ((tPt<fPt[0])||(tPt>fPt[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPt[0]<<" < Pt ="<<Pt<<" <"<<fPt[1]<<endl;
      return false;
    }




  //   cout << "Track has pids: "
  //        << track->PidProbElectron() << " "
  //        << track->PidProbMuon() << " "
  //        << track->PidProbPion() << " "
  //        << track->PidProbKaon() << " "
  //        << track->PidProbProton() << " "
  //        << track->PidProbElectron()+track->PidProbMuon()+track->PidProbPion()+track->PidProbKaon()+track->PidProbProton() << endl;


  if ((track->PidProbElectron()<fPidProbElectron[0])||(track->PidProbElectron()>fPidProbElectron[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPidProbElectron[0]<<" < e ="<<track->PidProbElectron()<<" <"<<fPidProbElectron[1]<<endl;
      return false;
    }
  if ((track->PidProbPion()<fPidProbPion[0])||(track->PidProbPion()>fPidProbPion[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPidProbPion[0]<<" < pi ="<<track->PidProbPion()<<" <"<<fPidProbPion[1]<<endl;
      return false;
    }
  if ((track->PidProbKaon()<fPidProbKaon[0])||(track->PidProbKaon()>fPidProbKaon[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPidProbKaon[0]<<" < k ="<<track->PidProbKaon()<<" <"<<fPidProbKaon[1]<<endl;
      return false;
    }
  if ((track->PidProbProton()<fPidProbProton[0])||(track->PidProbProton()>fPidProbProton[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPidProbProton[0]<<" < p  ="<<track->PidProbProton()<<" <"<<fPidProbProton[1]<<endl;
      return false;
    }
  if ((track->PidProbMuon()<fPidProbMuon[0])||(track->PidProbMuon()>fPidProbMuon[1]))
    {
      fNTracksFailed++;
      //cout<<"No Go Through the cut"<<endl;
      //cout<<fPidProbMuon[0]<<" <  mi="<<track->PidProbMuon()<<" <"<<fPidProbMuon[1]<<endl;
      return false;
    }

/*
  //****N Sigma Method -- electron rejection****
  if(fElectronRejection) 
    if(!IsElectron(track->NSigmaTPCE(),track->NSigmaTPCPi(),track->NSigmaTPCK(), track->NSigmaTPCP())) 
      return false;
*/

  if (fMostProbable) {

    int imost=0;
    tMost[0] = track->PidProbElectron()*PidFractionElectron(track->P().Mag());
    tMost[1] = 0.0;
    tMost[2] = track->PidProbPion()*PidFractionPion(track->P().Mag());
    tMost[3] = track->PidProbKaon()*PidFractionKaon(track->P().Mag());
    tMost[4] = track->PidProbProton()*PidFractionProton(track->P().Mag());
    float ipidmax = 0.0;

    //****N Sigma Method****
	if(fPIDMethod==0){
	  // Looking for pions
	  if (fMostProbable == 2) {
	    if (IsPionNSigma(track->P().Mag(), track->NSigmaTPCPi(), track->NSigmaTOFPi()))
	      imost = 2;

	  }
	  else if (fMostProbable == 3) {


	    if (IsKaonNSigma(track->P().Mag(), track->NSigmaTPCK(), track->NSigmaTOFK())){

	      imost = 3;
	    }
	  }
	  else if (fMostProbable == 4) { // proton nsigma-PID required contour adjusting (in LHC10h)
	    if ( IsProtonNSigma(track->P().Mag(), track->NSigmaTPCP(), track->NSigmaTOFP()) // && (TMath::Abs(track->NSigmaTPCP()) < TMath::Abs(track->NSigmaTPCPi())) && (TMath::Abs(track->NSigmaTPCP()) < TMath::Abs(track->NSigmaTPCK())) && (TMath::Abs(track->NSigmaTOFP()) < TMath::Abs(track->NSigmaTOFPi())) && (TMath::Abs(track->NSigmaTOFP()) < TMath::Abs(track->NSigmaTOFK()))
		 // && IsProtonTPCdEdx(track->P().Mag(), track->TPCsignal())
		 )
	      imost = 4;
	  }
	  else if (fMostProbable == 5) { // no-protons
	    if ( !IsProtonNSigma(track->P().Mag(), track->NSigmaTPCP(), track->NSigmaTOFP()) )
	      imost = 5;
	  }
	  else if (fMostProbable == 6) { //pions OR kaons OR protons
	    if (IsPionNSigma(track->P().Mag(), track->NSigmaTPCPi(), track->NSigmaTOFPi()))
	      imost = 6;
	    else if (IsKaonNSigma(track->P().Mag(), track->NSigmaTPCK(), track->NSigmaTOFK()))
	      imost = 6;
	    else if (IsProtonNSigma(track->P().Mag(), track->NSigmaTPCP(), track->NSigmaTOFP()) )
	      imost = 6;
	  }
	  else if (fMostProbable == 7) { // pions OR kaons OR protons OR electrons
	    if (IsPionNSigma(track->P().Mag(), track->NSigmaTPCPi(), track->NSigmaTOFPi()))
	      imost = 7;
	    else if (IsKaonNSigma(track->P().Mag(), track->NSigmaTPCK(), track->NSigmaTOFK()))
	      imost = 7;
	    else if (IsProtonNSigma(track->P().Mag(), track->NSigmaTPCP(), track->NSigmaTOFP()) )
	      imost = 7;
	    else if (TMath::Abs(track->NSigmaTPCE())<3)
	      imost = 7;

	  }
	  else if (fMostProbable == 8) { // TOF matching
	    if(track->NSigmaTOFPi() != -1000 || track->Pt()<0.5){
	      imost = 8;
	    }
	  }
	  else if (fMostProbable == 9) { // Other: no kaons, no pions, no protons
	    if (IsPionNSigma(track->P().Mag(), track->NSigmaTPCPi(), track->NSigmaTOFPi()))
	      imost = -1;
	    else if (IsKaonNSigma(track->P().Mag(), track->NSigmaTPCK(), track->NSigmaTOFK()))
	      imost = -1;
	    else if (IsProtonNSigma(track->P().Mag(), track->NSigmaTPCP(), track->NSigmaTOFP()) )
	      imost = -1;
	    else if(track->NSigmaTOFPi() != -1000 || track->Pt()<0.5){
	      imost = 9;
	    }
	  }
	  if (fMostProbable == 10) {//cut on Nsigma in pT not p
	    if (IsPionNSigma(track->Pt(), track->NSigmaTPCPi(), track->NSigmaTOFPi()))
	      imost = 10;
	  }
	  else if (fMostProbable == 11) {//cut on Nsigma in pT not p
	    if (IsKaonNSigma(track->Pt(), track->NSigmaTPCK(), track->NSigmaTOFK())){
	      imost = 11;
	    }
	  }
	  else if (fMostProbable == 12) { //cut on Nsigma in pT not p
	    if ( IsProtonNSigma(track->Pt(), track->NSigmaTPCP(), track->NSigmaTOFP()) )
	      imost = 12;
	  }


	}



    //****Contour Method****
	if(fPIDMethod==1){
	  for (int ip=0; ip<5; ip++)
	    if (tMost[ip] > ipidmax) { ipidmax = tMost[ip]; imost = ip; };

	  // Looking for pions
	  if (fMostProbable == 2) {
	    if (imost == 2) {
	      // Using the TPC to reject non-pions
	      if (!(IsPionTPCdEdx(track->P().Mag(), track->TPCsignal())))
		imost = 0;
	      if (0) {
		// Using the TOF to reject non-pions
		if (track->P().Mag() < 0.6) {
		  if (tTOFPidIn)
		    if (!IsPionTOFTime(track->P().Mag(), track->TOFpionTime()))
		      imost = 0;
		}
		else {
		  if (tTOFPidIn) {
		    if (!IsPionTOFTime(track->P().Mag(), track->TOFpionTime()))
		      imost = 0;
		  }
		  else {
		    imost = 0;
		  }
		}
	      }
	    }
	  }

	  // Looking for kaons
	  else if (fMostProbable == 3) {
	    //       if (imost == 3) {
	    // Using the TPC to reject non-kaons
	    if (track->P().Mag() < 0.6) {
	      if (!(IsKaonTPCdEdx(track->P().Mag(), track->TPCsignal())))
		imost = 0;
	      else imost = 3;
	      if (1) {
		// Using the TOF to reject non-kaons
		if (tTOFPidIn)
		  if (!IsKaonTOFTime(track->P().Mag(), track->TOFkaonTime()))
		    imost = 0;
	      }
	    }
	    else {
	      if (1) {
		if (tTOFPidIn) {
		  if (!IsKaonTOFTime(track->P().Mag(), track->TOFkaonTime()))
		    imost = 0;
		  else
		    imost = 3;
		}
		else {
		  if (!(IsKaonTPCdEdx(track->P().Mag(), track->TPCsignal())))
		    imost = 0;
		  else
		    imost = 3;
		}
	      }
	    }
	    //       }
	  }

	  // Looking for protons
	  else if (fMostProbable == 4) {
	    //       if (imost == 3) {
	    // Using the TPC to reject non-kaons
	    if (track->P().Mag() < 0.8) {
	      if (!(IsProtonTPCdEdx(track->P().Mag(), track->TPCsignal())))
		imost = 0;
	      else imost = 4;
	      if (0) {
		// Using the TOF to reject non-kaons
		if (tTOFPidIn)
		  if (!IsKaonTOFTime(track->P().Mag(), track->TOFkaonTime()))
		    imost = 0;
	      }
	    }
	    else {
	      if (0) {
		if (tTOFPidIn) {
		  if (!IsKaonTOFTime(track->P().Mag(), track->TOFkaonTime()))
		    imost = 0;
		  else
		    imost = 3;
		}
		else {
		  if (!(IsKaonTPCdEdx(track->P().Mag(), track->TPCsignal())))
		    imost = 0;
		  else
		    imost = 3;
		}
	      }
	    }
	    //       }
	  }
	}
    if (imost != fMostProbable) return false;
  }

  if(fElectronRejection)
  {
    if(IsElectron(track->NSigmaTPCE())) return false;
  }

  if(fPionRejection)
  {
    if(IsPionNSigma(track->P().Mag(), track->NSigmaTPCPi(), track->NSigmaTOFPi())) return false;
  }

  //fan
  //cout<<"****** Go Through the cut ******"<<endl;
  // cout<<fLabel<<" Label="<<track->Label()<<endl;
  // cout<<fCharge<<" Charge="<<track->Charge()<<endl;
  // cout<<fPt[0]<<" < Pt ="<<Pt<<" <"<<fPt[1]<<endl;
  //cout<<fRapidity[0]<<" < Rapidity ="<<tRapidity<<" <"<<fRapidity[1]<<endl;
  //cout<<fPidProbElectron[0]<<" <  e="<<track->PidProbElectron()<<" <"<<fPidProbElectron[1]<<endl;
  //cout<<fPidProbPion[0]<<" <  pi="<<track->PidProbPion()<<" <"<<fPidProbPion[1]<<endl;
  //cout<<fPidProbKaon[0]<<" <  k="<<track->PidProbKaon()<<" <"<<fPidProbKaon[1]<<endl;
  //cout<<fPidProbProton[0]<<" <  p="<<track->PidProbProton()<<" <"<<fPidProbProton[1]<<endl;
  //cout<<fPidProbMuon[0]<<" <  mi="<<track->PidProbMuon()<<" <"<<fPidProbMuon[1]<<endl;
  fNTracksPassed++ ;
  return true;


}
