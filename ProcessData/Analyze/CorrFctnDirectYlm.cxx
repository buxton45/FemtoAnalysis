#include "CorrFctnDirectYlm.h"

#include <TMath.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <iostream>

using namespace std;

// const double factorials[11] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0 };
// const int    maxjm = 9; // We cut the decomposition after 2
// const double els[10] = { 0.0,  1.0, 1.0, 1.0,  2.0,  2.0, 2.0, 2.0, 2.0, 3.0 };
// const double ems[10] = { 0.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
// const int    elsi[10] = { 0, 1,  1, 1,  2,  2, 2, 2, 2, 3 };
// const int    emsi[10] = { 0, -1, 0, 1, -2, -1, 0, 1, 2, 3 };

CorrFctnDirectYlm::CorrFctnDirectYlm(const char *name, int maxl, int ibin=30, double vmin=0.0, double vmax=0.3):
  numsreal(0),
  numsimag(0),
  densreal(0),
  densimag(0),
  cfctreal(0),
  cfctimag(0),
  binctn(0),
  binctd(0),
  covnum(0),
  covden(0),
  covcfc(0),
  covmnum(0),
  covmden(0),
  covmcfc(0),
  els(0),
  ems(0),
  elsi(0),
  emsi(0),
  factorials(0),
  ylmbuffer(0),
  mNormRadius(0.0),
  mNormBohr(0.0),
  mNormBinMin(0),
  mNormBinMax(0)
{
  fMaxL = maxl;
  maxjm = (maxl+1)*(maxl+1);

  //  cout <<  "Size is " << sizeof(double) << " " << sizeof(complex<double>) << endl;

  // Fill in factorials table
  factorials = (double *) malloc(sizeof(double) * (4 * (maxl + 1)));
  double fac = 1;
  factorials[0] = 1;
  for (int iter=1; iter<4*(maxl+1); iter++)
    {
      fac *= iter;
      factorials[iter] = fac;
    }

  // Fill in els and ems table
  int el = 0;
  int em = 0;
  int il = 0;
  els = (double *) malloc(sizeof(double) * (maxjm));
  ems = (double *) malloc(sizeof(double) * (maxjm));
  elsi = (int *) malloc(sizeof(int) * (maxjm));
  emsi = (int *) malloc(sizeof(int) * (maxjm));
  do {
    els[il] = el;
    ems[il] = em;
    elsi[il] = (int) el;
    emsi[il] = (int) em;

#ifdef _FINISH_DEBUG_
    cout << "il el em " << il << " " << elsi[il] << " " << emsi[il] << endl;
#endif
    em++;
    il++;
    if (em > el) {
      el++;
      em = -el;
    }
  }
  while (el <= maxl);

#ifdef _FINISH_DEBUG_
  for (il=0; il<maxjm; il++)
    cout << "il el em " << il << " " << elsi[il] << " " << emsi[il] << endl;
#endif

  // Create numerator and denominator historgrams
  //  int sthp = sizeof(TH1D *);
  //  numsreal = (TH1D **) malloc(sthp * maxjm);
//   numsreal = new TH1D * [maxjm];
//   numsimag = new TH1D * [maxjm];
//   densreal = new TH1D * [maxjm];
//   densimag = new TH1D * [maxjm];
//   cfctreal = new TH1D * [maxjm];
//   cfctimag = new TH1D * [maxjm];
  numsreal = (TH1D **) malloc(sizeof(TH1D *) * maxjm);
  numsimag = (TH1D **) malloc(sizeof(TH1D *) * maxjm);
  densreal = (TH1D **) malloc(sizeof(TH1D *) * maxjm);
  densimag = (TH1D **) malloc(sizeof(TH1D *) * maxjm);
  cfctreal = (TH1D **) malloc(sizeof(TH1D *) * maxjm);
  cfctimag = (TH1D **) malloc(sizeof(TH1D *) * maxjm);

  char bufname[200];
  for (int ihist=0; ihist<maxjm; ihist++) {
    sprintf(bufname, "NumReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    numsreal[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);
    sprintf(bufname, "NumImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    numsimag[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);
    sprintf(bufname, "DenReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    densreal[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);
    sprintf(bufname, "DenImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    densimag[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);
    sprintf(bufname, "CfnReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    cfctreal[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);
    sprintf(bufname, "CfnImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    cfctimag[ihist] = new TH1D(bufname, bufname, ibin, vmin, vmax);

    numsreal[ihist]->Sumw2();
    numsimag[ihist]->Sumw2();
    densreal[ihist]->Sumw2();
    densimag[ihist]->Sumw2();
    cfctreal[ihist]->Sumw2();
    cfctimag[ihist]->Sumw2();
  }

  sprintf(bufname, "BinCountNum%s", name);
  binctn = new TH1D(bufname, bufname, ibin, vmin, vmax);

  sprintf(bufname, "BinCountDen%s", name);
  binctd = new TH1D(bufname, bufname, ibin, vmin, vmax);

  ylmbuffer = (complex<double> *) malloc(sizeof(complex<double>) * maxjm);

  // Covariance matrices
  covmnum = (double *) malloc(sizeof(double) * maxjm * maxjm * 4 * ibin);
  covmden = (double *) malloc(sizeof(double) * maxjm * maxjm * 4 * ibin);
  covmcfc = (double *) malloc(sizeof(double) * maxjm * maxjm * 4 * ibin);

  covnum = 0;
  covden = 0;
  covcfc = 0;
}


CorrFctnDirectYlm::CorrFctnDirectYlm()
{
  CorrFctnDirectYlm("CorrFctnDirectYlm",2);
}

CorrFctnDirectYlm::~CorrFctnDirectYlm()
{
  for (int ihist=0; ihist<maxjm; ihist++) {
    delete numsreal[ihist];
    delete numsimag[ihist];
    delete densreal[ihist];
    delete densimag[ihist];
    delete cfctreal[ihist];
    delete cfctimag[ihist];
  }

  delete binctn;
  delete binctd;

//   //  delete numsreal;
//   //  delete numsimag;
//   //  delete densreal;
//   //  delete densimag;
//   //  delete cfctreal;
//   //  delete cfctimag;

//   free( numsreal);
//   free( numsimag);
//   free( densreal);
//   free( densimag);
//   free( cfctreal);
//   free( cfctimag);

//   free(factorials);
//   free(els);
//   free(ems);
//   free(elsi);
//   free(emsi);
//   free(ylmbuffer);

//   if (covmnum) free(covmnum);
//   if (covmden) free(covmden);
//   if (covmcfc) free(covmcfc);

  if (covnum) delete covnum;
  if (covden) delete covden;
  if (covcfc) delete covcfc;
}

double CorrFctnDirectYlm::ClebschGordan(double aJot1, double aEm1, double aJot2, double aEm2, double aJot, double aEm)
{
  int mint, maxt;
  double cgc = 0.0;
  int titer;
  double coef;

  maxt = lrint(aJot1 + aJot2 - aJot);
  mint = 0;
  if (lrint(aJot1 - aEm1) < maxt) maxt = lrint(aJot1 - aEm1);
  if (lrint(aJot2 + aEm2) < maxt) maxt = lrint(aJot2 + aEm2);
  if (lrint(-(aJot-aJot2+aEm1)) > mint) mint = lrint(-(aJot-aJot2+aEm1));
  if (lrint(-(aJot-aJot1-aEm2)) > mint) mint = lrint(-(aJot-aJot1-aEm2));

  for (titer = mint; titer<=maxt; titer ++)
    {
      coef = TMath::Power(-1, titer);
      coef *= TMath::Sqrt((2*aJot+1)*
			  factorials[lrint(aJot1+aEm1)] *
			  factorials[lrint(aJot1-aEm1)] *
			  factorials[lrint(aJot2+aEm2)] *
			  factorials[lrint(aJot2-aEm2)] *
			  factorials[lrint(aJot+aEm)] *
			  factorials[lrint(aJot-aEm)]);
      coef /= (factorials[titer] *
	       factorials[lrint(aJot1+aJot2-aJot-titer)] *
	       factorials[lrint(aJot1-aEm1-titer)] *
	       factorials[lrint(aJot2+aEm2-titer)] *
	       factorials[lrint(aJot-aJot2+aEm1+titer)] *
	       factorials[lrint(aJot-aJot1-aEm2+titer)]);

      cgc += coef;
    }

  cgc *= DeltaJ(aJot1, aJot2, aJot);

  return cgc;
}

double CorrFctnDirectYlm::DeltaJ(double aJot1, double aJot2, double aJot)
{
  if ((aJot1+aJot2-aJot) < 0) {
    //    cout << "J1+J2-J3 < 0 !!!" << " " << aJot1 << " " << aJot2 << " " << aJot << endl;
    return 0;
  }
  if ((aJot1-aJot2+aJot) < 0) {
    //    cout << "J1-J2+J3 < 0 !!!" << " " << aJot1 << " " << aJot2 << " " << aJot << endl;
    return 0;
  }
  if ((-aJot1+aJot2+aJot) < 0) {
    //    cout << "-J1+J2+J3 < 0 !!!" << " " << aJot1 << " " << aJot2 << " " << aJot << endl;
    return 0;
  }
  if ((aJot1+aJot2+aJot+1) < 0) {
    //    cout << "J1+J2+J3+1 < 0 !!!" << " " << aJot1 << " " << aJot2 << " " << aJot << endl;
    return 0;
  }
  double res = TMath::Sqrt(1.0 *
			   factorials[lrint(aJot1+aJot2-aJot)] *
			   factorials[lrint(aJot1-aJot2+aJot)] *
			   factorials[lrint(-aJot1+aJot2+aJot)] /
			   factorials[lrint(aJot1+aJot2+aJot+1)]);

  return res;
}

double CorrFctnDirectYlm::WignerSymbol(double aJot1, double aEm1, double aJot2, double aEm2, double aJot, double aEm)
{
  if (lrint(aEm1+aEm2+aEm) != 0.0)
    return 0.0;
  double cge = ClebschGordan(aJot1, aEm1, aJot2, aEm2, aJot, -aEm);
  if (lrint(abs(aJot1 - aJot2 - aEm)) % 2)
    cge *= -1.0;
  cge /= sqrt(2*aJot + 1);

  if (cge == -0.0) cge = 0.0;

  return cge;
}


void CorrFctnDirectYlm::GetMtilde(complex<double> *aMat, double *aMTilde)
{
  // Create the Mtilde for a given q bin
  double lzero, mzero;
  double lprim, mprim;
  double lbis, mbis;

  int lzeroi, mzeroi;
  int lprimi, mprimi;
  int lbisi, mbisi;

  for (int iz=0; iz< GetMaxJM()*2; iz++)
    for (int ip=0; ip<GetMaxJM()*2; ip++)
      aMTilde[iz*GetMaxJM()*2+ip] = 0.0;

  for (int izero = 0; izero<GetMaxJM(); izero++) {
    GetElEmForIndex(izero, &lzero, &mzero);
    GetElEmForIndex(izero, &lzeroi, &mzeroi);
//     if (mzero < 0)
//       continue;
    for (int ibis = 0; ibis<GetMaxJM(); ibis++) {
      GetElEmForIndex(ibis, &lbis, &mbis);
      GetElEmForIndex(ibis, &lbisi, &mbisi);

//       if (mbis<0) continue;

      complex<double> val = complex<double>(0.0, 0.0);
      std::vector<complex<double> > mcomp(maxjm);
      for (int iprim = 0; iprim<GetMaxJM(); iprim++) {

	GetElEmForIndex(iprim, &lprim, &mprim);
	GetElEmForIndex(iprim, &lprimi, &mprimi);

// 	if (mprim < 0 ) continue;

	if (abs(mzeroi) % 2) mcomp[iprim] = complex<double>(-1.0, 0.0); // (-1)^m
	else mcomp[iprim] = complex<double>(1.0, 0.0);

	mcomp[iprim] *= sqrt((2*lzero+1)*(2*lprim+1)*(2*lbis+1));   // P1
	mcomp[iprim] *= WignerSymbol(lzero, 0, lprim, 0, lbis, 0); // W1
	mcomp[iprim] *= WignerSymbol(lzero, -mzero, lprim, mprim, lbis, mbis); // W2
	mcomp[iprim] *= aMat[iprim];
	//	if (
	val += mcomp[iprim];
      }

      aMTilde[(izero*2)*(2*GetMaxJM()) + (ibis*2)]     =  real(val);
      aMTilde[(izero*2+1)*(2*GetMaxJM()) + (ibis*2)]   =  imag(val);
      if (imag(val) != 0.0)
	aMTilde[(izero*2)*(2*GetMaxJM()) + (ibis*2+1)]   = -imag(val);
      else
	aMTilde[(izero*2)*(2*GetMaxJM()) + (ibis*2+1)]   = 0.0;
      aMTilde[(izero*2+1)*(2*GetMaxJM()) + (ibis*2+1)] =  real(val);
    }
  }
}

int  CorrFctnDirectYlm::GetMaxJM()
{ return maxjm; }

void CorrFctnDirectYlm::GetElEmForIndex(int aIndex, double *aEl, double *aEm)
{
  *aEl = els[aIndex];
  *aEm = ems[aIndex];
}

void CorrFctnDirectYlm::GetElEmForIndex(int aIndex, int *aEl, int *aEm)
{
  *aEl = elsi[aIndex];
  *aEm = emsi[aIndex];
}

int CorrFctnDirectYlm::GetBin(int qbin, int ilmzero, int zeroimag, int ilmprim, int primimag)
{
  return (qbin*GetMaxJM()*GetMaxJM()*4 +
	  (ilmprim*2 + primimag) * GetMaxJM()*2 +
	  ilmzero*2 + zeroimag);
}

void CorrFctnDirectYlm::AddRealPair(double qout, double qside, double qlong, double weight)
{
  double kv = sqrt(qout*qout + qside*qside + qlong*qlong);
  int nqbin = binctn->GetXaxis()->FindFixBin(kv) - 1;

  SpherHarmonics::YlmUpToL(elsi[GetMaxJM()-1], qout, qside, qlong, ylmbuffer);
  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    //    ylmbuffer[ilm] = SpherHarmonics::Ylm(elsi[ilm], emsi[ilm], qout, qside, qlong);

    numsreal[ilm]->Fill(kv, real(ylmbuffer[ilm])*weight);
    numsimag[ilm]->Fill(kv, -imag(ylmbuffer[ilm])*weight);

    binctn->Fill(kv, 1.0);
  }

  // Fill in the error matrix
  int tabshift = nqbin*GetMaxJM()*GetMaxJM()*4;
  if (nqbin < binctn->GetNbinsX())
    for (int ilmzero=0; ilmzero<GetMaxJM(); ilmzero++)
      for (int ilmprim=0; ilmprim<GetMaxJM(); ilmprim++) {
	covmnum[GetBin(nqbin, ilmzero, 0, ilmprim, 0)] += real(ylmbuffer[ilmzero])*real(ylmbuffer[ilmprim])*weight*weight;
	covmnum[GetBin(nqbin, ilmzero, 0, ilmprim, 1)] += real(ylmbuffer[ilmzero])*-imag(ylmbuffer[ilmprim])*weight*weight;
	covmnum[GetBin(nqbin, ilmzero, 1, ilmprim, 0)] += -imag(ylmbuffer[ilmzero])*real(ylmbuffer[ilmprim])*weight*weight;
	covmnum[GetBin(nqbin, ilmzero, 1, ilmprim, 1)] += -imag(ylmbuffer[ilmzero])*-imag(ylmbuffer[ilmprim])*weight*weight;

      }

}

void CorrFctnDirectYlm::AddMixedPair(double qout, double qside, double qlong, double weight)
{
  double kv = sqrt(qout*qout + qside*qside + qlong*qlong);

  SpherHarmonics::YlmUpToL(elsi[GetMaxJM()-1], qout, qside, qlong, ylmbuffer);
  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    //    ylmbuffer[ilm] = SpherHarmonics::Ylm(elsi[ilm], emsi[ilm], qout, qside, qlong);

    densreal[ilm]->Fill(kv, real(ylmbuffer[ilm])*weight);
    densimag[ilm]->Fill(kv, -imag(ylmbuffer[ilm])*weight);

    binctd->Fill(kv, 1.0);
  }

  // Fill in the error matrix
  int nqbin = binctn->GetXaxis()->FindFixBin(kv) - 1;
  int tabshift = nqbin*GetMaxJM()*GetMaxJM()*4;
  if (nqbin < binctn->GetNbinsX())
    for (int ilmzero=0; ilmzero<GetMaxJM(); ilmzero++)
      for (int ilmprim=0; ilmprim<GetMaxJM(); ilmprim++) {
	covmden[GetBin(nqbin, ilmzero, 0, ilmprim, 0)] += real(ylmbuffer[ilmzero])*real(ylmbuffer[ilmprim]);
	covmden[GetBin(nqbin, ilmzero, 0, ilmprim, 1)] += real(ylmbuffer[ilmzero])*-imag(ylmbuffer[ilmprim]);
	covmden[GetBin(nqbin, ilmzero, 1, ilmprim, 0)] += -imag(ylmbuffer[ilmzero])*real(ylmbuffer[ilmprim]);
	covmden[GetBin(nqbin, ilmzero, 1, ilmprim, 1)] += -imag(ylmbuffer[ilmzero])*-imag(ylmbuffer[ilmprim]);

      }
}

void CorrFctnDirectYlm::AddRealPair(double *qvec, double weight) {
  AddRealPair(qvec[0], qvec[1], qvec[2], weight);
}

void CorrFctnDirectYlm::AddMixedPair(double *qvec, double weight) {
  AddMixedPair(qvec[0], qvec[1], qvec[2], weight);
}

void CorrFctnDirectYlm::Finish()
{
  std::vector<complex<double> > tMq0(maxjm);
  std::vector<complex<double> > tTq0(maxjm);
  double tMTilde[maxjm*maxjm*4];
  std::vector<complex<double> > tCq0(maxjm);

  int recalccov = 1;
  if ((covnum) && (covnum->GetBinContent(0,0,0)> 0.0)) {
    cout << "Detected calculated covariance matrix. Do not recalculate !!!" << endl;
    recalccov = 0;
  }

  // Create covariance storage elements
  PackCovariances();

  // If requested calculate the advanced normalization factor
  // *** WARNING !!! ***
  // This is the factor that assumes that the correlation function
  // calculated is the non-identical particle correlation function
  // dominated by the asymptotic behaviour of the Coulmb interaction
  // in the normalization region
  // For any other function use normal normalization !!!
  // *** WARNING !!! ***

  double normfactor = 1.0;
  if (mNormBinMax > 0) {
    double sksum = 0.0;
    double wksum = 0.0;

    double sk, wk, ks;
    if (mNormBinMin<1) mNormBinMin=1;
    if (mNormBinMax > densreal[0]->GetNbinsX()) mNormBinMax = densreal[0]->GetNbinsX();
    for (int ib=mNormBinMin; ib<= mNormBinMax; ib++) {
      ks = densreal[0]->GetXaxis()->GetBinCenter(ib);
      sk = numsreal[0]->GetBinContent(ib)/(densreal[0]->GetBinContent(ib)*(1.0-mNormPurity/(mNormRadius*mNormBohr*ks*ks)));
      wk = numsreal[0]->GetBinContent(ib);
      sksum += sk*wk;
      wksum += wk;
    }
    normfactor *= sksum/wksum;
    normfactor /= numsreal[0]->GetEntries()/densreal[0]->GetEntries();
  }

  for (int ibin=1; ibin<=numsreal[0]->GetNbinsX(); ibin++) {
    for (int ilm=0; ilm<maxjm; ilm++) {
      //      cout << numsimag[ilm]->GetBinContent(ibin) << endl;
      if (recalccov) {
	tMq0[ilm] = complex<double>(densreal[ilm]->GetBinContent(ibin)/(densreal[0]->GetEntries()/normfactor),
				    densimag[ilm]->GetBinContent(ibin)/(densreal[0]->GetEntries()/normfactor));
	tTq0[ilm] = complex<double>(numsreal[ilm]->GetBinContent(ibin)/numsreal[0]->GetEntries(),
				    numsimag[ilm]->GetBinContent(ibin)/numsreal[0]->GetEntries());
      }
      else {
	tMq0[ilm] = complex<double>(densreal[ilm]->GetBinContent(ibin)/normfactor,
				    densimag[ilm]->GetBinContent(ibin)/normfactor);
	tTq0[ilm] = complex<double>(numsreal[ilm]->GetBinContent(ibin),
				    numsimag[ilm]->GetBinContent(ibin));
      }
      //      cout << imag(tTq0[ilm]) << endl;
    }

    // Calculate the proper error matrix for T
    // from the temporary covariance matrices
    //    int tabshift = (ibin-1)*GetMaxJM()*GetMaxJM()*4;
    if (recalccov) {
      for (int ilmzero=0; ilmzero<GetMaxJM(); ilmzero++)
	for (int ilmprim=0; ilmprim<GetMaxJM(); ilmprim++) {
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)])) {
	    cout << "NaN !!!! RR " << ilmzero << " " << ilmprim << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)])) {
	    cout << "NaN !!!! RI " << ilmzero << " " << ilmprim << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)])) {
	    cout << "NaN !!!! IR " << ilmzero << " " << ilmprim << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)])) {
	    cout << "NaN !!!! II " << ilmzero << " " << ilmprim << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)] /= numsreal[0]->GetEntries();
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)] /= numsreal[0]->GetEntries();
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)] /= numsreal[0]->GetEntries();
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)] /= numsreal[0]->GetEntries();

	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)])) {
	    cout << "NaN !!!! RR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)])) {
	    cout << "NaN !!!! RI" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)])) {
	    cout << "NaN !!!! IR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)])) {
	    cout << "NaN !!!! II" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)] -= real(tTq0[ilmzero])*real(tTq0[ilmprim]);
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)] -= real(tTq0[ilmzero])*imag(tTq0[ilmprim]);
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)] -= imag(tTq0[ilmzero])*real(tTq0[ilmprim]);
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)] -= imag(tTq0[ilmzero])*imag(tTq0[ilmprim]);

	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)])) {
	    cout << "NaN !!!! RR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)])) {
	    cout << "NaN !!!! RI" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)])) {
	    cout << "NaN !!!! IR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)])) {
	    cout << "NaN !!!! II" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)] /= (numsreal[0]->GetEntries() - 1);
	  covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)] /= (numsreal[0]->GetEntries() - 1);
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)] /= (numsreal[0]->GetEntries() - 1);
	  covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)] /= (numsreal[0]->GetEntries() - 1);

	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 0)])) {
	    cout << "NaN !!!! RR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 0, ilmprim, 1)])) {
	    cout << "NaN !!!! RI" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 0)])) {
	    cout << "NaN !!!! IR" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	  if (isnan(covmnum[GetBin(ibin-1, ilmzero, 1, ilmprim, 1)])) {
	    cout << "NaN !!!! II" << endl;
	    cout << numsreal[0]->GetEntries() << " "
		 << real(tTq0[ilmzero]) << " "
		 << real(tTq0[ilmprim]) << " "
		 << imag(tTq0[ilmzero]) << " "
		 << imag(tTq0[ilmprim]) << " " << endl;
	  }
	}
    }

    GetMtilde(&tMq0.front(), tMTilde);

    // Perform the solution for the correlation function itself and the errors

    if (0) {
      // This is the "wrong" way
      // do not use it
      // Fill in the M matrix
      double mM[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mM[iter] = tMTilde[iter];

      gsl_matrix_view matM = gsl_matrix_view_array(mM, GetMaxJM()*2, GetMaxJM()*2);

      // Prepare halper matrices
      double mU[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mU[iter] = tMTilde[iter];

      gsl_matrix_view matU = gsl_matrix_view_array(mU, GetMaxJM()*2, GetMaxJM()*2);

      double mV[maxjm*maxjm*4];
      gsl_matrix_view matV = gsl_matrix_view_array(mV, GetMaxJM()*2, GetMaxJM()*2);

      double vS[maxjm*2];
      gsl_vector_view vecS = gsl_vector_view_array(vS, GetMaxJM()*2);

      double vW[maxjm*2];
      gsl_vector_view vecW = gsl_vector_view_array(vW, GetMaxJM()*2);

      // Decomposing the M matrix
      gsl_linalg_SV_decomp(&matU.matrix, &matV.matrix, &vecS.vector, &vecW.vector);

      double vB[maxjm*2];
      for (int iter=0; iter<GetMaxJM(); iter++) {
	vB[iter*2] = real(tTq0[iter]);
	vB[iter*2+1] = imag(tTq0[iter]);
      }

      // Prepare inputs for solving the problem
      gsl_vector_view vecB = gsl_vector_view_array(vB, GetMaxJM()*2);

      double vX[maxjm];
      gsl_vector_view vecX = gsl_vector_view_array(vX, GetMaxJM()*2);

      // Solving the problem
      gsl_linalg_SV_solve(&matU.matrix, &matV.matrix, &vecS.vector, &vecB.vector, &vecX.vector);

      for (int ilm=0; ilm<maxjm; ilm++) {
	cfctreal[ilm]->SetBinContent(ibin, vX[ilm*2]);
	cfctimag[ilm]->SetBinContent(ibin, vX[ilm*2+1]);
      }

      // Perform the solutions for the error matrix as well

      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	// We must always add errors
	mU[iter] = (tMTilde[iter]);

      // Decomposing the M matrix
      gsl_linalg_SV_decomp(&matU.matrix, &matV.matrix, &vecS.vector, &vecW.vector);

      for (int iterlmz=0; iterlmz<GetMaxJM()*2; iterlmz++) {
	int ilmz = iterlmz/2;
	int imgz = iterlmz%2;
	for (int ilm=0; ilm<GetMaxJM(); ilm++) {
	  vB[ilm*2]   = sqrt(fabs(covmnum[GetBin(ibin-1, ilmz, imgz, ilm, 0)]));
	  vB[ilm*2+1] = sqrt(fabs(covmnum[GetBin(ibin-1, ilmz, imgz, ilm, 1)]));
	}
	gsl_linalg_SV_solve(&matU.matrix, &matV.matrix, &vecS.vector, &vecB.vector, &vecX.vector);

	for (int ilm=0; ilm<GetMaxJM(); ilm++) {
	  if (!(isnan(vX[ilm*2])))
	    covmcfc[GetBin(ibin-1, ilmz, imgz, ilm, 0)] = vX[ilm*2];
	  else
	    covmcfc[GetBin(ibin-1, ilmz, imgz, ilm, 0)] = 1e-12;
	  if (!(isnan(vX[ilm*2+1])))
	    covmcfc[GetBin(ibin-1, ilmz, imgz, ilm, 1)] = vX[ilm*2+1];
	  else
	    covmcfc[GetBin(ibin-1, ilmz, imgz, ilm, 1)] = 1e-12;
	}

      }

      for (int ilm=0; ilm<maxjm; ilm++) {
	cfctreal[ilm]->SetBinError(ibin, (fabs(covmcfc[GetBin(ibin-1, ilm, 0, ilm, 0)])));
	cfctimag[ilm]->SetBinError(ibin, (fabs(covmcfc[GetBin(ibin-1, ilm, 1, ilm, 1)])));
      }
    }

//     cout << "=============================" << endl;
//     cout << "C calculation for bin " << (ibin-1) << endl;
//     cout << endl;
//     cout << "Input: " << endl;
//     cout << "T vector " << endl;
//     for (int ilm=0; ilm<GetMaxJM(); ilm++)
//       cout << real(tTq0[ilm]) << " " << imag(tTq0[ilm]) << "   ";
//     cout << endl << "M vector " << endl;
//     for (int ilm=0; ilm<GetMaxJM(); ilm++)
//       cout << real(tMq0[ilm]) << " " << imag(tMq0[ilm]) << "   ";
//     cout << endl;
    if (0) {
      // Here is the "right" way from recent Dave's note

      // First we must calculate the Delta^2 C matrix of errors

      // First we must have the Delta^2 T matrix

      double mDeltaT[maxjm*maxjm*4];
      for (int ilmzero=0; ilmzero<GetMaxJM()*2; ilmzero++)
	for (int ilmprim=0; ilmprim<GetMaxJM()*2; ilmprim++)
	  mDeltaT[(ilmzero*maxjm*2) + ilmprim] = covmnum[GetBin(ibin-1,ilmzero/2, ilmzero%2, ilmprim/2, ilmprim%2)];

      cout << "Delta T matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mDeltaT[ilmz*GetMaxJM()*2 + ilmp];
	}
	cout << endl;
      }

      cout << "Delta T matrix packed " << endl;
      double mDeltaTPacked[maxjm*maxjm*4];
      int msize = PackYlmMatrix(mDeltaT, mDeltaTPacked);
      cout << "Packed size is " << msize << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mDeltaTPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      gsl_matrix_view matDeltaT = gsl_matrix_view_array(mDeltaTPacked, msize, msize);

      // Prepare halper matrices
      double mU[maxjm*maxjm*4];
      for (int iter=0; iter<msize*msize; iter++)
	mU[iter] = mDeltaTPacked[iter];

      gsl_matrix_view matU = gsl_matrix_view_array(mU, msize, msize);

      double mM[maxjm*maxjm*4];
      double mMPacked[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mM[iter] = tMTilde[iter];
      PackYlmMatrix(mM, mMPacked);

      gsl_matrix_view matM = gsl_matrix_view_array(mMPacked, msize, msize);

      cout << "Mtilde matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
 	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mM[ilmz*GetMaxJM()*2 + ilmp];
 	}
 	cout << endl;
       }

      cout << "Mtilde matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mMPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // DeltaT^-1 * Mtilde = U

      //      gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matU.matrix, &matM.matrix);

      // do this via solving

      // First decompose the DeltaT matrix
      double mQT[maxjm*maxjm*4];
      gsl_matrix_view matQT = gsl_matrix_view_array(mQT, msize, msize);

      double mHT[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mHT[iter] = mDeltaTPacked[iter];
      gsl_matrix_view matHT = gsl_matrix_view_array(mHT, msize, msize);

      double vST[maxjm*2];
      gsl_vector_view vecST = gsl_vector_view_array(vST, msize);

      double vWT[maxjm*2];
      gsl_vector_view vecWT = gsl_vector_view_array(vWT, msize);

      // Decomposing the M matrix
      gsl_linalg_SV_decomp(&matHT.matrix, &matQT.matrix, &vecST.vector, &vecWT.vector);

      double vXT[maxjm*2];
      gsl_vector_view vecXT = gsl_vector_view_array(vXT, msize);

      double vCT[maxjm*2];
      gsl_vector_view vecCT = gsl_vector_view_array(vCT, msize);

      for (int itert=0; itert<msize; itert++) {
	for (int iterm=0; iterm<msize; iterm++)
	  vCT[iterm] = mMPacked[itert*msize + iterm];

	// Solving the problem
	gsl_linalg_SV_solve(&matHT.matrix, &matQT.matrix, &vecST.vector, &vecCT.vector, &vecXT.vector);

	for (int iterm=0; iterm<msize; iterm++)
	  mMPacked[itert*msize + iterm] = vXT[iterm];
      }


      // Mtilde * U = V

      double mX[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mX[iter] = tMTilde[iter];

      double mXPacked[maxjm*maxjm*4];
      PackYlmMatrix(mX, mXPacked);
      gsl_matrix_view matX = gsl_matrix_view_array(mXPacked, msize, msize);

      double mV[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mV[iter] = 0;

      gsl_matrix_view matV = gsl_matrix_view_array(mV, msize, msize);

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &matX.matrix, &matM.matrix, 0.0, &matV.matrix);

      // DeltaT^-1 * T = X

      double vB[maxjm*2];
      for (int iter=0; iter<GetMaxJM(); iter++) {
	vB[iter*2] = real(tTq0[iter]);
	vB[iter*2+1] = imag(tTq0[iter]);
      }

      double vBPacked[maxjm*2];
      PackYlmVector(vB, vBPacked);


      // Prepare inputs for solving the problem
      gsl_vector_view vecB = gsl_vector_view_array(vBPacked, msize);

      cout << "B vector " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	cout.precision(3);
	cout.width(10);
	cout << vB[ilmz];
      }
      cout << endl;

      cout << "B vector packed " << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	cout << vBPacked[ilmp];
      }
      cout << endl;

      gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, &matU.matrix, &vecB.vector);

      cout << "B vector packed solution" << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	cout << vBPacked[ilmp];
      }
      cout << endl;

      // V^-1 MtildeT = Y
      //      gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matV.matrix, &matX.matrix);

      // MtildeT * X = Y

      double vY[maxjm*2];
      for (int iter=0; iter<GetMaxJM()*2; iter++) {
	vY[iter] = 0.0;
      }

      // Prepare inputs for solving the problem
      gsl_vector_view vecY = gsl_vector_view_array(vY, msize);

      gsl_blas_dgemv (CblasTrans, 1.0, &matX.matrix, &vecB.vector, 0.0, &vecY.vector);

      cout << "Y vector packed solution" << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	cout << vY[ilmp];
      }
      cout << endl;

      double vC[maxjm*2];
      for (int iter=0; iter<GetMaxJM()*2; iter++) {
 	vC[iter] = vY[iter];
      }

      // Prepare inputs for solving the problem
      gsl_vector_view vecC = gsl_vector_view_array(vC, msize);
      //      gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, &matV.matrix, &vecC.vector);
      // Solve

      double mQ[maxjm*maxjm*4];
      gsl_matrix_view matQ = gsl_matrix_view_array(mQ, msize, msize);

      double mH[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mH[iter] = mV[iter];
      gsl_matrix_view matH = gsl_matrix_view_array(mH, msize, msize);

      double vS[maxjm*2];
      gsl_vector_view vecS = gsl_vector_view_array(vS, msize);

      double vW[maxjm*2];
      gsl_vector_view vecW = gsl_vector_view_array(vW, msize);

      double vX[maxjm*2];
      gsl_vector_view vecX = gsl_vector_view_array(vX, msize);

      // Decomposing the M matrix
      gsl_linalg_SV_decomp(&matH.matrix, &matQ.matrix, &vecS.vector, &vecW.vector);

      // Solving the problem
      gsl_linalg_SV_solve(&matH.matrix, &matQ.matrix, &vecS.vector, &vecC.vector, &vecX.vector);

      cout << "C vector packed solution" << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	//	cout << vX[ilmp];
	cout << vX[ilmp];
      }
      cout << endl;

      int mpack=0;
      int el, em;
      for (int ilm=0; ilm<maxjm; ilm++) {
	// 	cfctreal[ilm]->SetBinContent(ibin, vC[mpack++]);
 	cfctreal[ilm]->SetBinContent(ibin, vX[mpack++]);
	GetElEmForIndex(ilm, &el, &em);
	if(em==0)
	  cfctimag[ilm]->SetBinContent(ibin, 0);
	else
	  //	  cfctimag[ilm]->SetBinContent(ibin, vC[mpack++]);
	  cfctimag[ilm]->SetBinContent(ibin, vX[mpack++]);
      }

      // invert the V matrix to get C errors
      double mS[maxjm*maxjm*4];

      for (int iterz=0; iterz<msize; iterz++)
	for (int iterp=0; iterp<msize; iterp++)
	  if (iterp == iterz)
	    mS[iterz*msize + iterp] = 1.0;
	  else
	    mS[iterz*msize + iterp] = 0.0;

      gsl_matrix_view matS = gsl_matrix_view_array(mS, msize, msize);

      // Invert V

      gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matV.matrix, &matS.matrix);

      mpack = 0;
      for (int ilm=0; ilm<maxjm; ilm++) {
 	cfctreal[ilm]->SetBinError(ibin, sqrt(mS[mpack*msize+mpack]));
	mpack++;
	GetElEmForIndex(ilm, &el, &em);
	if(em==0)
	  cfctimag[ilm]->SetBinError(ibin, 0);
	else {
	  cfctimag[ilm]->SetBinError(ibin, sqrt(mS[mpack*msize+mpack]));
	  mpack++;
	}
      }



//       *** END of new method ***

//       double vS[maxjm*2];
//       gsl_vector_view vecS = gsl_vector_view_array(vS, GetMaxJM()*2);


//       double mV[maxjm*maxjm*4];
//       gsl_matrix_view matV = gsl_matrix_view_array(mV, GetMaxJM()*2, GetMaxJM()*2);

//       double vS[maxjm*2];
//       gsl_vector_view vecS = gsl_vector_view_array(vS, GetMaxJM()*2);

//       double vW[maxjm*2];
//       gsl_vector_view vecW = gsl_vector_view_array(vW, GetMaxJM()*2);
    }
    if (0) {

      // Rewrite the new way to use the solving wherever there is inversion
      double mDeltaT[maxjm*maxjm*4];
      for (int ilmzero=0; ilmzero<GetMaxJM()*2; ilmzero++)
	for (int ilmprim=0; ilmprim<GetMaxJM()*2; ilmprim++)
	  mDeltaT[(ilmzero*maxjm*2) + ilmprim] = fabs(covmnum[GetBin(ibin-1,ilmzero/2, ilmzero%2, ilmprim/2, ilmprim%2)]);

      cout << "Delta T matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mDeltaT[ilmz*GetMaxJM()*2 + ilmp];
	}
	cout << endl;
      }

      double mDeltaTPacked[maxjm*maxjm*4];
      int msize = PackYlmMatrix(mDeltaT, mDeltaTPacked);

      cout << "Delta T matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mDeltaTPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // (1) Solve (DeltaT)^1 Mtilde = Q

      // Prepare halper matrices

      double mM[maxjm*maxjm*4];
      double mMPacked[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mM[iter] = tMTilde[iter];
      PackYlmMatrix(mM, mMPacked);

      gsl_matrix_view matM = gsl_matrix_view_array(mMPacked, msize, msize);

      cout << "Mtilde matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
 	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mM[ilmz*GetMaxJM()*2 + ilmp];
 	}
 	cout << endl;
       }

      cout << "Mtilde matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mMPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // Inverting matrix DeltaT.

      double mU[maxjm*maxjm*4];
      InvertYlmIndependentMatrix(mDeltaT, mU);

      double mDTInvertedPacked[maxjm*maxjm*4];
      PackYlmMatrix(mU, mDTInvertedPacked);

      gsl_matrix_view matDTI = gsl_matrix_view_array(mDTInvertedPacked, msize, msize);

      cout << "Delta T matrix inverted packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
 	for (int ilmp=0; ilmp<msize; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mDTInvertedPacked[ilmz*msize + ilmp];
 	}
 	cout << endl;
      }

      // (2) Multiply DeltaT^1 M = Q
      double mQ[maxjm*maxjm*4];
      for (int iter=0; iter<msize*msize; iter++)
	mQ[iter] = 0.0;
      gsl_matrix_view matQ = gsl_matrix_view_array(mQ, msize, msize);

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &matDTI.matrix, &matM.matrix, 0.0, &matQ.matrix);

      double mTest[maxjm*maxjm*4];
      gsl_matrix_view matTest = gsl_matrix_view_array(mTest, msize, msize);

      double mF[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mF[iter] = mDeltaTPacked[iter];
      gsl_matrix_view matF = gsl_matrix_view_array(mF, msize, msize);

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &matF.matrix, &matQ.matrix, 0.0, &matTest.matrix);

      cout << "Test matrix packed - compare to Mtilde" << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mTest[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // (2) Multiply Mtilde^T Q = P

      double mP[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mP[iter] = 0;

      gsl_matrix_view matP = gsl_matrix_view_array(mP, msize, msize);

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &matM.matrix, &matQ.matrix, 0.0, &matP.matrix);

      cout << "P matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mP[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // (3) Solve P^-1 Mtilde^T = R
      double mPUnpacked[maxjm*maxjm*4];
      UnPackYlmMatrix(mP, mPUnpacked, msize);

      cout << "P matrix unpacked " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mPUnpacked[ilmz*GetMaxJM()*2 + ilmp];
	}
	cout << endl;
      }

      // Invert the P matrix

      double mPInverted[maxjm*maxjm*4];
      InvertYlmIndependentMatrix(mPUnpacked, mPInverted);

      double mPInvertedPacked[maxjm*maxjm*4];
      PackYlmMatrix(mPInverted, mPInvertedPacked);

      gsl_matrix_view matPI = gsl_matrix_view_array(mPInvertedPacked, msize, msize);

//       cout << "P matrix inverted packed " << endl;
//       for (int ilmz=0; ilmz<msize; ilmz++) {
//  	for (int ilmp=0; ilmp<msize; ilmp++) {
//  	  cout.precision(3);
//  	  cout.width(10);
//  	  cout << mPInvertedPacked[ilmz*msize + ilmp];
//  	}
//  	cout << endl;
//       }


//       //      gsl_matrix_view matR = gsl_matrix_view_array(mR, msize, msize);

//       double mG[maxjm*maxjm*4];
//       for (int iter=0; iter<maxjm*maxjm*4; iter++)
// 	mG[iter] = mP[iter];
//       gsl_matrix_view matG = gsl_matrix_view_array(mG, msize, msize);

//       // Decomposing the M matrix
//       gsl_linalg_SV_decomp(&matG.matrix, &matS.matrix, &vecST.vector, &vecWT.vector);

//       for (int itert=0; itert<msize; itert++) {
// 	for (int iterm=0; iterm<msize; iterm++)
// 	  vCT[iterm] = mMPacked[iterm*msize + itert];
// 	  // Transvere !!!      ^^^^^         ^^^^^

// 	// Solving the problem
// 	gsl_linalg_SV_solve(&matG.matrix, &matS.matrix, &vecST.vector, &vecCT.vector, &vecXT.vector);

// 	for (int iterm=0; iterm<msize; iterm++)
// 	  mR[itert*msize + iterm] = vXT[iterm];
//       }

      double mR[maxjm*maxjm*4];
      for (int ir=0; ir<maxjm*maxjm*4; ir++)
	mR[ir] = 0.0;
      gsl_matrix_view matR = gsl_matrix_view_array(mR, msize, msize);

      // (2) Multiply P^-1 M (Trans) = R

      cout << "Matrix M Packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mMPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &matPI.matrix, &matM.matrix, 1.0, &matR.matrix);

      cout << "R matrix packed " << endl;

      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mR[ilmz*msize + ilmp];
	}
	cout << endl;
      }

      // (4) Solve DeltaT^-1 T = L
      double vL[maxjm*2];
      gsl_vector_view vecL = gsl_vector_view_array(vL, msize);


//       // Decomposing the M matrix
//       gsl_linalg_SV_decomp(&matF.matrix, &matS.matrix, &vecST.vector, &vecWT.vector);

      double vB[maxjm*2];
      for (int iter=0; iter<GetMaxJM(); iter++) {
	vB[iter*2] = real(tTq0[iter]);
	vB[iter*2+1] = imag(tTq0[iter]);
      }

      double vBPacked[maxjm*2];
      PackYlmVector(vB, vBPacked);

      gsl_vector_view vecB = gsl_vector_view_array(vBPacked, msize);

//       // Solving the problem
//       gsl_linalg_SV_solve(&matF.matrix, &matS.matrix, &vecST.vector, &vecB.vector, &vecL.vector);

      cout << "L vector packed " << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
 	cout.precision(3);
 	cout.width(10);
 	cout << vL[ilmp];
      }
      cout << endl;

      // Multiply DeltaT^-1 T = L

      gsl_blas_dgemv(CblasNoTrans, 1.0, &matDTI.matrix, &vecB.vector, 0.0, &vecL.vector);

      // (5) Multiply R L = C

      double vY[maxjm*2];
      for (int iter=0; iter<GetMaxJM()*2; iter++) {
	vY[iter] = 0.0;
      }

      // Prepare inputs for solving the problem
      gsl_vector_view vecY = gsl_vector_view_array(vY, msize);

      gsl_blas_dgemv (CblasNoTrans, 1.0, &matR.matrix, &vecL.vector, 0.0, &vecY.vector);

      cout << "C vector packed " << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	cout << vY[ilmp];
      }
      cout << endl;

      int mpack=0;
      int el, em;
      for (int ilm=0; ilm<maxjm; ilm++) {
	// 	cfctreal[ilm]->SetBinContent(ibin, vC[mpack++]);
 	cfctreal[ilm]->SetBinContent(ibin, vY[mpack++]);
	GetElEmForIndex(ilm, &el, &em);
	if(em==0)
	  cfctimag[ilm]->SetBinContent(ibin, 0);
	else
	  //	  cfctimag[ilm]->SetBinContent(ibin, vC[mpack++]);
	  cfctimag[ilm]->SetBinContent(ibin, vY[mpack++]);
      }

      // invert the P matrix to get C errors
      //      double mS[maxjm*maxjm*4];

//       for (int iterz=0; iterz<msize; iterz++)
// 	for (int iterp=0; iterp<msize; iterp++)
// 	  if (iterp == iterz)
// 	    mS[iterz*msize + iterp] = 1.0;
// 	  else
// 	    mS[iterz*msize + iterp] = 0.0;

      //      gsl_matrix_view matS = gsl_matrix_view_array(mS, msize, msize);

      // Invert V

//       gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matP.matrix, &matS.matrix);

      mpack = 0;
      for (int ilm=0; ilm<maxjm; ilm++) {
 	cfctreal[ilm]->SetBinError(ibin, sqrt(mPInvertedPacked[mpack*msize+mpack]));
	mpack++;
	GetElEmForIndex(ilm, &el, &em);
	if(em==0)
	  cfctimag[ilm]->SetBinError(ibin, 0);
	else {
	  cfctimag[ilm]->SetBinError(ibin, sqrt(mPInvertedPacked[mpack*msize+mpack]));
	  mpack++;
	}
      }

    }
    if (1) {

      if (numsreal[0]->GetBinContent(ibin) > 0) {

      // Rewrite the new way to use the solving wherever there is inversion
      double mDeltaT[maxjm*maxjm*4];
      for (int ilmzero=0; ilmzero<GetMaxJM()*2; ilmzero++)
	for (int ilmprim=0; ilmprim<GetMaxJM()*2; ilmprim++)
	  mDeltaT[(ilmzero*maxjm*2) + ilmprim] = (covmnum[GetBin(ibin-1,ilmzero/2, ilmzero%2, ilmprim/2, ilmprim%2)]);

#ifdef _FINISH_DEBUG_
      cout << "Delta T matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
 	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mDeltaT[ilmz*GetMaxJM()*2 + ilmp];
 	}
 	cout << endl;
      }
#endif

      double mDeltaTPacked[maxjm*maxjm*4];
      int msize = PackYlmMatrixIndependentOnly(mDeltaT, mDeltaTPacked);

#ifdef _FINISH_DEBUG_
      cout << "Delta T matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mDeltaTPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      // (1) Solve (DeltaT)^1 Mtilde = Q

      // Prepare halper matrices

      double mM[maxjm*maxjm*4];
      double mMPacked[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mM[iter] = tMTilde[iter];
      PackYlmMatrixIndependentOnly(mM, mMPacked);

      gsl_matrix_view matM = gsl_matrix_view_array(mMPacked, msize, msize);

#ifdef _FINISH_DEBUG_
      cout << "Mtilde matrix " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
 	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mM[ilmz*GetMaxJM()*2 + ilmp];
 	}
 	cout << endl;
       }

      cout << "Mtilde matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mMPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      // Inverting matrix DeltaT.

      double mU[maxjm*maxjm*4];
      InvertYlmIndependentMatrix(mDeltaT, mU);

      double mDTInvertedPacked[maxjm*maxjm*4];
      PackYlmMatrixIndependentOnly(mU, mDTInvertedPacked);

      gsl_matrix_view matDTI = gsl_matrix_view_array(mDTInvertedPacked, msize, msize);

#ifdef _FINISH_DEBUG_
      cout << "Delta T matrix inverted packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
 	for (int ilmp=0; ilmp<msize; ilmp++) {
 	  cout.precision(3);
 	  cout.width(10);
 	  cout << mDTInvertedPacked[ilmz*msize + ilmp];
 	}
 	cout << endl;
      }
#endif

      // (2) Multiply DeltaT^1 M = Q
      double mQ[maxjm*maxjm*4];
      for (int iter=0; iter<msize*msize; iter++)
	mQ[iter] = 0.0;
      gsl_matrix_view matQ = gsl_matrix_view_array(mQ, msize, msize);

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &matDTI.matrix, &matM.matrix, 0.0, &matQ.matrix);

      double mTest[maxjm*maxjm*4];
      gsl_matrix_view matTest = gsl_matrix_view_array(mTest, msize, msize);

      double mF[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
 	mF[iter] = mDeltaTPacked[iter];
      gsl_matrix_view matF = gsl_matrix_view_array(mF, msize, msize);

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &matF.matrix, &matQ.matrix, 0.0, &matTest.matrix);

#ifdef _FINISH_DEBUG_
      cout << "Test matrix packed - compare to Mtilde" << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mTest[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      // (2) Multiply Mtilde^T Q = P

      double mP[maxjm*maxjm*4];
      for (int iter=0; iter<maxjm*maxjm*4; iter++)
	mP[iter] = 0;

      gsl_matrix_view matP = gsl_matrix_view_array(mP, msize, msize);

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &matM.matrix, &matQ.matrix, 0.0, &matP.matrix);

#ifdef _FINISH_DEBUG_
      cout << "P matrix packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mP[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      // (3) Solve P^-1 Mtilde^T = R
      double mPUnpacked[maxjm*maxjm*4];
      UnPackYlmMatrixIndependentOnly(mP, mPUnpacked, msize);

#ifdef _FINISH_DEBUG_
      cout << "P matrix unpacked " << endl;
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mPUnpacked[ilmz*GetMaxJM()*2 + ilmp];
	}
	cout << endl;
      }
#endif

      // Invert the P matrix

      double mPInverted[maxjm*maxjm*4];
      InvertYlmIndependentMatrix(mPUnpacked, mPInverted);

      double mPInvertedPacked[maxjm*maxjm*4];
      PackYlmMatrixIndependentOnly(mPInverted, mPInvertedPacked);

      gsl_matrix_view matPI = gsl_matrix_view_array(mPInvertedPacked, msize, msize);

#ifdef _FINISH_DEBUG_
      cout << "P matrix inverted packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
  	for (int ilmp=0; ilmp<msize; ilmp++) {
  	  cout.precision(3);
  	  cout.width(10);
  	  cout << mPInvertedPacked[ilmz*msize + ilmp];
  	}
  	cout << endl;
      }
#endif


//       //      gsl_matrix_view matR = gsl_matrix_view_array(mR, msize, msize);

//       double mG[maxjm*maxjm*4];
//       for (int iter=0; iter<maxjm*maxjm*4; iter++)
// 	mG[iter] = mP[iter];
//       gsl_matrix_view matG = gsl_matrix_view_array(mG, msize, msize);

//       // Decomposing the M matrix
//       gsl_linalg_SV_decomp(&matG.matrix, &matS.matrix, &vecST.vector, &vecWT.vector);

//       for (int itert=0; itert<msize; itert++) {
// 	for (int iterm=0; iterm<msize; iterm++)
// 	  vCT[iterm] = mMPacked[iterm*msize + itert];
// 	  // Transvere !!!      ^^^^^         ^^^^^

// 	// Solving the problem
// 	gsl_linalg_SV_solve(&matG.matrix, &matS.matrix, &vecST.vector, &vecCT.vector, &vecXT.vector);

// 	for (int iterm=0; iterm<msize; iterm++)
// 	  mR[itert*msize + iterm] = vXT[iterm];
//       }

      double mR[maxjm*maxjm*4];
      for (int ir=0; ir<maxjm*maxjm*4; ir++)
	mR[ir] = 0.0;
      gsl_matrix_view matR = gsl_matrix_view_array(mR, msize, msize);

      // (2) Multiply P^-1 M (Trans) = R

#ifdef _FINISH_DEBUG_
      cout << "Matrix M Packed " << endl;
      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mMPacked[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &matPI.matrix, &matM.matrix, 1.0, &matR.matrix);

#ifdef _FINISH_DEBUG_
      cout << "R matrix packed " << endl;

      for (int ilmz=0; ilmz<msize; ilmz++) {
	for (int ilmp=0; ilmp<msize; ilmp++) {
	  cout.precision(3);
	  cout.width(10);
	  cout << mR[ilmz*msize + ilmp];
	}
	cout << endl;
      }
#endif

      // (4) Solve DeltaT^-1 T = L
      double vL[maxjm*2];
      gsl_vector_view vecL = gsl_vector_view_array(vL, msize);

//       // Decomposing the M matrix
//       gsl_linalg_SV_decomp(&matF.matrix, &matS.matrix, &vecST.vector, &vecWT.vector);

      double vB[maxjm*2];
      for (int iter=0; iter<GetMaxJM(); iter++) {
	vB[iter*2] = real(tTq0[iter]);
	vB[iter*2+1] = imag(tTq0[iter]);
      }

      double vBPacked[maxjm*2];
      PackYlmVectorIndependentOnly(vB, vBPacked);

      gsl_vector_view vecB = gsl_vector_view_array(vBPacked, msize);

//       // Solving the problem
//       gsl_linalg_SV_solve(&matF.matrix, &matS.matrix, &vecST.vector, &vecB.vector, &vecL.vector);

#ifdef _FINISH_DEBUG_
      cout << "L vector packed " << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
 	cout.precision(3);
 	cout.width(10);
 	cout << vL[ilmp];
      }
      cout << endl;
#endif

      // Multiply DeltaT^-1 T = L

      gsl_blas_dgemv(CblasNoTrans, 1.0, &matDTI.matrix, &vecB.vector, 0.0, &vecL.vector);

      // (5) Multiply R L = C

      double vY[maxjm*2];
      for (int iter=0; iter<GetMaxJM()*2; iter++) {
	vY[iter] = 0.0;
      }

      // Prepare inputs for solving the problem
      gsl_vector_view vecY = gsl_vector_view_array(vY, msize);

      gsl_blas_dgemv (CblasNoTrans, 1.0, &matR.matrix, &vecL.vector, 0.0, &vecY.vector);

#ifdef _FINISH_DEBUG_
      cout << "C vector packed " << endl;
      for (int ilmp=0; ilmp<msize; ilmp++) {
	cout.precision(3);
	cout.width(10);
	cout << vY[ilmp];
      }
      cout << endl;
#endif
      int mpack=0;
      int el, em;
      for (int ilm=0; ilm<maxjm; ilm++) {
	// 	cfctreal[ilm]->SetBinContent(ibin, vC[mpack++]);
	GetElEmForIndex(ilm, &el, &em);
	if ((el%2)==1) {
	  cfctreal[ilm]->SetBinContent(ibin, 0.0);
	  cfctimag[ilm]->SetBinContent(ibin, 0.0);
	}
	if (em<0) {
	  cfctreal[ilm]->SetBinContent(ibin, 0.0);
	  cfctimag[ilm]->SetBinContent(ibin, 0.0);
	}
	else {
	  cfctreal[ilm]->SetBinContent(ibin, vY[mpack++]);
	  if(em==0)
	    cfctimag[ilm]->SetBinContent(ibin, 0);
	  else
	    //	  cfctimag[ilm]->SetBinContent(ibin, vC[mpack++]);
	    cfctimag[ilm]->SetBinContent(ibin, vY[mpack++]);
	}
      }

      // invert the P matrix to get C errors
      //      double mS[maxjm*maxjm*4];

//       for (int iterz=0; iterz<msize; iterz++)
// 	for (int iterp=0; iterp<msize; iterp++)
// 	  if (iterp == iterz)
// 	    mS[iterz*msize + iterp] = 1.0;
// 	  else
// 	    mS[iterz*msize + iterp] = 0.0;

      //      gsl_matrix_view matS = gsl_matrix_view_array(mS, msize, msize);

      // Invert V

//       gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matP.matrix, &matS.matrix);

      mpack = 0;
      for (int ilm=0; ilm<maxjm; ilm++) {
	GetElEmForIndex(ilm, &el, &em);
	if (em <0 ) {
	  cfctreal[ilm]->SetBinError(ibin, 0);
	  cfctimag[ilm]->SetBinError(ibin, 0);
	}
	else {
	  cfctreal[ilm]->SetBinError(ibin, sqrt(fabs(mPInvertedPacked[mpack*msize+mpack])));
	  mpack++;
	  if(em==0)
	    cfctimag[ilm]->SetBinError(ibin, 0);
	  else {
	    cfctimag[ilm]->SetBinError(ibin, sqrt(fabs(mPInvertedPacked[mpack*msize+mpack])));
	    mpack++;
	  }
	}
      }

      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	  if (ilmp > ilmz)
	    covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = mPInverted[ilmz*GetMaxJM()*2 + ilmp];
	  else
	    covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = mPInverted[ilmp*GetMaxJM()*2 + ilmz];
	}
      }
      }
      else {
	for (int ilm=0; ilm<maxjm; ilm++) {
	  cfctreal[ilm]->SetBinError(ibin, 0);
	  cfctimag[ilm]->SetBinError(ibin, 0);
	}

	for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++) {
	  for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	    covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = 0.0;
	  }
	}
      }
    }
  }

  PackCfcCovariance();
}

void CorrFctnDirectYlm::Write()
{
  for (int ilm=0; ilm<maxjm; ilm++) {
    numsreal[ilm]->Write();
    densreal[ilm]->Write();
    cfctreal[ilm]->Write();
    numsimag[ilm]->Write();
    densimag[ilm]->Write();
    cfctimag[ilm]->Write();
  }
  if (covnum) covnum->Write();
  if (covden) covden->Write();
  if (covcfc) covcfc->Write();
}

void CorrFctnDirectYlm::ReadFromFile(TFile *infile, const char *name, int maxl)
{
  if (maxl != fMaxL) {
    cout << "Cannot read function for L " << maxl << " into a container with L "<< fMaxL << endl;
    return;
  }
  cout << "Reading in numerators and denominators" << endl;
  cout << "Reading function " << name << endl;
  char bufname[200];
  for (int ihist=0; ihist<maxjm; ihist++) {
    sprintf(bufname, "NumReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    if (numsreal[ihist]) delete numsreal[ihist];
    cout << "Getting " << bufname << endl;
    numsreal[ihist] = new TH1D(*((TH1D *) infile->Get(bufname)));
//     cout << "Reading " << ihist << " " << elsi[ihist] << " " << (emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist]) << " " << bufname << endl;

    sprintf(bufname, "NumImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    if (numsimag[ihist]) delete numsimag[ihist];
    numsimag[ihist] = new TH1D(*((TH1D *) infile->Get(bufname)));
//     cout << "Reading " << ihist << " " << elsi[ihist] << " " << (emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist]) << " " << bufname << endl;
//     cout << numsimag[ihist]->GetBinContent(1) <<endl;

    sprintf(bufname, "DenReYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    if (densreal[ihist]) delete densreal[ihist];
    densreal[ihist] = new TH1D(*((TH1D *) infile->Get(bufname)));

    sprintf(bufname, "DenImYlm%i%i%s", elsi[ihist], emsi[ihist]<0 ? elsi[ihist]-emsi[ihist] : emsi[ihist], name);
    if (densimag[ihist]) delete densimag[ihist];
    densimag[ihist] = new TH1D(*((TH1D *) infile->Get(bufname)));
  }

  if (covnum) delete covnum;
  sprintf(bufname, "CovNum%s", name);
  if (infile->Get(bufname))
    covnum = new TH3D (*((TH3D *) infile->Get(bufname)));
  else
    covnum = 0;

  if (covden) delete covden;
  sprintf(bufname, "CovDen%s", name);
  if (infile->Get(bufname))
    covden = new TH3D (*((TH3D *) infile->Get(bufname)));
  else
    covden = 0;

  if ((covnum) && (covden)) {
    cout << "Unpacking covariance matrices from file " << endl;
    UnpackCovariances();
  }
  else {

    cout << "Creating fake covariance matrices" << endl;

    for (int ibin=1; ibin<=numsreal[0]->GetNbinsX(); ibin++) {
      double nent = numsreal[0]->GetEntries();
      double nentd = densreal[0]->GetEntries();
      for (int ilmx=0; ilmx<GetMaxJM(); ilmx++) {
	for (int ilmy=0; ilmy<GetMaxJM(); ilmy++) {
	  double t1t2rr = numsreal[ilmx]->GetBinContent(ibin)*numsreal[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ri = numsreal[ilmx]->GetBinContent(ibin)*numsimag[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ir = numsimag[ilmx]->GetBinContent(ibin)*numsreal[ilmy]->GetBinContent(ibin)/nent/nent;
	  double t1t2ii = numsimag[ilmx]->GetBinContent(ibin)*numsimag[ilmy]->GetBinContent(ibin)/nent/nent;
	  if (ilmx == ilmy) {
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nent*(TMath::Power(numsreal[ilmx]->GetBinError(ibin)/nent,2)*(nent-1) + t1t2rr);
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nent*t1t2ri;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nent*t1t2ir;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nent*(TMath::Power(numsimag[ilmx]->GetBinError(ibin)/nent,2)*(nent-1) + t1t2rr);
	  }
	  else {
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nent*t1t2rr;
	    covmnum[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nent*t1t2ri;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nent*t1t2ir;
	    covmnum[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nent*t1t2ii;
	  }
	  t1t2rr = densreal[ilmx]->GetBinContent(ibin)*densreal[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ri = densreal[ilmx]->GetBinContent(ibin)*densimag[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ir = densimag[ilmx]->GetBinContent(ibin)*densreal[ilmy]->GetBinContent(ibin)/nentd/nentd;
	  t1t2ii = densimag[ilmx]->GetBinContent(ibin)*densimag[ilmy]->GetBinContent(ibin)/nentd/nentd;

	  covmden[GetBin(ibin-1, ilmx, 0, ilmy, 0)] = nentd*t1t2rr;
	  covmden[GetBin(ibin-1, ilmx, 0, ilmy, 1)] = nentd*t1t2ri;
	  covmden[GetBin(ibin-1, ilmx, 1, ilmy, 0)] = nentd*t1t2ir;
	  covmden[GetBin(ibin-1, ilmx, 1, ilmy, 1)] = nentd*t1t2ii;
	}
      }
    }
  }

  // Recalculating the correlation functions
  Finish();
}

int CorrFctnDirectYlm::PackYlmVector(double *invec, double *outvec)
{
  int ioutcount = 0;
  int em, el;
  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    GetElEmForIndex(ilm, &el, &em);
    outvec[ioutcount++] = invec[ilm*2];
    if (em == 0)
      continue;
    outvec[ioutcount++] = invec[ilm*2 + 1];
  }

  return ioutcount;
}

int  CorrFctnDirectYlm::PackYlmVectorIndependentOnly(double *invec, double *outvec){
  int ioutcount = 0;
  int em, el;
  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    GetElEmForIndex(ilm, &el, &em);
    if (em<0) continue;
    outvec[ioutcount++] = invec[ilm*2];
    if (em == 0)
      continue;
    outvec[ioutcount++] = invec[ilm*2 + 1];
  }

  return ioutcount;
}

int CorrFctnDirectYlm::PackYlmMatrix(double *inmat, double *outmat)
{
  int ioutcountz = 0;
  int ioutcountp = 0;
  int emz, elz;
  int emp, elp;
  int finalsize = 0;

  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    GetElEmForIndex(ilm, &elz, &emz);
    finalsize++;
    if (emz == 0) continue;
    finalsize++;
  }

  for (int ilmz=0; ilmz<GetMaxJM(); ilmz++) {
    GetElEmForIndex(ilmz, &elz, &emz);
    ioutcountp = 0;
    for (int ilmp=0; ilmp<GetMaxJM(); ilmp++) {
      GetElEmForIndex(ilmp, &elp, &emp);
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 0, ilmp, 0)];
      ioutcountp++;
      if (emp == 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 0, ilmp, 1)];
      ioutcountp++;
    }
    ioutcountz++;

    if (emz == 0) continue;
    ioutcountp = 0;
    for (int ilmp=0; ilmp<GetMaxJM(); ilmp++) {
      GetElEmForIndex(ilmp, &elp, &emp);
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 1, ilmp, 0)];
      ioutcountp++;
      if (emp == 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 1, ilmp, 1)];
      ioutcountp++;
    }
    ioutcountz++;
  }

  return ioutcountz;
}

void CorrFctnDirectYlm::UnPackYlmMatrix(double *inmat, double *outmat, int insize)
{
  int lmax = (-3 + sqrt(1 + 8*insize)) / 4;
  int maxt = 2*(lmax+1)*(lmax+1);
  int cntz = 0;
  int cntp = 0;
  int elz, emz, elp, emp;

  //  cout << "Sizes " << insize << " " << lmax << " " << maxt  <<endl;

  for (int ilmz =0; ilmz<maxt; ilmz++) {
    GetElEmForIndex(ilmz/2, &elz, &emz);
    if ((emz == 0) && (ilmz %2)) {
      for (int ilmp=0; ilmp<maxt; ilmp++) {
	outmat[ilmz*maxt + ilmp] = 0.0;
      }
    }
    else {
      cntp = 0;
      for (int ilmp=0; ilmp<maxt; ilmp++) {
	GetElEmForIndex(ilmp/2, &elp, &emp);
	if ((emp == 0) && (ilmp % 2)) {
	  outmat[ilmz*maxt + ilmp] = 0.0;
	}
	else
	  outmat[ilmz*maxt + ilmp] = inmat[cntz*insize + cntp++];
      }
      cntz++;
    }
  }
}

int CorrFctnDirectYlm::PackYlmMatrixIndependentOnly(double *inmat, double *outmat)
{
  int ioutcountz = 0;
  int ioutcountp = 0;
  int emz, elz;
  int emp, elp;
  int finalsize = 0;

  for (int ilm=0; ilm<GetMaxJM(); ilm++) {
    GetElEmForIndex(ilm, &elz, &emz);
    if (emz < 0) continue;
    finalsize++;
    if (emz == 0) continue;
    finalsize++;
  }

  //  cout << "Final size " << finalsize << endl;

  for (int ilmz=0; ilmz<GetMaxJM(); ilmz++) {
    GetElEmForIndex(ilmz, &elz, &emz);
    ioutcountp = 0;

    if (emz < 0) continue;
    for (int ilmp=0; ilmp<GetMaxJM(); ilmp++) {
      GetElEmForIndex(ilmp, &elp, &emp);
      if (emp < 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 0, ilmp, 0)];
      ioutcountp++;
      if (emp == 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 0, ilmp, 1)];
      ioutcountp++;
    }
    ioutcountz++;

    if (emz == 0) continue;
    ioutcountp = 0;
    for (int ilmp=0; ilmp<GetMaxJM(); ilmp++) {
      GetElEmForIndex(ilmp, &elp, &emp);
      if (emp < 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 1, ilmp, 0)];
      ioutcountp++;
      if (emp == 0) continue;
      outmat[ioutcountz*finalsize + ioutcountp] = inmat[GetBin(0, ilmz, 1, ilmp, 1)];
      ioutcountp++;
    }
    ioutcountz++;
  }

  return ioutcountz;
}

void CorrFctnDirectYlm::GetIndependentLM(int ibin, int *el, int *em, int *im)
{
  int cbin=ibin;
  if (cbin == 0) { *el=0; *em=0; *im=0; return; }
  else cbin--;
  if (cbin == 0) { *el=2; *em=0; *im=0; return; }
  else cbin--;
  *im = cbin%2;
  *el = 2;
  *em = cbin/2+1;

  return;
}

int CorrFctnDirectYlm::UnPackYlmMatrixIndependentOnly(double *inmat, double *outmat, int insize)
{
  int lmax = sqrt(insize) - 1;
  //  cout << "lmax is  " << lmax << endl;
  if (0) {
    lmax *=2;
  }
  int tmax = (lmax+1)*(lmax+1)*2;
  int indexfrom[tmax];
  int multfrom[tmax];

  int el, em;
  if (1) {
    for (int iter=0; iter<tmax; iter++) {
      int im = iter % 2;
      GetElEmForIndex(iter/2, &el, &em);
      if (em == 0) {
	if (im == 1) {
	  indexfrom[iter] = 0;
	  multfrom[iter] = 0;
	}
	else {
	  indexfrom[iter] =  el*el;
	  multfrom[iter] = 1;
	}
      }
      else if (em < 0) {
	indexfrom[iter] = (el*el) + (-em)*2 - 1;
	if (im) indexfrom[iter]++;
	if ((-em)%2)
	  if (im) multfrom[iter] = 1;
	  else multfrom[iter] = -1;
	else
	  if (im) multfrom[iter] = -1;
	  else multfrom[iter] = 1;
      }
      else if (em > 0) {
	indexfrom[iter] = (el*el) + (em)*2 - 1;
	if (im) indexfrom[iter]++;
	multfrom[iter] = 1;
      }
    }
  }
  if (0) {
    int lc, mc;
    for (int iter=0; iter<tmax; iter++) {
      indexfrom[iter]=0;
      multfrom[iter]=0;
    }
    int imax = lmax/2;
    int ibmax = 0;
    int elz, emz, imz, elp, emp, imp;
    for (int it=0; it<=imax; it++) {
      ibmax++;
      ibmax += it*2*2;
    }
    for (int iz=0; iz<ibmax; iz++) {
      GetIndependentLM(iz, &elz, &emz, &imz);
      for (int ip=0; ip<ibmax; ip++) {
	GetIndependentLM(ip, &elp, &emp, &imp);
	indexfrom[GetBin(0, GetIndexForLM(elz, emz), imz, GetIndexForLM(elp, emp), imp)] = iz*ibmax+ip;
	multfrom[GetBin(0, GetIndexForLM(elz, emz), imz, GetIndexForLM(elp, emp), imp)] = 1.0;
      }
    }
  }
//   cout << "From Mult " << endl;
//   for (int iter=0; iter<tmax; iter++)
//     cout << indexfrom[iter] << " ";
//   cout << endl;
//   for (int iter=0; iter<tmax; iter++)
//     cout << multfrom[iter] << " ";
//   cout << endl;

  for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
    for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
      outmat[ilmz*GetMaxJM()*2 + ilmp] = inmat[(indexfrom[ilmz]*insize) + indexfrom[ilmp]] * multfrom[ilmz]*multfrom[ilmp];
}

void CorrFctnDirectYlm::PackCovariances()
{
  char bufname[200];
  sprintf(bufname, "CovNum%s", numsreal[0]->GetName()+10);

  if (covnum) delete covnum;
  covnum = new TH3D(bufname,bufname,
		    numsreal[0]->GetNbinsX(), numsreal[0]->GetXaxis()->GetXmin(), numsreal[0]->GetXaxis()->GetXmax(),
		    GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5,
		    GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5);

  for (int ibin=1; ibin<=covnum->GetNbinsX(); ibin++)
    for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
      for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	covnum->SetBinContent(ibin, ilmz+1, ilmp+1, covmnum[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)]);

  if (covden) delete covden;
  sprintf(bufname, "CovDen%s", numsreal[0]->GetName()+10);
  covden  = new TH3D(bufname,bufname,
		     densreal[0]->GetNbinsX(), densreal[0]->GetXaxis()->GetXmin(), densreal[0]->GetXaxis()->GetXmax(),
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5,
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5);

  for (int ibin=1; ibin<=covden->GetNbinsX(); ibin++)
    for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
      for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	covden->SetBinContent(ibin, ilmz+1, ilmp+1, covmden[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)]);

  if (covcfc) delete covcfc;
  sprintf(bufname, "CovCfc%s", numsreal[0]->GetName()+10);
  covcfc  = new TH3D(bufname,bufname,
		     cfctreal[0]->GetNbinsX(), cfctreal[0]->GetXaxis()->GetXmin(), cfctreal[0]->GetXaxis()->GetXmax(),
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5,
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5);

  for (int ibin=1; ibin<=covcfc->GetNbinsX(); ibin++)
    for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
      for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	covcfc->SetBinContent(ibin, ilmz+1, ilmp+1, covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)]);

}

void CorrFctnDirectYlm::PackCfcCovariance()
{
  char bufname[200];

  if (covcfc) delete covcfc;
  sprintf(bufname, "CovCfc%s", numsreal[0]->GetName()+10);
  covcfc  = new TH3D(bufname,bufname,
		     cfctreal[0]->GetNbinsX(), cfctreal[0]->GetXaxis()->GetXmin(), cfctreal[0]->GetXaxis()->GetXmax(),
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5,
		     GetMaxJM()*2, -0.5, GetMaxJM()*2 - 0.5);

  double tK, tE, tB;
  for (int ibin=1; ibin<=covcfc->GetNbinsX(); ibin++)
    for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
      for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++) {
	tK = covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)];
// 	tE = cfctreal[0]->GetEntries();
// 	if (ilmz%2) {
// 	  if (ilmp%2) {
// 	    tB = cfctimag[ilmz/2]->GetBinContent(ibin)*cfctimag[ilmp/2]->GetBinContent(ibin);
// 	  }
// 	  else {
// 	    tB = cfctimag[ilmz/2]->GetBinContent(ibin)*cfctreal[ilmp/2]->GetBinContent(ibin);
// 	  }
// 	}
// 	else {
// 	  if (ilmp%2) {
// 	    tB = cfctreal[ilmz/2]->GetBinContent(ibin)*cfctimag[ilmp/2]->GetBinContent(ibin);
// 	  }
// 	  else {
// 	    tB = cfctreal[ilmz/2]->GetBinContent(ibin)*cfctreal[ilmp/2]->GetBinContent(ibin);
// 	  }
// 	}

	covcfc->SetBinContent(ibin, ilmz+1, ilmp+1, tK);
      }
  covcfc->SetBinContent(0,0,0,1.0);
}

void CorrFctnDirectYlm::InvertYlmIndependentMatrix(double *inmat, double *outmat)
{
  // Invert the Ylm matrix by inverting only the matrix
  // with independent elements and filling in the rest
  // according to sign rules

  double mU[maxjm*maxjm*4];
  int isize = PackYlmMatrixIndependentOnly(inmat, mU);
  //  cout << "Independent count " << isize << endl;

  gsl_matrix_view matU = gsl_matrix_view_array(mU, isize, isize);

//   cout << "Input matrix independent only " << endl;
//   for (int ilmz=0; ilmz<isize; ilmz++) {
//     for (int ilmp=0; ilmp<isize; ilmp++) {
//       cout.precision(3);
//       cout.width(10);
//       cout << mU[ilmz*isize + ilmp];
//     }
//     cout << endl;
//   }

  // Identity matrix helper for inversion
  double mI[maxjm*maxjm*4];
  for (int iterm=0; iterm<isize; iterm++)
    for (int iterp=0; iterp<isize; iterp++)
      if (iterm == iterp)
	mI[iterm*isize+iterp] = 1.0;
      else
	mI[iterm*isize+iterp] = 0.0;
  gsl_matrix_view matI = gsl_matrix_view_array(mI, isize, isize);

  // Invert the matrix
  gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, &matU.matrix, &matI.matrix);

//   cout << "Input independent only inverted " << endl;
//   for (int ilmz=0; ilmz<isize; ilmz++) {
//     for (int ilmp=0; ilmp<isize; ilmp++) {
//       cout.precision(3);
//       cout.width(10);
//       cout << mI[ilmz*isize + ilmp];
//     }
//     cout << endl;
//   }

  // Fill in the lower triangle - the matrix must be symmetric
//   for (int ilmz=0; ilmz<isize; ilmz++) {
//     for (int ilmp=ilmz+1; ilmp<isize; ilmp++) {
//       mI[ilmp*isize + ilmz] = mI[ilmz*isize + ilmp];
//     }
//   }

//   cout << "Input independent only inverted filled in" << endl;
//   for (int ilmz=0; ilmz<isize; ilmz++) {
//     for (int ilmp=0; ilmp<isize; ilmp++) {
//       cout.precision(3);
//       cout.width(10);
//       cout << mI[ilmz*isize + ilmp];
//     }
//     cout << endl;
//   }

  UnPackYlmMatrixIndependentOnly(mI, outmat, isize);
}

void CorrFctnDirectYlm::UnpackCovariances()
{
  if (covnum) {
    for (int ibin=1; ibin<=covnum->GetNbinsX(); ibin++)
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	  covmnum[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = covnum->GetBinContent(ibin, ilmz+1, ilmp+1);

  }
  if (covden) {
    for (int ibin=1; ibin<=covden->GetNbinsX(); ibin++)
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	  covmden[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = covden->GetBinContent(ibin, ilmz+1, ilmp+1);
  }

  if (covcfc) {
    for (int ibin=1; ibin<=covcfc->GetNbinsX(); ibin++)
      for (int ilmz=0; ilmz<GetMaxJM()*2; ilmz++)
	for (int ilmp=0; ilmp<GetMaxJM()*2; ilmp++)
	  covmcfc[GetBin(ibin-1, ilmz/2, ilmz%2, ilmp/2, ilmp%2)] = covcfc->GetBinContent(ibin, ilmz+1, ilmp+1);
  }
}

int CorrFctnDirectYlm::GetIndexForLM(int el, int em)
{
  for (int iter=0; iter<maxjm; iter++)
    if ((el == elsi[iter]) && (em == emsi[iter]))
      return iter;
  return -1;
}

TH1D *CorrFctnDirectYlm::GetNumRealHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)numsreal[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}
TH1D *CorrFctnDirectYlm::GetNumImagHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)numsimag[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}

TH1D *CorrFctnDirectYlm::GetDenRealHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)densreal[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}
TH1D *CorrFctnDirectYlm::GetDenImagHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)densimag[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}

TH1D *CorrFctnDirectYlm::GetCfnRealHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)cfctreal[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}
TH1D *CorrFctnDirectYlm::GetCfnImagHist(int el, int em)
{
  if (GetIndexForLM(el, em)>=0)
    return (TH1D*)cfctimag[GetIndexForLM(el, em)]->Clone();
  else
    return 0;
}


void CorrFctnDirectYlm::SetAdvancedNormalization(double radius, double bohr, double purity, int binmin, int binmax)
{
  mNormRadius = radius;
  mNormBohr = bohr;
  mNormPurity = purity;
  mNormBinMin = binmin;
  mNormBinMax = binmax;
}
