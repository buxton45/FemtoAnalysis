#include "SpherHarmonics.h"



//________________________________________________________________________________________________________________
SpherHarmonics::SpherHarmonics() 
{

}


//________________________________________________________________________________________________________________
SpherHarmonics::~SpherHarmonics()
{

}

//________________________________________________________________________________________________________________
complex<double> SpherHarmonics::ceiphi(double phi){
  return complex<double>(cos(phi),sin(phi));
}

//________________________________________________________________________________________________________________
double SpherHarmonics::legendre(int ell,double ctheta){
  return gsl_sf_legendre_Pl(ell,ctheta);
}

//________________________________________________________________________________________________________________
complex<double> SpherHarmonics::Ylm(int ell,int m,double theta,double phi){
  double ctheta;
  complex<double> answer;
  complex<double> ci(0.0,1.0);
  ctheta=cos(theta);
  answer=gsl_sf_legendre_sphPlm(ell,abs(m),ctheta)*ceiphi(m*phi);
  //  if(m<0) answer=answer*pow(-1.0,abs(m));
  if (m<0) {
    if (abs(m) % 2) answer *= -1.0;
  }
  return answer;
}

//________________________________________________________________________________________________________________
complex<double> SpherHarmonics::Ylm(int ell, int m, double x, double y, double z){
  complex<double> answer;
  double ctheta,phi;
  double r = sqrt(x*x+y*y+z*z);
  if ( r < 1e-10 || fabs(z) < 1e-10 ) ctheta = 0.0;
  else ctheta=z/r;
  phi=atan2(y,x);
  answer=gsl_sf_legendre_sphPlm(ell,abs(m),ctheta)*ceiphi(m*phi);
  //  if(m<0) answer=answer*pow(-1.0,abs(m));
  if (m<0) {
    if (abs(m) % 2) answer *= -1.0;
  }
  return answer;	
}

//________________________________________________________________________________________________________________
void SpherHarmonics::YlmUpToL(int lmax, double x, double y, double z, complex<double> *ylms)
{
  complex<double> answer;
  double ctheta,phi;
  int lcur = 0;  
  double lpol;
  
  double coss[lmax];
  double sins[lmax];

  double r = sqrt(x*x+y*y+z*z);
  if ( r < 1e-10 || fabs(z) < 1e-10 ) ctheta = 0.0;
  else ctheta=z/r;
  phi=atan2(y,x);
  
  for (int iter=1; iter<=lmax; iter++) {
    coss[iter-1] = cos(iter*phi);
    sins[iter-1] = sin(iter*phi);
  }
  ylms[lcur++] = gsl_sf_legendre_sphPlm(0, 0, ctheta) * complex<double>(1,0);
  
  for (int il = 1; il<=lmax; il++) {
    for (int im=0; im<=il; im++) {
      lpol = gsl_sf_legendre_sphPlm(il, im, ctheta);
      if (im) {
	ylms[lcur+il+im] = lpol*complex<double>(coss[im-1],sins[im-1]);
	if (im%2)
	  ylms[lcur+il-im] = -lpol*complex<double>(coss[im-1],-sins[im-1]);
	else
	  ylms[lcur+il-im] = lpol*complex<double>(coss[im-1],-sins[im-1]);
      }
      else {
	ylms[lcur+il] = lpol*complex<double>(1,0);
      }
    }
    lcur += 2*il + 1;
  }
}

//________________________________________________________________________________________________________________
void SpherHarmonics::YlmUpToL(int lmax, double ctheta, double phi, complex<double> *ylms)
{
  int lcur = 0;  
  double lpol;
  
  double coss[lmax];
  double sins[lmax];

  for (int iter=1; iter<=lmax; iter++) {
    coss[iter-1] = cos(iter*phi);
    sins[iter-1] = sin(iter*phi);
  }
  ylms[lcur++] = gsl_sf_legendre_sphPlm(0, 0, ctheta) * complex<double>(1,0);
  
  for (int il = 1; il<=lmax; il++) {
    for (int im=0; im<=il; im++) {
      lpol = gsl_sf_legendre_sphPlm(il, im, ctheta);
      if (im) {
	ylms[lcur+il+im] = lpol*complex<double>(coss[im-1],sins[im-1]);
	if (im%2)
	  ylms[lcur+il-im] = -lpol*complex<double>(coss[im-1],-sins[im-1]);
	else
	  ylms[lcur+il-im] = lpol*complex<double>(coss[im-1],-sins[im-1]);
      }
      else {
	ylms[lcur+il] = lpol*complex<double>(1,0);
      }
    }
    lcur += 2*il + 1;
  }
}

//________________________________________________________________________________________________________________
double SpherHarmonics::ReYlm(int ell, int m, double theta, double phi){
	return real(SpherHarmonics::Ylm(ell,m,theta,phi));
}

//________________________________________________________________________________________________________________
double SpherHarmonics::ImYlm(int ell, int m, double theta, double phi){
	return imag(SpherHarmonics::Ylm(ell,m,theta,phi));
}

//________________________________________________________________________________________________________________
double SpherHarmonics::ReYlm(int ell, int m, double x,double y,double z){
	return real(SpherHarmonics::Ylm(ell,m,x,y,z));
}

//________________________________________________________________________________________________________________
double SpherHarmonics::ImYlm(int ell, int m, double x,double y,double z){
	return imag(SpherHarmonics::Ylm(ell,m,x,y,z));
}
