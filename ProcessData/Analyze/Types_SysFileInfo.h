/* Types_SysFileInfo.h */


#ifndef TYPES_SYSFILEINFO_H
#define TYPES_SYSFILEINFO_H

#include <vector>
#include <complex>
#include <cassert>
#include <iostream>
using std::vector;
using namespace std;

#include "TString.h"

#include "Types.h"

#include "Types_SysFileInfo_20161027.h"
#include "Types_SysFileInfo_20180505.h"
#include "Types_SysFileInfo_20190319.h"

//----------------------------------------------------


extern const SystematicsFileInfo GetFileInfo_LamK(int aNumber, TString aParentDate);
extern const SystematicsFileInfo GetFileInfo_XiKch(int aNumber, TString aParentDate);



#endif







