/* Types_SysFileInfo.cxx */

#include "Types_SysFileInfo.h"

#include "Types_SysFileInfo_20161027.cxx"
#include "Types_SysFileInfo_20180505.cxx"
#include "Types_SysFileInfo_20181205.cxx"
#include "Types_SysFileInfo_20190319.cxx"

#include <cassert>


//***************************************************************************************************************************************

const SystematicsFileInfo GetFileInfo_LamK(int aNumber, TString aParentDate)
{
  if     (aParentDate.EqualTo("20161027")) return GetFileInfo_LamK_20161027(aNumber);
  else if(aParentDate.EqualTo("20180505")) return GetFileInfo_LamK_20180505(aNumber);
  else if(aParentDate.EqualTo("20181205")) return GetFileInfo_LamK_20181205(aNumber);
  else if(aParentDate.EqualTo("20190319")) return GetFileInfo_LamK_20190319(aNumber);
  else assert(0);
}



//***************************************************************************************************************************************

const SystematicsFileInfo GetFileInfo_XiKch(int aNumber, TString aParentDate)
{
  if     (aParentDate.EqualTo("20161027")) return GetFileInfo_XiKch_20161027(aNumber);
  else if(aParentDate.EqualTo("20180505")) return GetFileInfo_XiKch_20180505(aNumber);
  else if(aParentDate.EqualTo("20190319")) return GetFileInfo_XiKch_20190319(aNumber);
  else assert(0);
}




