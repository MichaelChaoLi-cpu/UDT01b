/* Resolve the complete production shared library without calling solver entry points. */
#include <dlfcn.h>
#include <stdio.h>
int main(void){
 const char *names[]={"__sfincs_lib_MOD_sfincs_initialize","__sfincs_lib_MOD_sfincs_update","__sfincs_lib_MOD_sfincs_finalize","__sfincs_ncinput_MOD_read_netcdf_ampr_data","__sfincs_continuity_MOD_compute_water_levels","__udt_rain_clock_MOD_udt_decode_rain_seconds","__udt_clock_budget_MOD_udt_budget_initialize","__udt_clock_budget_MOD_udt_budget_finish","__udt_clock_budget_MOD_udt_budget_capture","__udt_clock_budget_MOD_udt_budget_close"};
 void *library=dlopen("/usr/local/lib/libsfincs.so",RTLD_NOW|RTLD_LOCAL);
 if(!library){fprintf(stderr,"%s\n",dlerror());return 1;}
 for(unsigned i=0;i<sizeof(names)/sizeof(*names);i++){
  dlerror();void *address=dlsym(library,names[i]);const char *error=dlerror();
  if(error||!address){fprintf(stderr,"Missing %s: %s\n",names[i],error?error:"null");return 2;}
  printf("RESOLVED %s\n",names[i]);
 }
 dlclose(library);puts("PASS: RTLD_NOW library load and symbol resolution only; zero solver entry-point calls");return 0;
}
