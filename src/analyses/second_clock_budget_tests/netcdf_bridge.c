/* Test-only bridge to the installed real NetCDF C library; no solver dependency. */
#include <netcdf.h>
#include <stddef.h>
#include <string.h>
int udt_nc_dimlen(int id,int dim,int *len){size_t n;int e=nc_inq_dimlen(id,dim,&n);*len=(int)n;return e;}
int udt_nc_att(int id,int var,const char *name,char *dst,int cap){
 size_t n=0;int e=nc_inq_attlen(id,var,name,&n);if(e)return e;
 if(n>(size_t)cap)return NC_EMAXNAME;
 memset(dst,' ',cap);return nc_get_att_text(id,var,name,dst);
}
int udt_nc_slice(int id,int var,float *dst,const int *start,const int *count){
 size_t a[3],b[3];for(int i=0;i<3;i++){a[i]=start[2-i]-1;b[i]=count[2-i];}
 return nc_get_vara_float(id,var,a,b,dst);
}
