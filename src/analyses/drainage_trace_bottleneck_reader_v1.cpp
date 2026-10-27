// Read-only streaming aggregation of previously qualified pump-step CSV logs.
// Build on macOS: clang++ -O3 -std=c++17 this.cpp -o /private/tmp/udt_drainage_trace_reader
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>
#include <CommonCrypto/CommonDigest.h>
struct HashBuffer:std::streambuf {
 FILE* f; std::vector<char> buf; CC_SHA256_CTX ctx;
 explicit HashBuffer(const char*p):f(std::fopen(p,"rb")),buf(8*1024*1024){if(!f)throw std::runtime_error("Cannot open trace");CC_SHA256_Init(&ctx);}
 int_type underflow()override{size_t n=std::fread(buf.data(),1,buf.size(),f);if(!n)return traits_type::eof();CC_SHA256_Update(&ctx,buf.data(),(CC_LONG)n);setg(buf.data(),buf.data(),buf.data()+n);return traits_type::to_int_type(*gptr());}
 std::string digest(){unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(d,&ctx);char out[65];for(int i=0;i<32;++i)std::sprintf(out+2*i,"%02x",d[i]);out[64]=0;return out;}
 ~HashBuffer(){std::fclose(f);}
};
void check(bool x,const char*m){if(!x)throw std::runtime_error(m);}
struct Site{int id,pool,source,receiver;double rate;};
// volumes or durations, with clear non-causal fixed-state screening semantics.
struct Agg{double nominal=0,actual=0,trigger=0,head=0,both=0,storage=0,available=0,envelope=0,saturated=0,deep_saturated=0,wet=0,deep=0,seconds=0,maximum=0;};
Agg add(Agg a,Agg b){a.nominal+=b.nominal;a.actual+=b.actual;a.trigger+=b.trigger;a.head+=b.head;a.both+=b.both;a.storage+=b.storage;a.available+=b.available;a.envelope+=b.envelope;a.saturated+=b.saturated;a.deep_saturated+=b.deep_saturated;a.wet+=b.wet;a.deep+=b.deep;a.seconds+=b.seconds;a.maximum=std::max(a.maximum,b.maximum);return a;}
Agg scale(Agg a,double x){a.nominal*=x;a.actual*=x;a.trigger*=x;a.head*=x;a.both*=x;a.storage*=x;a.available*=x;a.envelope*=x;a.saturated*=x;a.deep_saturated*=x;a.wet*=x;a.deep*=x;a.seconds*=x;return a;}
void fields(std::ostream&o,const Agg&a){o<<a.nominal<<','<<a.actual<<','<<a.trigger<<','<<a.head<<','<<a.both<<','<<a.storage<<','<<a.available<<','<<a.envelope<<','<<a.saturated<<','<<a.deep_saturated<<','<<a.wet<<','<<a.deep<<','<<a.seconds<<','<<a.maximum;}
const char*HEADER="nominal_reserved_m3,actual_m3,trigger_only_nominal_m3,head_only_nominal_m3,both_blocked_nominal_m3,eligible_storage_shortfall_nominal_m3,eligible_reserved_available_m3,fixed_state_pool_envelope_m3,saturated_seconds,saturated_with_deep_intake_seconds,eligible_intake_seconds,deep_eligible_intake_seconds,observed_seconds,maximum_actual_rate_m3_s";
int main(int argc,char**argv){try{
 check(argc==5,"Need trace metadata output-prefix O1-or-static");bool dynamic=std::string(argv[4])=="O1";
 std::ifstream meta(argv[2]);std::vector<Site> sites;Site s;
 while(meta>>s.id>>s.pool>>s.source>>s.receiver>>s.rate)sites.push_back(s);
 int n=sites.size();check(n>0,"No sites");std::array<double,5> nominal{};
 for(int i=0;i<n;i++){check(sites[i].id==i+1&&sites[i].pool>=0&&sites[i].pool<5,"Bad metadata");nominal[sites[i].pool]+=sites[i].rate;}
 for(double x:nominal)check(std::abs(x-.5)<1e-10,"Frozen pool allowance mismatch");
 HashBuffer buf(argv[1]);std::istream in(&buf);std::string line;std::getline(in,line);
 check(line.find("actual_end_time_s,dt_s,pump_id")==0,"Bad trace header");
 std::vector<Agg> siteagg(n);std::array<Agg,5> poolagg{};std::array<std::array<Agg,5>,108> bins{};
 long long records=0,steps=0;double previous=0,end=0,dt=0,maxerr=0;const double reserve=1+64*std::pow(2.,-23);
 while(std::getline(in,line)){
  if(line.empty())continue;
  std::array<Agg,5> current{};std::array<double,5> hardware{};std::array<bool,5> deep{};
  for(int i=0;i<n;i++){
   if(i){check(bool(std::getline(in,line)),"Partial step");}
   char *p=line.data(),*tail;double v[11];
   for(int k=0;k<11;k++){v[k]=std::strtod(p,&tail);check(tail!=p&&*tail==',',"Invalid numeric CSV field");p=tail+1;}
   bool limited=(*p=='T');p=std::strchr(p,',');check(p,"Missing gate flag");bool blocked=(*(p+1)=='T');
   const Site &si=sites[i];check((int)v[2]==si.id&&(int)v[3]==si.source&&(int)v[4]==si.receiver,"Site identity mismatch");
   if(i==0){end=v[0];dt=v[1];check(dt>0&&std::abs(end-previous-dt)<1e-5,"Clock mismatch");}
   else check(std::abs(v[0]-end)<1e-9&&std::abs(v[1]-dt)<1e-9,"Within-step clock mismatch");
   double requested=v[6],actual=v[7],prior=v[8],head=v[9],depth=v[10];bool tr=depth<.05,hd=head>5,eligible=!(tr||hd);
   check(blocked==!eligible,"Trigger/head flag mismatch");check(actual>=0&&actual<=requested+1e-10&&actual<=prior+1e-7,"Transfer bounds failed");
   double req=(dynamic?.5:si.rate)*dt;check(std::abs(req-requested)<1e-10,"Requested rate mismatch");
   check(limited==((prior/reserve<requested)&&eligible),"Storage flag mismatch");
   double nr=si.rate*dt,avail=eligible?std::min(nr,std::max(prior/reserve,0.)):0.;
   Agg a;a.nominal=nr;a.actual=actual;a.trigger=tr&&!hd?nr:0;a.head=hd&&!tr?nr:0;a.both=tr&&hd?nr:0;a.storage=eligible?nr-avail:0;a.available=avail;a.wet=eligible?dt:0;a.deep=eligible&&depth>=.3?dt:0;a.seconds=dt;a.maximum=actual/dt;
   siteagg[i]=add(siteagg[i],a);current[si.pool]=add(current[si.pool],a);
   hardware[si.pool]+=eligible?std::min(.5*dt,std::max(prior/reserve,0.)):0.;deep[si.pool]=deep[si.pool]||(eligible&&depth>=.3);
   if(!dynamic)maxerr=std::max(maxerr,std::abs(actual-avail));
   ++records;
  }
  for(int g=0;g<5;g++){
   Agg &a=current[g];a.seconds=dt;a.maximum=a.actual/dt;a.envelope=std::min(.5*dt,hardware[g]);
   check(a.actual<=a.envelope+1e-8,"Pool extraction exceeds fixed-state envelope");
   if(dynamic)maxerr=std::max(maxerr,std::abs(a.actual-a.envelope));
   bool sat=a.actual>=.5*dt*(1.-1e-4);a.saturated=sat?dt:0.;a.deep_saturated=sat&&deep[g]?dt:0.;
   poolagg[g]=add(poolagg[g],a);
   // Split crossing steps conservatively across exact five-minute bins.
   double st=std::max(0.,end-dt),last=std::min(32400.,end);
   while(st<last-1e-12){int bi=std::min(107,(int)std::floor(st/300));double stop=std::min(last,(bi+1)*300.);check(stop>st,"Invalid bin split");bins[bi][g]=add(bins[bi][g],scale(a,(stop-st)/dt));st=stop;}
  }
  previous=end;++steps;
 }
 check(std::abs(end-32400)<.001&&maxerr<1e-7,"End clock or operator expectation failed");
 std::string prefix=argv[3];std::ofstream pools(prefix+"_pool.csv"),bytime(prefix+"_time.csv"),ss(prefix+"_site.csv"),js(prefix+"_validation.json");
 pools<<std::setprecision(17)<<"pool_id,"<<HEADER<<'\n';bytime<<std::setprecision(17)<<"bin,start_s,end_s,pool_id,"<<HEADER<<'\n';ss<<std::setprecision(17)<<"pump_id,pool_id,"<<HEADER<<'\n';
 for(int g=0;g<5;g++){pools<<g<<',';fields(pools,poolagg[g]);pools<<'\n';}
 for(int b=0;b<108;b++)for(int g=0;g<5;g++){bytime<<b<<','<<b*300<<','<<(b+1)*300<<','<<g<<',';fields(bytime,bins[b][g]);bytime<<'\n';}
 for(int i=0;i<n;i++){ss<<i+1<<','<<sites[i].pool<<',';fields(ss,siteagg[i]);ss<<'\n';}
 js<<std::setprecision(17)<<"{\"passed\":true,\"records\":"<<records<<",\"steps\":"<<steps<<",\"max_operator_error_m3\":"<<maxerr<<",\"end_s\":"<<end<<",\"trace_sha256\":\""<<buf.digest()<<"\"}\n";
 std::cout<<"TRACE_DONE "<<prefix<<" records="<<records<<std::endl;
 }catch(std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
