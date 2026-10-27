"""Strict adapter for the pinned solver's ten-key echo and hashed pump extension.
Never edits solver receipts or inputs. Numerical gates remain in the frozen parent.
"""
from pathlib import Path
import json,hashlib
CORE=('schema_version','run_id','parent_id','family_id','split','forcing_sha256','geometry_sha256','source_commit','patch_sha256','solver_digest')
EXTRA='treatment_sha256'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_identity_bytes(staged,echo,identity,pump_sha256):
 if set(identity)!=set(CORE+(EXTRA,)):return False
 full=('\n'.join(k+'='+str(identity[k]) for k in CORE+(EXTRA,))+'\n').encode()
 expected=('\n'.join(k+'='+str(identity[k]) for k in CORE)+'\n').encode()
 return staged==full and echo==expected and identity[EXTRA]==pump_sha256

def identity_receipt_check(folder,run):
 try:
  identity=run['identity'];pump=sha(folder/'udt_pumps.dat')
  return validate_identity_bytes((folder/'udt_run_identity.txt').read_bytes(),(folder/'udt_diagnostic_identity.txt').read_bytes(),identity,pump) and json.loads((folder/'udt_run_identity.json').read_text())==identity and run['input_sha256']['udt_pumps.dat']==pump
 except (OSError,ValueError,KeyError):return False

def selftest():
 identity={k:'value_'+k for k in CORE};identity[EXTRA]='a'*64
 full=('\n'.join(k+'='+identity[k] for k in CORE+(EXTRA,))+'\n').encode()
 echo=('\n'.join(k+'='+identity[k] for k in CORE)+'\n').encode()
 check={}
 check['exact_schema_valid']=validate_identity_bytes(full,echo,identity,'a'*64)
 for k in CORE:
  bad=echo.replace((k+'='+identity[k]).encode(),(k+'=tampered').encode(),1)
  check['reject_echo_tamper_'+k]=not validate_identity_bytes(full,bad,identity,'a'*64)
 check['reject_pump_change']=not validate_identity_bytes(full,echo,identity,'b'*64)
 check['reject_missing_treatment']=not validate_identity_bytes(echo,echo,identity,'a'*64)
 check['reject_extra_staged_field']=not validate_identity_bytes(full+b'unknown=x\n',echo,identity,'a'*64)
 check['reject_extra_echo_field']=not validate_identity_bytes(full,echo+b'unknown=x\n',identity,'a'*64)
 check['reject_unknown_manifest_key']=not validate_identity_bytes(full,echo,{**identity,'unknown':'x'},'a'*64)
 check['reject_reordered_echo']=not validate_identity_bytes(full,b'\n'.join(reversed(echo.rstrip(b'\n').split(b'\n')))+b'\n',identity,'a'*64)
 assert all(check.values()),check
 return {'passed':True,'checks':check,'no_numerical_tolerance_changes':True}
if __name__=='__main__':print(json.dumps(selftest(),indent=2))

