"""Build/link readiness report; never executes a solver or changes old labels."""
from pathlib import Path
import os,sys,json,re,argparse,html,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import OUT,REPORT,COMMIT,TAG,sha,dump

def update_sop(root):
 path=root/'docs/AnaSOP.md';folder=root/REPORT;backup=folder/'previous-AnaSOP.md'
 if not backup.exists():backup.write_bytes(path.read_bytes())
 before=path.read_text();parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M)
 addition4='''The complete CPU solver and bundled Fortran NetCDF dependency have now been built and linked from the pinned, verified source plus the frozen correction. The full production library resolves its solver and diagnostic entry points without invoking them. Isolated tests use the actual production rainfall reader and Fortran NetCDF interface for the same 50 forcing inputs, with unchanged decoded seconds and corresponding rainfall integrals; production-linked bookkeeping fixtures also pass. Source, patch, build flags, dependency versions, binary checksums and the local image manifest are bound together. This is build, link and isolated-interface qualification. No solver initialization, hydraulic update, new event response or training has occurred; hydraulic end-to-end qualification and reference release remain pending. An upstream astronomical-tide array-bound compiler warning is retained in the build record; astronomical-tide execution is not qualified by these checks.

'''
 addition6='''The complete versioned CPU image now passes dependency, linking, dynamic-load and isolated production-interface checks. Keep this build-level qualification distinct from numerical event validation: no hydraulic update or new event response has been generated. Bind all future descendants to the recorded source and immutable image identity while preserving the original family partition and reference policy. Separately authorized end-to-end hydraulic checks must still establish actual-grid clock, source, boundary and storage diagnostics before admitting labels. The planned Reference uncertainty and Scenario coverage and split audit outputs retain that distinction.

'''
 for i,s in enumerate(parts):
  if s.startswith('## 4. '):
   s=s.replace('The complete operational solver image has not been rebuilt or qualified.', 'The subsequent complete CPU build closes the build and link gap; hydraulic end-to-end qualification remains pending.')
   s=s.replace('Source integration and isolated bookkeeping are tested; full-solver operational clock and volume qualification remains outstanding.', 'Source integration, complete CPU build and link, and isolated production bookkeeping are tested; hydraulic end-to-end clock and volume qualification remains outstanding.')
   if addition4 not in s:s=s.rstrip()+'\n\n'+addition4
   parts[i]=s
  elif s.startswith('## 6. '):
   s=s.replace('Complete operational build and dependency qualification, then separately authorized hydraulic end-to-end checks, are still required before constructing admissible responses.', 'Complete operational build and dependency qualification are now checked; separately authorized hydraulic end-to-end checks are still required before constructing admissible responses.')
   marker='### Connection to active measures'
   if addition6 not in s:
    assert marker in s;s=s.replace(marker,addition6+marker)
   parts[i]=s
  elif s.startswith('## 7. '):
   s=s.replace('operational image: held; response library: absent', 'CPU image build/link: verified; hydraulic end-to-end: held; response library: absent')
   s=s.replace('qualify the complete operational transport and diagnostic build before separately authorized reference runs', 'retain the verified CPU build identity and qualify hydraulic end-to-end outputs in separately authorized checks before reference release')
   s=s.replace('All 58 descendants remain unrun and held pending complete operational qualification;', 'The complete CPU image now passes build, dependency, linking and isolated production-interface checks. All 58 descendants remain unrun and held pending hydraulic end-to-end qualification;')
   parts[i]=s
 after=''.join(parts);path.write_text(after)
 old=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M)
 changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(old,parts)if a!=b and a.startswith('## ')}
 assert changed=={4,6,7};assert all(a==b for a,b in zip(old,parts)if not a.startswith(('## 4. ','## 6. ','## 7. ')))
 dump(folder/'anasop-update.json',{'skills':{'4':'data-preprocessing','6':'estimation-framework-planning','7':'estimation-framework-planning'},'sections_changed':sorted(changed),'evidence':'complete CPU build, production Fortran NetCDF link, loader and isolated input/ledger receipts','user_scope':'build/link only; unchanged forcing lineage and splits; no training or new hydraulic runs','old_sha256':sha(backup),'current_sha256':sha(path),'other_sections_unchanged':True,'new_final_variables':0,'new_formulas':0,'section8_outputs_still_planned':True})

def report(root):
 folder=root/REPORT;out=root/OUT;summary=json.loads((folder/'summary.json').read_text());binding=json.loads((out/'solver_image_binding.json').read_text());sources=json.loads((out/'source_binding.json').read_text())
 assert summary['all_checks_passed']
 digest=binding['oci_manifest_digest'];local_id=binding['local_image_id'];nc=binding['Fortran_NetCDF']
 warnings=[];site=''
 for line in(out/'isolated_probe_outputs/build-receipts/make.log').read_text().splitlines():
  if re.match(r'^\S+\.(?:f90|F90|h):\d+:',line):site=line
  if line.startswith('Warning:'):
   kind='upstream_legacy_extension'
   if 'out of bounds'in line:kind='upstream_astronomical_tide_table_bounds'
   elif 'Type mismatch'in line:kind='bundled_NetCDF_legacy_argument_mismatch'
   elif 'redefined'in line:kind='bundled_NetCDF_macro_redefinition'
   warnings.append({'source_site':site,'message':line,'category':kind,'new_patch_warning':False})
 dump(folder/'compiler-warnings.json',{'warnings':warnings,'count':len(warnings),'upstream_sources_unchanged':True,'astronomical_tide_execution_qualified':False,'hydrodynamic_execution_not_performed':True})

 text=f'''# 完整 CPU 求解器构建与依赖核验\n\n结果：完整 Linux AMD64 CPU 求解器及 Fortran NetCDF 已构建并通过链接、动态装载及隔离生产接口检查。没有运行新的水动力模拟，没有训练。\n\n## 版本绑定\n\n- 固定上游提交：`{COMMIT}`。核验 {sources['verified_source_blobs']} 个源文件的 Git blob。\n- 镜像标签：`{TAG}`。\n- OCI index digest：`{digest}`。\n- Docker local image identity：`{local_id}`。\n- 秒级读取／同一步预算补丁 SHA256：`{binding['interface_patch_sha256']}`。\n- `solver_image_binding.json` 绑定源码、补丁、镜像 manifest/config、本地 ID、平台及运行时二进制校验值。\n- 本地离线归档 `solver_image.tar` 已保存；全部 OCI blob 校验通过，不曾推送外部仓库。\n- `source_binding.json` 绑定完整源文件清单；`build_integration.patch` 仅登记模块编译次序。编译使用 GNU/OpenMP，明确关闭 OpenACC，关闭浮点乘加收缩；完整参数、工具链和包版本保存在构建收据中。\n\n## 验证边界\n\n{summary['checks']} 项核验通过。完整上游自动构建链编译自带 Fortran NetCDF 和全部求解器模块，保留真正的 Fortran NetCDF `.mod` 与库，不使用上一步测试签名桥接器或依赖 shell。运行时 `ldd` 未发现缺失库；`RTLD_NOW` 可完整装载新共享库并解析十个指定入口，仅解析地址，不调用求解器初始化、推进或终止。\n\n真实 Fortran NetCDF 双精度读写探针通过。生产库中的实际降雨读取子程序逐份读取原 50 个时间序列的小空间 fixture，时刻和相应雨量时间积分偏差均为 0；雨量 real32 值、有效零和翻转保持一致。不是重新生成 50 份全域降雨，也不是 50 次暴雨模拟。\n\n生产库中的同一步诊断累计器通过预设账本 fixture（包含正反交换、10m/5m 边界宽度、干输入、负体积、配置／时钟／缺项拒收）；连续方程从未执行。该检验验证真实依赖和组件，不验证真实地形、洪水过程或全域数值误差。\n\n58 个原有未来副本已绑定新镜像 identity，原曲线、父事件、family、划分、旧结果及准入政策均保持不变。原默认 loader 和参考准入仍保留 hold。\n\n## 编译警告保留\n\n构建日志保留 9 条上游警告：NetCDF 宏与旧接口参数兼容提示、原有 Fortran 语法扩展，以及天文潮表一处静态数组越界警告。新加入的两个模块没有产生这些警告。未修改冻结上游文件来消除警告；天文潮执行路径未被本次测试覆盖，启用前必须单独核验。细目见 `compiler-warnings.json`。当前结论限于构建／链接与已执行的接口检查。\n\n## 可复核文件\n\n- processed 目录：`{OUT}`。\n- `context/Dockerfile`、`context/build-and-check.sh`：完整可复现构建配方。\n- `docker-build.log`：完整生产镜像构建输出；`isolated_probe_outputs/build-receipts`：编译、配置、安装、包清单、链接与符号收据。\n- `50_production_reader_checks.csv`、`production_budget_fixture_checks.csv`：实际生产库隔离检查。\n- `run_bindings.json`：58 个副本的版本绑定，不构成执行或参考放行。\n- AnaSOP 第4、6、7节更新构建／链接状态；其余章节与候选变量状态不变。\n\n重建入口：先运行 `src/preprocessing/prepare_sfincs_cpu_build.py` 准备新版本目录，再用 Docker build 构建 runtime target 并保存 metadata。冻结版本拒绝覆盖。隔离核验脚本为 `src/analyses/verify_sfincs_cpu_image.py`。所有测试容器均指定测试或 shell entrypoint，网络禁用；从未使用镜像默认的 sfincs 命令。\n\n下一道验证是单独授权的真实网格水动力端到端检查，重点核对真实时钟、预算、输出对接和边界；当前 build/link 通过不能自动放行新情景或训练。\n'''
 (folder/'README.md').write_text(text)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整 CPU 求解器构建核验</title><style>body{font:16px/1.75 system-ui,sans-serif;background:#eff3f5;color:#203746;margin:0}main{max-width:1040px;margin:auto;padding:28px}h1{font-size:28px;line-height:1.4}h2{font-size:21px}section{background:#fff;border:1px solid #dbe3e7;border-radius:12px;padding:23px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:#eaf3f4;border-radius:9px;padding:16px}.card strong{display:block;font-size:25px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dfe7ed;padding:10px}code{overflow-wrap:anywhere;font-size:13px}a{color:#126979}.limit{border-left:4px solid #b98542;padding-left:13px}.scroll{overflow:auto}@media(max-width:650px){main{padding:12px}.cards{grid-template-columns:1fr}section{padding:16px}}</style><main><p>Mili · 三个区整体模型 · 完整构建与依赖核验</p><h1>完整 CPU 镜像已构建，真实 Fortran NetCDF 与链接检查通过</h1><p>保留秒级降雨修正和同一步预算诊断；没有新水动力模拟或训练。</p><section><div class="cards"><div class="card"><strong>__CHECKS__ 项</strong>构建与生产接口检查通过</div><div class="card"><strong>50 / 50</strong>实际生产降雨读取通过</div><div class="card"><strong>58 个</strong>原副本绑定镜像，仍未运行</div></div></section><section><h2>版本已绑定</h2><table><tr><th>组件</th><th>固定值</th></tr><tr><td>上游提交</td><td><code>__COMMIT__</code></td></tr><tr><td>镜像</td><td><code>__TAG__</code></td></tr><tr><td>OCI index</td><td><code>__DIGEST__</code></td></tr><tr><td>平台</td><td>Linux AMD64 · GNU/OpenMP CPU</td></tr><tr><td>源码校验</td><td>424 个源文件的 Git blob；冻结接口补丁保持不变</td></tr></table><p>完整绑定还保留编译参数、Fortran NetCDF 版本、动态库依赖和运行时二进制 SHA256。</p></section><section><h2>这次补上了什么</h2><div class="scroll"><table><tr><th>检查</th><th>实际完成</th></tr><tr><td>完整编译和链接</td><td>全部正式求解器模块与自带 Fortran NetCDF；无依赖 shell</td></tr><tr><td>运行时依赖</td><td>ldd 无缺失库；RTLD_NOW 成功装载真实共享库并解析十个指定入口</td></tr><tr><td>真实 Fortran NetCDF</td><td>实际 .mod、库链接及双精度读写探针通过，无测试桥接器</td></tr><tr><td>50 份原情景时序</td><td>生产降雨读取子程序逐份通过；节点及时间转换雨量积分偏差均为零</td></tr><tr><td>同一步诊断</td><td>实际生产库的预设账本测试、反向交换、负体积和拒收检查通过</td></tr></table></div><p class="limit">动态加载只解析入口地址；降雨测试只读取小空间 fixture；账本测试只累计预设项。没有调用求解器初始化、更新或连续方程。</p></section><section><h2>仍未验证的部分</h2><p>编译日志保留上游天文潮表的一处静态数组越界警告，未修改上游代码；天文潮执行路径未核验。<a href="compiler-warnings.json">全部编译警告记录</a>已保存。</p><p>真实网格上的水动力端到端时钟、预算和输出对接仍需单独授权核验。完整构建通过不等于洪水模拟已通过、物理准确或参考标签已放行。原曲线、父事件谱系、划分、旧结果及准入规则均保持不变。</p><p>下一步只需针对真实网格验证运行链路；当前没有新增情景响应、训练或封存响应读取。</p></section><section><h2>核验资料</h2><p><a href="README.md">复现说明</a> · <a href="validation.json">检查明细</a> · <a href="50_production_reader_checks.csv">50 份生产读取记录</a> · <a href="production_budget_fixture_checks.csv">诊断组件检查</a> · <a href="../../../processed/three_ward_cpu_solver_build_v1/solver_image_binding.json">源码—镜像绑定</a> · <a href="../../../processed/three_ward_cpu_solver_build_v1/context/Dockerfile">Dockerfile</a> · <a href="docker-build.log">构建日志</a> · <a href="../../../processed/three_ward_cpu_solver_build_v1/oci_archive_receipt.json">离线镜像校验</a></p><p>AnaSOP 第4、6、7节已同步；研究变量仍为候选，主动措施阶段安排不变。</p></section></main></html>'''
 for k,v in {'__CHECKS__':str(summary['checks']),'__COMMIT__':COMMIT,'__TAG__':TAG,'__DIGEST__':digest}.items():page=page.replace(k,html.escape(v))
 (folder/'review.html').write_text(page)
 dump(out/'readiness.json',{'build_link_qualified':True,'real_Fortran_NetCDF_qualified':True,'production_reader_components_qualified':True,'hydrodynamic_end_to_end_qualified':False,'pending_descendants_unrun':58,'execution_release':False,'reference_release':False,'training':0,'remaining':['separately authorized real-grid hydraulic end-to-end clock and budget checks','upstream astronomical-tide array-bound warning must be assessed before enabling astronomical tidal inputs','seven-group optional aggregation','reference admission and response library construction']})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--update-sop',action='store_true');a=p.parse_args()
 if(ROOT/OUT/'freeze.json').exists():raise FileExistsError('Frozen delivery')
 if a.update_sop:update_sop(ROOT)
 report(ROOT)
