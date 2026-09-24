"""Export verified Q1 results and figures, including template-compatible CSV."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def csv_file(path, header, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def tex_table(caption, label, columns, header, rows):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering',
        r'\caption{'+caption+'}', r'\label{'+label+'}', r'\begin{tabular}{'+columns+'}',
        r'\toprule', ' & '.join(header)+r' \\', r'\midrule',
        *[' & '.join(map(str,row))+r' \\' for row in rows], r'\bottomrule',
        r'\end{tabular}', r'\end{table}', ''])


def export(out, baseline, sensitivity, priorities, validation):
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'solution.json', baseline)
    write_json(out/'sensitivity.json', sensitivity)
    write_json(out/'priority_comparison.json', priorities)
    write_json(out/'validation.json', validation)
    trips, m = baseline['trips'], baseline['metrics']
    csv_file(out/'Q1_单点组批.csv', ['架次编号','服务区编号','机型编号','货箱编号列表',
        '总质量（kg）','总体积（m³）','往返时间（s）','架次能耗（kWh）','返航SOC（%）'],
        [[t['id'],t['node'],t['type'],';'.join(t['box_ids']),t['mass_kg'],t['volume_m3'],
          t['duration_s'],t['energy_kwh'],100*t['return_soc']] for t in trips])
    csv_file(out/'Q1_逐箱归属.csv', ['货箱编号','服务区编号','架次编号','机型编号'],
             [[b,t['node'],t['id'],t['type']] for t in trips for b in t['box_ids']])
    csv_file(out/'Q1_安全载荷.csv', ['返航余量','服务区编号','机型编号','最大安全载荷（kg）'],
             [[r['reserve'],c['node'],c['type'],c['safe_payload_kg']]
              for r in sensitivity for c in r['safe_payloads']])
    csv_file(out/'Q1_余量敏感性.csv', ['返航余量','最优架次数','能耗（kWh）','累计作业时间（s）','最低返航SOC'],
             [[r['reserve'],r['metrics']['sorties'],r['metrics']['energy_kwh'],
               r['metrics']['cumulative_time_s'],r['metrics']['min_return_soc']] for r in sensitivity])
    names = {'sorties_first':'架次优先','energy_first':'能耗优先','time_first':'时间优先'}
    csv_file(out/'Q1_目标优先级对照.csv', ['优先级','架次数','能耗（kWh）','累计作业时间（s）'],
             [[names[r['priority']],r['metrics']['sorties'],r['metrics']['energy_kwh'],
               r['metrics']['cumulative_time_s']] for r in priorities])
    csv_file(out/'Q1_航线地形.csv', ['服务区编号','单程距离（m）','最高DEM（m）','巡航海拔（m）','去程爬升（m）','返程爬升（m）'],
             [[a['geometry'][k] for k in ('node','distance_m','dem_max_m','cruise_m','out_up_m','back_up_m')]
              for a in baseline['areas']])
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
                         'font.size':10,'pdf.fonttype':42,'figure.dpi':150})

    def save(fig, name):
        fig.savefig(out/f'{name}.png', dpi=170, bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    nodes = [a['geometry']['node'] for a in baseline['areas']]
    fig, ax = plt.subplots(figsize=(10,3.6), layout='constrained')
    for tid, marker in zip('ABC', ['o','s','^']):
        ax.plot(nodes,[c['safe_payload_kg'] for c in baseline['safe_payloads'] if c['type']==tid],
                marker=marker,label=f'{tid}型',linewidth=1.3)
    ax.set(ylabel='最大安全载荷 / kg', xlabel='服务区', ylim=(0,85))
    ax.legend(ncol=3,loc='lower right'); ax.grid(alpha=.2)
    save(fig,'01_safe_payload')
    fig, ax = plt.subplots(figsize=(10,3.6), layout='constrained')
    x = np.arange(len(trips))
    ax.bar(x, [100*t['return_soc'] for t in trips],color=['#168a83' if t['type']=='B' else '#db8735' for t in trips])
    ax.axhline(20,color='#b53030',ls='--',label='20%最低余量')
    ax.set_xticks(x,[f"{t['node']}\n{t['type']}" for t in trips],fontsize=8)
    ax.set(ylabel='返航 SOC / %',xlabel='服务区与机型',ylim=(0,80)); ax.legend(); ax.grid(axis='y',alpha=.2)
    save(fig,'02_return_soc')
    fig, axes = plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    rho = [r['reserve']*100 for r in sensitivity]
    for ax, key, title, color in zip(axes,['sorties','energy_kwh'],['最优架次数','总能耗 / kWh'],['#168a83','#db8735']):
        ax.plot(rho,[r['metrics'][key] for r in sensitivity],'o-',color=color)
        ax.set(xlabel='要求返航余量 / %',ylabel=title,xticks=rho);ax.grid(alpha=.2)
    axes[0].set_yticks([18,19,20]);save(fig,'03_reserve_sensitivity')

    base_rows = [['交付箱数／质量',f"{m['boxes']}箱／{m['mass_kg']:.0f} kg"],
                 ['运输架次数',m['sorties']],['A／B／C型架次','0／9／9'],
                 ['总能耗',f"{m['energy_kwh']:.6f} kWh"],
                 ['累计作业时间',f"{m['cumulative_time_s']:.3f} s（{m['cumulative_time_s']/3600:.3f} h）"],
                 ['最低返航SOC',f"{m['min_return_soc']*100:.4f}\\%"]]
    (out/'table_baseline.tex').write_text(tex_table('第一问基准方案指标','tab:q1:baseline','lr',['指标','结果'],base_rows),encoding='utf-8',newline='\n')
    area_rows=[]
    for node in nodes:
        ts=[t for t in trips if t['node']==node]
        area_rows.append([node,'、'.join(t['type'] for t in ts),'、'.join(f"{t['mass_kg']:.0f}" for t in ts),
                          f"{sum(t['energy_kwh'] for t in ts):.4f}"])
    (out/'table_batches.tex').write_text(tex_table('各服务区组批与能耗','tab:q1:batches','lccr',
        ['服务区','各架次机型','各架次载重/kg','合计能耗/kWh'],area_rows),encoding='utf-8',newline='\n')
    cap_rows=[[node]+[f"{next(c['safe_payload_kg'] for c in baseline['safe_payloads'] if c['node']==node and c['type']==tid):.3f}" for tid in 'ABC'] for node in nodes]
    (out/'table_capacity.tex').write_text(tex_table('20\\%返航余量下各机型最大安全载荷（kg）','tab:q1:capacity','lrrr',
        ['服务区','A型','B型','C型'],cap_rows),encoding='utf-8',newline='\n')
    sense_rows=[[f"{100*r['reserve']:.0f}\\%",r['metrics']['sorties'],f"{r['metrics']['energy_kwh']:.4f}",
                 f"{r['metrics']['cumulative_time_s']:.2f}",f"{100*r['metrics']['min_return_soc']:.2f}\\%"] for r in sensitivity]
    (out/'table_sensitivity.tex').write_text(tex_table('返航余量敏感性','tab:q1:sensitivity','rrrrr',
        ['要求余量','架次','能耗/kWh','累计时间/s','最低SOC'],sense_rows),encoding='utf-8',newline='\n')
    priority_rows=[[names[r['priority']],r['metrics']['sorties'],f"{r['metrics']['energy_kwh']:.6f}",
                    f"{r['metrics']['cumulative_time_s']:.3f}"] for r in priorities]
    (out/'table_priorities.tex').write_text(tex_table('目标优先级的对照结果','tab:q1:priorities','lrrr',
        ['首要目标','架次','能耗/kWh','累计时间/s'],priority_rows),encoding='utf-8',newline='\n')
    best_e=next(r['metrics'] for r in priorities if r['priority']=='energy_first')
    text=f'''# 第一问：单点往返安全运力与精确组批

采用原始节点、货箱、机型与30米DEM，按“架次数、能耗、累计作业时间”词典序全量枚举与动态规划。
物理计算统一使用WGS84局部米制坐标、沿线全部相交DEM像元最高值、9.80665 m/s²重力加速度。
仅服务单区，返程空载；不施加机队、电池周转、送达时限与通信限制。

## 基准结果

80箱、758 kg、2.011 m³全部且仅配送一次；{m['sorties']}架次（A0、B9、C9）。
总能耗 **{m['energy_kwh']:.6f} kWh**；累计作业时间 **{m['cumulative_time_s']:.3f} s**，约 **{m['cumulative_time_s']/3600:.3f} h**。
最低返航SOC **{100*m['min_return_soc']:.4f}%**，高于20%要求。
累计时间是各架次之和，不表示多机并行完工时间。
水平飞行能耗{m['horizontal_kwh']:.6f} kWh，爬升能耗{m['climb_kwh']:.6f} kWh。

## 节能权衡

同一物理模型下，将能耗放在首位，得到{best_e['sorties']}架次、{best_e['energy_kwh']:.6f} kWh。
相对基准只节约{m['energy_kwh']-best_e['energy_kwh']:.6f} kWh（{100*(m['energy_kwh']-best_e['energy_kwh'])/m['energy_kwh']:.4f}%），
却增加1架次及{(best_e['cumulative_time_s']-m['cumulative_time_s'])/60:.3f}分钟累计作业时间。
因此18架次方案接近该模型的最低能耗，并非能耗明显偏高；“最低能耗”与“最少架次下最低能耗”须区别。

## 参考资料与计算口径

参考正文给出59.036497 kWh；本次统一口径比它增加{m['energy_kwh']-59.036497:.6f} kWh，约{100*(m['energy_kwh']/59.036497-1):.4f}%。
参考资料以约15米间隔采样地形并使用不同的平面距离和重力常数。
本实现按原题检查航段穿过的全部DEM像元，避免定距采样遗漏短暂穿过的像元；采用实际重算结果，不沿用参考数字。
没有取得参考资料宣称的独立MATLAB实现，本报告不重复该交叉验证声明。

## 敏感性与完整输出

| 要求余量 | 架次 | 能耗/kWh | 累计时间/s | 最低SOC |
|---:|---:|---:|---:|---:|
'''
    text+='\n'.join(f"| {100*r['reserve']:.0f}% | {r['metrics']['sorties']} | {r['metrics']['energy_kwh']:.6f} | {r['metrics']['cumulative_time_s']:.3f} | {100*r['metrics']['min_return_soc']:.4f}% |" for r in sensitivity)
    text+='''

![安全载荷](01_safe_payload.png)
![逐架次返航SOC](02_return_soc.png)
![余量敏感性](03_reserve_sensitivity.png)

`Q1_单点组批.csv`保留原题模板9列；逐箱归属、45组基准安全载荷及全部余量情景载荷、航线地形、优先级对照另列CSV。
`solution.json`保存全部候选模式、计数状态需求、选中模式和箱号；其他两份情景JSON保存完整替代方案。

## 核验与最优性范围

独立验证器用包围矩形内的线段—像元相交计算重建最高地形，重算货箱、能耗、时间、SOC和安全载荷。
此外独立枚举计数组批，再用HiGHS整数规划按三层目标逐层交叉验证各区DP目标值。
所有运行情景都通过；详见validation.json和manifest.json。DP无随机数、候选截断或状态截断。
最优性适用于声明的单点直线往返、确定性物理模型；整数计数精确，物理量为双精度浮点，交叉核验有数值容差。
有限DEM不代表亚像元真实障碍物；未建模的风场、悬停和充电损失不包含在本次能耗内。
仅测试10%、15%、20%、25%、30%，不将25%解释为精确换批阈值。

复现：在仓库根目录执行 `Group/Lcn/.venv/python.exe -B Group/Lcn/Code/run_q1.py`。
只读复核：追加 `--verify-only`。单元测试：在Code目录执行 `../.venv/python.exe -B -m unittest q1.test_q1 -v`。
'''
    (out/'report.md').write_text(text,encoding='utf-8',newline='\n')
