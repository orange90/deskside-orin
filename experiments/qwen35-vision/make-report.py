from pathlib import Path
import json,html
root=Path('outputs/orin-desk-agent/evidence/57-qwen35-vision');s=json.loads((root/'summary.json').read_text())
names={'sit':'坐姿','stand':'站姿','away':'离开','unknown':'未知'}
cases=[('U01','历史未知 · 11 月 10 日 09:31:53','画面可见站立者；文件时间对应旧未知区间，非同步帧编号。'),('U02','历史未知 · 11 月 11 日 15:00:23','用户已确认当时出门。截图旧标签已遮盖后送入模型。'),('C01','对照 · 明显站立','11 月 11 日照片；重点比较两种提示的结果变化。'),('C02','对照 · 坐在工位','11 月 10 日坐姿照片。'),('C03','遮挡 · 用户确认仍坐着','11 月 11 日 14:34:58；椅背遮住大部分身体。')]
css='*{box-sizing:border-box}body{margin:0;background:#f4f5f0;color:#253c35;font:16px/1.7 system-ui,-apple-system,"PingFang SC"}main{max-width:1500px;margin:auto;padding:30px}h1{font-size:30px;margin:0}.muted{color:#6e7b73;font-size:14px}.stats{display:flex;gap:15px;margin:22px 0}.stat{background:#e4eddd;border-radius:14px;padding:18px 24px;flex:1}.case{margin:22px 0;border:1px solid #e0e7db;border-radius:16px;background:white;overflow:hidden}.case h2{font-size:20px;padding:18px 22px 0;margin:0}.case>p{margin:0;padding:0 22px 16px}.body{display:grid;grid-template-columns:56% 44%}.body img{display:block;width:100%}.results{padding:12px 24px}.label{font-size:24px;font-weight:600}.variant{padding:12px 0;border-bottom:1px solid #edf0e8}.variant:last-child{border:0}.evidence{font-size:14px;color:#647568}a{color:#28634c}blockquote{margin:12px 0;padding-left:15px;border-left:3px solid #b6cba4}.note{padding:18px;background:white;border-radius:12px} @media(max-width:750px){.body{grid-template-columns:1fr}.stats{display:block}.stat{margin-bottom:10px}}'
out=f'<!doctype html><meta charset="utf-8"><title>Qwen3.5 视觉照片实测</title><style>{css}</style><main><h1>Qwen3.5-4B，看懂这些工位照片了吗？</h1><p class="muted">2026-11-12 · AGX Orin 64GB / 30W · 独立 llama.cpp CUDA 服务 · Q4_K_M 文本 + F16 视觉编码器 · 未替换线上模型</p><div class="stats">'
for profile,title in [('explained','判断 + 简短依据'),('compact','精简提示 / 仅状态')]:
 d=s[profile];out+=f'<div class="stat"><b>{title}</b><div>中位 {d["median_s"]:.2f} 秒 · P95 {d["p95_s"]:.2f} 秒</div><span class="muted">5 张 × 3 次，提示缓存命中均为 0</span></div>'
out+='</div><div class="note">完整回答版能判断三张明显坐站图，两张遮挡 / 空场仍未知。精简版把 C01 的明显站立误判为坐姿。原始输出保留，未挑选有利结果；重复同一照片不是独立准确率样本。</div>'
for case,title,desc in cases:
 a=s['explained']['by_case'][case];b=s['compact']['by_case'][case]
 out+=f'<section class="case" id="{case}"><h2>{case} · {title}</h2><p class="muted">{desc}</p><div class="body"><a href="inputs/{case}.png"><img src="inputs/{case}.png"></a><div class="results">'
 for r,t in [(a,'判断 + 依据'),(b,'精简提示 / 仅状态')]:
  labels=' / '.join(names[k]+f'（{n}/3）' for k,n in r['labels'].items())
  out+=f'<div class="variant"><div class="muted">{t}</div><div class="label">{labels}</div><div>中位 {r["median_s"]:.2f} 秒</div><div class="muted">范围 {r["min_s"]:.2f}–{r["max_s"]:.2f} 秒</div>'
  if 'evidence' in r['result']:out+=f'<blockquote class="evidence">模型依据：{html.escape(r["result"]["evidence"])}</blockquote>'
  out+='</div>'
 out+='</div></div></section>'
out+='<p class="muted">时间从向本机实验服务发送请求到收到完整结果，包含图片传输、视觉编码、提示处理与生成，不含文件读取和图片遮盖。关闭思考和提示缓存，单请求串行；生产服务继续运行，耗时包含实际资源竞争。该配置不满足有效姿态切换 P95≤3秒的目标。</p></main>'
(root/'Qwen视觉照片对照.html').write_text(out)
print('Local image/result report written')
