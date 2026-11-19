# -*- coding: utf-8 -*-
"""
Рисует ОДИН или несколько циклов (.shp) целиком: все меняющиеся аналоговые сигналы по подсистемам
+ временная диаграмма дискретных сигналов. Несколько файлов накладываются друг на друга.

    python plot_cycle.py OP1_20260928_000825.shp                 -> OP1_20260928_000825.html
    python plot_cycle.py a.shp b.shp c.shp -o compare.html       -> сравнение циклов
"""
import argparse, os, re, sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shpreader import read_shp

SKIP = re.compile(r'spare|^off$|^default$|^$', re.I)
GROUPS = [  # (заголовок, регулярка по имени канала)
    ('Ползун: позиция и скорость', r'^slide:(encoder|average|relative|postion|\s*speed)|^slide: speed'),
    ('Ползун: усилия', r'force|^slide:.*pressure|^slide:(head|ring)'),
    ('Ползун: клапаны (задание / золотник)', r'^slide.*\[V\]|^slide: CMD'),
    ('Ползун: регулятор скорости', r'^slide:Speed|^Slide:Speed|^slide:Pump|^Slide:'),
    ('Гидростанция: давления', r'^hyd:(high|low)|accu|DieCl|CLP_'),
    ('Гидростанция: температуры и уровни', r'^hyd:(temp|level|service)'),
    ('Привод насосов: дренаж и обороты', r'^Oildr'),
    ('Насосы: угол наклона (задание/факт)', r'^Pump .*Pivoting'),
    ('Насосы: клапан (задание/факт)', r'^Pump .*Valve'),
    ('Насосы: давление', r'^Pump .*actPressure|^PFG'),
    ('Нижний выталкиватель (BEj)', r'^bed ejector|^BEj'),
    ('Верхний выталкиватель / подушка (SCu)', r'^slide ejector|^SCu'),
    ('Центрирующие цилиндры CCyl', r'^CCyl'),
    ('Нижние цилиндры LCyl', r'^LCyl'),
    ('Подвижный стол MB', r'^MB'),
    ('Прочее', r'.'),
]
PAL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('-o', '--out')
    ap.add_argument('--step', type=int, default=4, help='прореживание: 1 = все точки (10 мс), 4 = 40 мс')
    a = ap.parse_args()
    out = a.out or os.path.splitext(os.path.basename(a.files[0]))[0] + '.html'
    runs = [(os.path.basename(f), read_shp(f)) for f in a.files]
    multi = len(runs) > 1
    r0 = runs[0][1]
    names = r0['ana_names']
    A = r0['ana']
    varying = [j for j, n in enumerate(names) if not SKIP.search(n) and np.nanstd(A[:, j]) > 0]
    if multi:  # канал интересен, если меняется хоть в одном файле
        varying = [j for j, n in enumerate(names) if not SKIP.search(n) and any(np.nanstd(r['ana'][:, j]) > 0 for _, r in runs)]
    used, html, first = set(), [], True

    def emit(fig):
        nonlocal first
        h = pio.to_html(fig, include_plotlyjs='inline' if first else False, full_html=False, config={'displaylogo': False})
        first = False
        return h

    # обзор цикла
    key = [('slide:average val.encoder [mm]', 'Позиция ползуна, мм'), ('slide:total force [kN]', 'Усилие, кН'),
           ('hyd:high pressure [bar]', 'Высокое давление, бар'), ('slide:deviation left-right [mm]', 'Перекос Л-П, мм')]
    ix = {n: j for j, n in reversed(list(enumerate(names)))}
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[k[1] for k in key], vertical_spacing=0.05)
    for k, (fn, r) in enumerate(runs):
        t = (np.arange(r['ana'].shape[0])[::a.step] * 0.01).astype(np.float32)
        for i, (nm, _) in enumerate(key):
            if nm in ix:
                fig.add_scatter(x=t, y=r['ana'][::a.step, ix[nm]].astype(np.float32), mode='lines', line=dict(color=PAL[k % len(PAL)], width=1.4),
                                name=fn, legendgroup=fn, showlegend=(i == 0), row=i + 1, col=1)
    fig.update_xaxes(title_text='время от начала записи, с', row=4, col=1)
    fig.update_layout(height=820, margin=dict(l=50, r=10, t=40, b=40))
    html.append('<h2>Обзор цикла</h2>' + emit(fig))

    # группы
    for title, rx in GROUPS:
        chans = [j for j in varying if j not in used and re.search(rx, names[j])]
        if not chans:
            continue
        used.update(chans)
        fig = go.Figure()
        for c_i, j in enumerate(chans):
            for k, (fn, r) in enumerate(runs):
                t = (np.arange(r['ana'].shape[0])[::a.step] * 0.01).astype(np.float32)
                col = PAL[(k if multi else c_i) % len(PAL)]
                fig.add_scatter(x=t, y=r['ana'][::a.step, j].astype(np.float32), mode='lines', line=dict(color=col, width=1.2, dash=None if k == 0 or not multi else 'dot'),
                                name=names[j] + ((' · ' + fn) if multi else ''), legendgroup=names[j],
                                visible=True if c_i < 8 else 'legendonly')
        fig.update_layout(title=title + ('  (%d сигн.; часть скрыта — включайте в легенде)' % len(chans) if len(chans) > 8 else ''),
                          height=380, margin=dict(l=50, r=10, t=50, b=40), legend=dict(font=dict(size=10)), xaxis_title='с')
        html.append(emit(fig))

    # дискретные
    D = r0['dig']; dn = r0['dig_names']
    act = [j for j, n in enumerate(dn) if j > 0 and not SKIP.search(n) and D[:, j].any()]
    if act:
        t = (np.arange(D.shape[0])[::a.step * 2] * 0.01).astype(np.float32)
        z = D[::a.step * 2, act].T.astype(np.int8)
        fig = go.Figure(go.Heatmap(z=z, x=t, y=['%s' % dn[j] for j in act], colorscale=[[0, '#eef2f7'], [1, '#1f77b4']], showscale=False,
                                   hovertemplate='%{y}<br>t=%{x:.2f} c<br>%{z}<extra></extra>'))
        fig.update_layout(title='Дискретные сигналы (синее = включено), файл %s' % runs[0][0], height=max(300, 16 * len(act) + 80),
                          margin=dict(l=260, r=10, t=50, b=40), xaxis_title='с', yaxis=dict(autorange='reversed', tickfont=dict(size=9)))
        html.append('<h2>Последовательность управления</h2>' + emit(fig))

    head = '<h1>Цикл пресса: %s</h1><p style="color:#666">%d аналоговых сигналов меняются, %d дискретных активны. Длительность %.1f с, шаг 10 мс. Клик по легенде — скрыть/показать линию, двойной клик — оставить одну.</p>' % (
        ', '.join(fn for fn, _ in runs), len(varying), len(act), A.shape[0] * 0.01)
    page = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Цикл пресса</title>' \
           '<style>body{font:14px system-ui,Segoe UI,sans-serif;max-width:1300px;margin:0 auto;padding:16px;background:#fff;color:#1d1d1f}</style></head><body>' + head + '\n'.join(html) + '</body></html>'
    open(out, 'w', encoding='utf-8').write(page)
    print('Готово:', out)


if __name__ == '__main__':
    main()
