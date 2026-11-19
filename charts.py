# -*- coding: utf-8 -*-
"""Дополнительные графики для отчёта analyze.py."""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PAL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#393b79', '#ad494a']

# Физические группы: заголовок, пояснение, список признаков
GROUPS = [
    ('Дренаж корпусов насосов MD1–MD6', 'Главный индикатор износа аксиально-поршневых насосов: рост = внутренние утечки.',
     ['casedrain_md%d_mean' % k for k in range(1, 7)]),
    ('Насосы: ошибка угла наклона под нагрузкой', 'Насколько фактический угол люльки не успевает за заданием. Рост = износ регулятора.',
     ['pump%d%d_pivot_err_active' % (a, b) for a in range(1, 7) for b in (1, 2)]),
    ('Насосы: запаздывание угла (×10 мс)', 'Рост = регулятор стал медленнее.',
     ['pump%d%d_pivot_lag' % (a, b) for a in range(1, 7) for b in (1, 2)]),
    ('Насосы: ошибка клапана управления', 'Рост = загрязнение/износ клапана регулятора.',
     ['pump%d%d_valve_err_rms' % (a, b) for a in range(1, 7) for b in (1, 2)]),
    ('Насосы: макс. давление за цикл, бар', '', ['pump%d%d_press_max' % (a, b) for a in range(1, 7) for b in (1, 2)]),
    ('Сервоклапаны: ошибка задание-золотник (RMS)', 'Рост = загрязнение масла, износ золотника, заедание.',
     ['v12Y102_err_rms', 'v12Y069_21_err_rms', 'v12Y069_22_err_rms', 'v20Y019_err_rms', 'v31Y102_err_rms',
      'vCCylL_Y22_2_err_rms', 'vCCylR_Y22_1_err_rms', 'vLCylL_Y21_2_err_rms', 'vLCylR_Y21_1_err_rms', 'vMB_Y1_err_rms', 'vMB_Y2_err_rms']),
    ('Сервоклапаны: запаздывание (×10 мс)', 'Рост = медленный отклик.',
     ['v12Y102_lag', 'v12Y069_21_lag', 'v12Y069_22_lag', 'v20Y019_lag', 'v31Y102_lag', 'vCCylL_Y22_2_lag', 'vCCylR_Y22_1_lag',
      'vLCylL_Y21_2_lag', 'vLCylR_Y21_1_lag', 'vMB_Y1_lag', 'vMB_Y2_lag']),
    ('Масло: температура, °C', 'Рост при той же нагрузке = охлаждение или внутренние утечки (греют масло).',
     ['temp_bottom_mean', 'temp_upper_mean']),
    ('Масло: уровни баков', 'Медленное падение = внешняя утечка.', ['level_bottom_mean', 'level_upper_mean']),
    ('Высокое давление: минимум и просадка, бар', 'Просадка растёт = насосы/аккумуляторы не держат.', ['hp_min', 'hp_drop', 'hp_mean']),
    ('Аккумуляторы и зажим штампа, бар', 'Падение минимумов = азот/заряд/утечка контура.',
     ['accu_rv1_min', 'accu_rv3_min', 'dieclamp_c1_min', 'dieclamp_c2_min']),
    ('Ползун: перекос лево-право, мм', 'Рост = износ направляющих, разбаланс.', ['lr_dev_absmax_mm', 'lr_dev_press_mean_mm']),
    ('Ползун: времена, с', 'Рост при той же работе = гидравлика теряет производительность.',
     ['press_time_s', 'force_rise_s', 'press_start_s', 'slide_moving_s']),
    ('Ползун: скорости, мм/с', '', ['v_down_max', 'v_up_max']),
    ('Ползун: дрейф в покое наверху, мм/с', 'Ползун «сползает» при закрытых клапанах = утечка через уплотнения/клапаны.', ['slide_drift_top_mm_s']),
    ('Центрирующие/нижние цилиндры: ошибка позиции (RMS)', 'Рост = износ/утечки/клапан.',
     ['CCylLeft_posdev_rms', 'CCylRight_posdev_rms', 'LCylLeft_posdev_rms', 'LCylRight_posdev_rms']),
    ('Выталкиватели и MB', '', ['bej_speed_max', 'sej_speed_max', 'MB_vel_err_rms']),
    ('Частотник привода насосов: ошибка оборотов', 'Рост = двигатель/насос тяжелее крутится.', ['fc_speed_err_rms']),
]

# Каналы для графиков формы цикла: (имя канала, подпись)
CYCLE_CH = [('slide:average val.encoder [mm]', 'Позиция ползуна, мм'),
            ('slide:total force [kN]', 'Усилие, кН'),
            ('hyd:high pressure [bar]', 'Высокое давление, бар'),
            ('slide:deviation left-right [mm]', 'Перекос Л-П, мм'),
            ('slide: speed [mm/s]', 'Скорость ползуна, мм/с'),
            ('Pump 1.1 actPivoting', 'Угол насоса 1.1, %')]


def _short(human, c):
    s = human(c)
    return s.split(':')[0] if len(s) > 40 else s


def _anchor(t):
    return 's%d' % (abs(hash(t)) % 10 ** 8)


def build(ctx, fig_html):
    """Возвращает список (заголовок, html). ctx — словарь с данными из analyze.main()."""
    P, B, G, med, sig, eng, T, Ttop, S, q, freq, human, subsystem, an_rows = (
        ctx[k] for k in ('P', 'B', 'G', 'med', 'sig', 'eng', 'T', 'Ttop', 'S', 'q', 'freq', 'human', 'subsystem', 'an_rows'))
    out = []
    unit = 'день' if freq == 'D' else 'час'
    Gz = (G[eng] - med[eng]) / sig[eng]
    subs = pd.Series({c: subsystem(c) for c in eng})

    # 1. Карта здоровья подсистем ------------------------------------------------
    H = {}
    for sname in subs.unique():
        cols = subs[subs == sname].index
        H[sname] = Gz[cols].abs().apply(lambda r: np.sort(r.dropna().values)[-3:].mean() if r.notna().any() else np.nan, axis=1)
    H = pd.DataFrame(H).T
    if H.shape[1] > 0:
        fig = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index, colorscale=[[0, '#2ca02c'], [0.15, '#ffe680'], [0.4, '#ff7f0e'], [1, '#b30000']],
                                   zmin=0, zmax=max(10, np.nanpercentile(H.values, 98) if np.isfinite(H.values).any() else 10),
                                   colorbar=dict(title='σ'), hovertemplate='%{y}<br>%{x}<br>отклонение %{z:.1f}σ<extra></extra>'))
        fig.update_layout(height=120 + 28 * len(H), margin=dict(l=200, r=10, t=10, b=30))
        hi = H.max(axis=0)
        fig2 = go.Figure(go.Scatter(x=hi.index, y=hi.values, mode='lines+markers', line=dict(color='#d62728')))
        fig2.add_hrect(y0=0, y1=3, fillcolor='#2ca02c', opacity=0.1, line_width=0)
        fig2.add_hrect(y0=3, y1=6, fillcolor='#ff7f0e', opacity=0.1, line_width=0)
        fig2.update_layout(height=260, title='Общий индекс нездоровья пресса (худшая подсистема за %s), σ' % unit, margin=dict(l=40, r=10, t=40, b=30))
        out.append(('Карта здоровья по подсистемам',
                    '<p class="muted">Каждая клетка — насколько подсистема в этот %s отличается от нормы (среднее трёх худших признаков, в σ). '
                    'Зелёный — норма, жёлтый/оранжевый — отклонение, красный — сильное. Так видно, где и когда начинает «болеть».</p>' % unit
                    + fig_html(fig) + fig_html(fig2)))

    # 2. Аномалии по подсистемам во времени ----------------------------------------
    Zl = ctx['Zl']
    flag = (Zl > ctx['z'])
    if flag.values.any():
        bys = flag.T.groupby(subs).any().T
        bys.index = P['t_start']
        cnt = bys.resample(freq).sum()
        fig = go.Figure()
        for i, c in enumerate(cnt.columns):
            if cnt[c].sum() > 0:
                fig.add_bar(x=cnt.index, y=cnt[c], name=c, marker_color=PAL[i % len(PAL)])
        fig.update_layout(barmode='stack', height=320, margin=dict(l=40, r=10, t=10, b=30), legend=dict(orientation='h'))
        out.append(('Аномалии по подсистемам во времени',
                    '<p class="muted">Сколько аномальных циклов за %s, с разбивкой по подсистеме, в которой нашли выброс. Учащение выбросов в одной подсистеме — частый признак приближающегося отказа.</p>' % unit
                    + fig_html(fig)))

    # 3. Группы физических признаков ---------------------------------------------
    parts = []
    for title, note, cols in GROUPS:
        cols = [c for c in cols if c in G.columns and G[c].notna().any()]
        if not cols:
            continue
        fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=('Медиана за %s (реальные значения)' % unit, 'Отклонение от нормы, σ'))
        for i, c in enumerate(cols):
            col = PAL[i % len(PAL)]
            nm = _short(human, c)
            fig.add_scatter(x=G.index, y=G[c], mode='lines+markers', marker=dict(size=4), line=dict(color=col, width=1.6), name=nm, legendgroup=c, row=1, col=1)
            fig.add_scatter(x=Gz.index, y=Gz[c], mode='lines', line=dict(color=col, width=1.4), name=nm, legendgroup=c, showlegend=False, row=1, col=2)
        fig.add_hrect(y0=-3, y1=3, fillcolor='#999', opacity=0.15, line_width=0, row=1, col=2)
        fig.update_layout(height=340, margin=dict(l=40, r=10, t=40, b=30), legend=dict(font=dict(size=10)))
        parts.append('<h3>%s</h3><p class="muted">%s</p>%s' % (title, note, fig_html(fig)))
    if parts:
        out.append(('Подсистемы подробно',
                    '<p class="muted">Слева — реальные значения (можно отключать линии кликом по легенде), справа — те же признаки в σ от нормы; серая полоса ±3σ — норма.</p>' + ''.join(parts)))

    # 4. Было / стало -------------------------------------------------------------
    top = [c for c in Ttop['feature'] if c in P.columns][:9]
    if top:
        tail = P.tail(max(10, int(len(P) * 0.2)))
        nr = int(np.ceil(len(top) / 3))
        fig = make_subplots(rows=nr, cols=3, subplot_titles=[_short(human, c)[:45] for c in top])
        for i, c in enumerate(top):
            r, cc = i // 3 + 1, i % 3 + 1
            fig.add_histogram(x=B[c], name='норма', marker_color='#1f77b4', opacity=0.6, showlegend=(i == 0), histnorm='probability', row=r, col=cc)
            fig.add_histogram(x=tail[c], name='последние 20%', marker_color='#d62728', opacity=0.6, showlegend=(i == 0), histnorm='probability', row=r, col=cc)
        fig.update_layout(barmode='overlay', height=260 * nr, margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation='h'))
        out.append(('Было / стало', '<p class="muted">Распределение признаков с самым сильным трендом: синим — базовый период, красным — последние 20% циклов. Сдвиг или расширение «горба» = изменение поведения.</p>' + fig_html(fig)))

    # 5. Корреляции ---------------------------------------------------------------
    cc = [c for c in T['feature'] if c in eng][:25]
    if len(cc) >= 3 and len(G) >= 5:
        C = G[cc].corr(method='spearman')
        lab = [_short(human, c)[:40] for c in cc]
        fig = go.Figure(go.Heatmap(z=C.values, x=lab, y=lab, zmin=-1, zmax=1, colorscale='RdBu_r',
                                   hovertemplate='%{y}<br>%{x}<br>ρ=%{z:.2f}<extra></extra>'))
        fig.update_layout(height=700, margin=dict(l=260, r=10, t=10, b=200), xaxis=dict(tickangle=45))
        out.append(('Что меняется вместе', '<p class="muted">Корреляция (Спирмен) между дневными медианами 25 самых «трендовых» признаков. Красный блок = признаки растут вместе: часто это одна первопричина (напр. износ насоса → дренаж ↑, температура ↑, время прессования ↑).</p>' + fig_html(fig)))

    # 6. Перед остановами ---------------------------------------------------------
    pc = ctx['pre_counts']
    pc = pc[pc > 0].sort_values(ascending=False)
    if len(S) and len(pc):
        long = S.sort_values('простой, ч', ascending=False).head(12)
        import re as _re
        seen, feats = set(), []
        for c in pc.index:
            k = _re.sub(r'_(mean|min|max|p95|rms|absmax|active)$', '', c)
            if k not in seen:
                seen.add(k); feats.append(c)
            if len(feats) == 4:
                break
        fig = make_subplots(rows=len(feats), cols=1, shared_xaxes=True, subplot_titles=[_short(human, c) for c in feats], vertical_spacing=0.06)
        for k, (_, s) in enumerate(long.iterrows()):
            ts = pd.Timestamp(s['останов с'])
            w = P[(P['t_start'] <= ts) & (P['t_start'] > ts - pd.Timedelta(hours=72))]
            if len(w) < 3:
                continue
            x = (w['t_start'] - ts).dt.total_seconds() / 3600
            for j, c in enumerate(feats):
                z = (w[c] - med[c]) / sig[c]
                z = z.rolling(5, min_periods=1).median()
                fig.add_scatter(x=x, y=z, mode='lines', line=dict(color=PAL[k % len(PAL)], width=1.3), name='%s (%.1f ч)' % (ts.strftime('%d.%m %H:%M'), s['простой, ч']),
                                legendgroup=str(k), showlegend=(j == 0), row=j + 1, col=1)
        for j in range(len(feats)):
            fig.add_hrect(y0=-3, y1=3, fillcolor='#999', opacity=0.15, line_width=0, row=j + 1, col=1)
        fig.update_xaxes(title_text='часы до останова', row=len(feats), col=1)
        fig.update_layout(height=230 * len(feats) + 80, margin=dict(l=40, r=10, t=40, b=40), legend=dict(font=dict(size=10)))
        out.append(('Как ведут себя признаки перед остановами',
                    '<p class="muted">12 самых длинных простоев, совмещённых по моменту останова (0 = последний цикл). Показаны признаки, которые чаще всего отклонялись перед остановами, в σ от нормы. '
                    'Если линии систематически уходят из серой полосы за N часов до 0 — это и есть «как умирает» узел и запас времени на предупреждение.</p>' + fig_html(fig)))

    # 7. Связи: усилие/работа, ход штампа/нижняя точка, длительность цикла -----------
    sc = []
    tnum = (P['t_start'] - P['t_start'].min()).dt.total_seconds() / 86400
    pairs = [('force_max_kN', 'work_kJ', 'Усилие vs работа'), ('die_strokes', 'pos_bottom_mm', 'Ходов штампа vs нижняя точка (износ штампа)'),
             ('temp_bottom_mean', 'press_time_s', 'Температура масла vs время прессования'),
             ('force_max_kN', 'hp_drop', 'Усилие vs просадка давления')]
    pairs = [p for p in pairs if p[0] in P and p[1] in P and P[p[0]].notna().any() and P[p[1]].notna().any()]
    if pairs:
        fig = make_subplots(rows=(len(pairs) + 1) // 2, cols=2, subplot_titles=[p[2] for p in pairs], vertical_spacing=0.16)
        for i, (x, y, _) in enumerate(pairs):
            fig.add_scattergl(x=P[x], y=P[y], mode='markers', marker=dict(size=4, color=tnum, colorscale='Viridis', showscale=(i == 0),
                              colorbar=dict(title='дни')), showlegend=False, row=i // 2 + 1, col=i % 2 + 1,
                              hovertemplate=x + '=%{x}<br>' + y + '=%{y}<extra></extra>')
            fig.update_xaxes(title_text=_short(human, x)[:45], row=i // 2 + 1, col=i % 2 + 1)
            fig.update_yaxes(title_text=_short(human, y)[:30], row=i // 2 + 1, col=i % 2 + 1)
        fig.update_layout(height=380 * ((len(pairs) + 1) // 2), margin=dict(l=60, r=10, t=40, b=50))
        sc.append(fig_html(fig))
    if 'cycle_s' in P:
        d = P.copy(); d['day'] = d['t_start'].dt.floor('D' if freq == 'D' else 'h')
        fig = go.Figure(go.Box(x=d['day'], y=d['cycle_s'], boxpoints=False, marker_color='#1f77b4'))
        fig.update_layout(height=300, title='Длительность цикла по %s, с' % ('дням' if freq == 'D' else 'часам'), margin=dict(l=40, r=10, t=40, b=30))
        sc.append(fig_html(fig))
    if sc:
        out.append(('Связи и производительность', '<p class="muted">Цвет точки — время (фиолетовый — начало периода, жёлтый — конец). Если облако «уезжает» с цветом, при тех же условиях узел ведёт себя иначе — признак износа.</p>' + ''.join(sc)))

    # 8. Форма цикла: эволюция и аномалии (читает исходные .shp) ----------------------
    out.extend(cycle_shapes(P, an_rows, freq, fig_html))
    return out


def _load_curves(path, step=5):
    from shpreader import read_shp
    r = read_shp(path)
    ix = {n: j for j, n in reversed(list(enumerate(r['ana_names'])))}
    t = np.arange(r['ana'].shape[0]) * 0.01
    res = {}
    for nm, _ in CYCLE_CH:
        if nm in ix:
            res[nm] = r['ana'][::step, ix[nm]]
    return t[::step], res


def cycle_shapes(P, an_rows, freq, fig_html, max_days=30):
    out = []
    if 'force_max_kN' not in P or not len(P):
        return out
    # типичный цикл по каждому дню (ближе всего к медиане усилия дня)
    d = P[['t_start', 'file', 'force_max_kN']].copy()
    d['day'] = d['t_start'].dt.floor('D' if freq == 'D' else 'h')
    picks = []
    for day, g in d.groupby('day'):
        i = (g['force_max_kN'] - g['force_max_kN'].median()).abs().idxmin()
        picks.append((day, P.loc[i, 'file']))
    if len(picks) > max_days:
        picks = [picks[int(i)] for i in np.linspace(0, len(picks) - 1, max_days)]
    picks = [p for p in picks if os.path.exists(p[1])]
    if len(picks) >= 2:
        chans = [c for c in CYCLE_CH[:4]]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[c[1] for c in chans])
        n = len(picks)
        for k, (day, f) in enumerate(picks):
            try:
                t, cur = _load_curves(f)
            except Exception:
                continue
            frac = k / max(1, n - 1)
            col = 'rgb(%d,%d,%d)' % (int(40 + 200 * frac), int(90 + 40 * (1 - frac)), int(200 * (1 - frac)))
            for j, (nm, lab) in enumerate(chans):
                if nm in cur:
                    fig.add_scatter(x=t, y=cur[nm], mode='lines', line=dict(color=col, width=1), name=str(day)[:16],
                                    legendgroup=str(k), showlegend=(j == 0), row=j // 2 + 1, col=j % 2 + 1)
        fig.update_layout(height=620, margin=dict(l=40, r=10, t=40, b=30), legend=dict(font=dict(size=9)))
        fig.update_xaxes(title_text='с от начала записи')
        out.append(('Как меняется форма цикла со временем',
                    '<p class="muted">По одному типичному циклу за каждый %s (до %d шт.): синие — начало периода, красные — конец. '
                    'Растянувшаяся кривая усилия, сдвинутая нижняя точка или «провал» давления у поздних циклов — изменение поведения пресса.</p>' % ('день' if freq == 'D' else 'час', max_days)
                    + fig_html(fig)))
    # аномальные циклы против типичного
    if an_rows and picks:
        ref = picks[len(picks) // 2][1]
        files = {os.path.basename(f): f for f in P['file']}
        anf = [files.get(r['файл']) for r in an_rows[:6]]
        anf = [f for f in anf if f and os.path.exists(f)]
        if anf:
            chans = CYCLE_CH
            fig = make_subplots(rows=3, cols=2, subplot_titles=[c[1] for c in chans])
            try:
                t, cur = _load_curves(ref)
                for j, (nm, lab) in enumerate(chans):
                    if nm in cur:
                        fig.add_scatter(x=t, y=cur[nm], mode='lines', line=dict(color='#888', width=3), name='типичный цикл',
                                        legendgroup='ref', showlegend=(j == 0), row=j // 2 + 1, col=j % 2 + 1)
            except Exception:
                pass
            for k, f in enumerate(anf):
                try:
                    t, cur = _load_curves(f)
                except Exception:
                    continue
                for j, (nm, lab) in enumerate(chans):
                    if nm in cur:
                        fig.add_scatter(x=t, y=cur[nm], mode='lines', line=dict(color=PAL[(k + 1) % len(PAL)], width=1.2), name=os.path.basename(f),
                                        legendgroup=f, showlegend=(j == 0), row=j // 2 + 1, col=j % 2 + 1)
            fig.update_layout(height=860, margin=dict(l=40, r=10, t=40, b=30), legend=dict(font=dict(size=9)))
            out.append(('Аномальные циклы против нормального',
                        '<p class="muted">Серая толстая линия — типичный цикл; цветные — 6 самых сильных аномалий. Видно, в какой момент цикла и в каком сигнале они расходятся.</p>' + fig_html(fig)))
    return out
