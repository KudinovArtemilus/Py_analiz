# -*- coding: utf-8 -*-
"""
Отчёт ЗА СУТКИ: как отработал пресс в выбранный день по сравнению с нормой (вся история до этого дня).

    python daily.py features_OP1.parquet --op OP1                    # последний день в данных
    python daily.py features_OP1.parquet --op OP1 --day 2026-09-28   # конкретный день
    python daily.py features_OP1.parquet --op OP1 --all-days         # отдельный отчёт на каждый день

Результат: <out>/day_ГГГГ-ММ-ДД_OP1.html и diagnosis_ГГГГ-ММ-ДД_OP1.txt
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import diagnose as D

# Ключевые показатели для таблицы «день против нормы» и почасовых графиков
KEY = ['force_max_kN', 'cycle_s', 'press_time_s', 'force_rise_s',
       'hp_min', 'hp_drop', 'temp_bottom_mean', 'temp_upper_mean', 'level_bottom_mean', 'level_upper_mean',
       'dieclamp_c1_min', 'dieclamp_c2_min', 'accu_rv1_min', 'accu_rv3_min',
       'casedrain_md2_mean', 'casedrain_md3_mean', 'casedrain_md4_mean', 'casedrain_md5_mean', 'casedrain_md6_mean',
       'lr_dev_absmax_mm', 'v12Y102_err_absmax', 'fc_speed_err_rms']
PLOT = ['force_max_kN', 'cycle_s', 'hp_min', 'temp_bottom_mean', 'dieclamp_c2_min', 'casedrain_md3_mean',
        'lr_dev_absmax_mm', 'v12Y102_err_absmax']


def one_day(df, day, a, out_dir):
    d0 = pd.Timestamp(day); d1 = d0 + pd.Timedelta(days=1)
    upto = df[df['t_start'] < d1].reset_index(drop=True)
    dayall = upto[upto['t_start'] >= d0]
    if dayall.empty:
        print('За %s циклов нет. Есть дни: %s' % (day, ', '.join(sorted(df['t_start'].dt.strftime('%Y-%m-%d').unique()))))
        return None
    prod = upto['force_max_kN'] > A.PROD_FORCE_KN
    P = upto[prod].reset_index(drop=True)
    T = P[P['t_start'] >= d0]
    H = P[P['t_start'] < d0]
    if a.baseline_start or a.baseline_end:
        m = pd.Series(True, index=P.index)
        if a.baseline_start: m &= P['t_start'] >= pd.Timestamp(a.baseline_start)
        if a.baseline_end: m &= P['t_start'] <= pd.Timestamp(a.baseline_end)
        B, bnote = P[m], 'заданный период'
    elif len(H) >= 30:
        B, bnote = H, 'вся история до этого дня (%d циклов)' % len(H)
    else:
        B, bnote = P.iloc[:max(10, int(len(P) * 0.2))], 'мало истории до этого дня — норма по первым 20%% данных (%d циклов)' % max(10, int(len(P) * 0.2))
    eng, raw, dig = A.select_columns(upto)
    feats = eng + raw
    med, sig = A.robust_stats(B[feats], P[feats])
    span_h = (P['t_start'].max() - P['t_start'].min()).total_seconds() / 3600
    diag, notes = D.evaluate(P, B, sig, span_h, T=T if len(T) else None)

    # простои внутри суток (включая паузу от последнего цикла прошлых суток)
    gaps_all = upto['t_start'].diff().dt.total_seconds() / 60
    gap_thr = max(a.gap_min, 4 * float(gaps_all.median()))
    i0 = max(0, dayall.index[0] - 1)
    stops = D.stops_summary(upto.iloc[i0:], gap_thr)
    g = upto.iloc[i0:]['t_start'].diff().dt.total_seconds() / 60
    down_h = float(g[g > gap_thr].sum() / 60)
    ds = upto['die_strokes'] if 'die_strokes' in upto else None
    die_changes = int((D.die_change_mask(ds).values & (upto['t_start'] >= d0).values).sum()) if ds is not None else 0

    # аномалии дня (выбросы относительно соседних циклов)
    Rm = P[eng].rolling(51, center=True, min_periods=10).median()
    Zl = ((P[eng] - Rm) / sig[eng]).abs().clip(upper=100)
    day_idx = T.index
    an = []
    for i in day_idx:
        z = Zl.loc[i].dropna()
        if len(z) and z.max() > a.z:
            top = z.sort_values(ascending=False).head(3)
            an.append((P.loc[i, 't_start'], os.path.basename(str(P.loc[i, 'file'])), float(top.iloc[0]),
                       '; '.join('%s: %s (обычно %s)' % (A.human(c), D._fmt(P.loc[i, c]), D._fmt(Rm.loc[i, c])) for c in top.index)))
    an.sort(key=lambda r: -r[2])

    # таблица «день против нормы»
    rows = []
    q = B[[c for c in KEY if c in B]].quantile([0.05, 0.5, 0.95])
    for c in KEY:
        if c not in T or T[c].notna().sum() == 0 or c not in q:
            continue
        v = T[c].median(); lo, mid, hi = q.loc[0.05, c], q.loc[0.5, c], q.loc[0.95, c]
        z = (v - med.get(c, mid)) / sig.get(c, np.nan) if sig.get(c, 0) else 0
        mark = '🟢' if lo <= v <= hi else ('🟡' if abs(z) < 4 else '🔴')
        rows.append((mark, A.human(c), D._fmt(v), D._fmt(mid), '%s … %s' % (D._fmt(lo), D._fmt(hi)), '%+.1f' % z if np.isfinite(z) else '—'))

    short = span_h < 24 * 7
    dlabel = d0.strftime('%d.%m.%Y')
    txt = D.to_text(diag, a.op or '', 'сутки %s' % dlabel, len(T), stops, short, notes)
    head = ['СУТКИ %s, %s: циклов %d (рабочих %d), простоев %.1f ч, замен штампа %d, аномальных циклов %d. Норма: %s.' % (
        dlabel, a.op or '', len(dayall), len(T), down_h, die_changes, len(an), bnote), '']
    txt = '\n'.join(head) + txt
    base = '%s_%s' % (d0.strftime('%Y-%m-%d'), a.op or 'all')
    with open(os.path.join(out_dir, 'diagnosis_%s.txt' % base), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print('\n' + txt + '\n')

    write_html(os.path.join(out_dir, 'day_%s.html' % base), a, dlabel, dayall, T, B, P, q, diag, notes, stops, short,
               rows, an, down_h, die_changes, bnote)
    return base


def write_html(path, a, dlabel, dayall, T, B, P, q, diag, notes, stops, short, rows, an, down_h, die_changes, bnote):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    h = ['<h1>Пресс %s — отчёт за сутки %s</h1>' % (esc(a.op or ''), dlabel),
         '<p class="muted">Норма: %s. Сравнивается каждый цикл этих суток с нормой.</p>' % esc(bnote)]
    work_h = float(T['cycle_s'].sum() / 3600) if 'cycle_s' in T else 0
    kp = [('Циклов', len(dayall)), ('Рабочих (ковка)', len(T)), ('Простои, ч', '%.1f' % down_h),
          ('Замен штампа', die_changes), ('Аномальных циклов', len(an)),
          ('Узлов: проблема / наблюдать', '%d / %d' % (sum(d['status'] == D.ALARM for d in diag), sum(d['status'] == D.WARN for d in diag)))]
    h.append('<div class="kpis">' + ''.join('<div class="kpi"><b>%s</b><span>%s</span></div>' % (v, k) for k, v in kp) + '</div>')
    h.append('<h2>Диагноз за сутки</h2>' + D.to_html(diag, stops, short, notes))

    h.append('<h2>День против нормы</h2><p class="muted">Медиана за сутки против нормы. 🟢 в обычном диапазоне (5–95% нормы), '
             '🟡 вне диапазона, но не сильно, 🔴 сильно (больше 4σ).</p>')
    h.append('<div class="tw"><table><tr><th></th><th>Показатель</th><th>За сутки</th><th>Норма</th><th>Обычный диапазон</th><th>σ</th></tr>' +
             ''.join('<tr>' + ''.join('<td>%s</td>' % esc(x) for x in r) + '</tr>' for r in rows) + '</table></div>')

    cols = [c for c in PLOT if c in T and T[c].notna().any()]
    if cols and len(T):
        nr = (len(cols) + 1) // 2
        fig = make_subplots(rows=nr, cols=2, subplot_titles=[A.human(c)[:55] for c in cols], vertical_spacing=0.08)
        for i, c in enumerate(cols):
            r_, c_ = i // 2 + 1, i % 2 + 1
            if c in q:
                fig.add_hrect(y0=q.loc[0.05, c], y1=q.loc[0.95, c], fillcolor='#2ca02c', opacity=0.12, line_width=0, row=r_, col=c_)
            fig.add_scatter(x=T['t_start'], y=T[c], mode='markers', marker=dict(size=4, color='#1f77b4'), showlegend=False, row=r_, col=c_,
                            hovertemplate='%{x|%H:%M}<br>%{y}<extra></extra>')
            hr = T.set_index('t_start')[c].resample('h').median().dropna()
            fig.add_scatter(x=hr.index, y=hr.values, mode='lines', line=dict(color='#d62728', width=2), showlegend=False, row=r_, col=c_)
        fig.update_xaxes(tickformat='%H:%M')
        fig.update_layout(height=260 * nr, margin=dict(l=40, r=10, t=40, b=30))
        h.append('<h2>Сутки по часам</h2><p class="muted">Точки — циклы, красная линия — медиана за час, зелёная полоса — норма.</p>' +
                 pio.to_html(fig, include_plotlyjs='inline', full_html=False, config={'displaylogo': False}))
    if 'cycle_s' in dayall:
        cnt = dayall.set_index('t_start')['cycle_s'].resample('h').count()
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color='#1f77b4'))
        fig.update_layout(height=230, title='Циклов в час', margin=dict(l=40, r=10, t=40, b=30), xaxis=dict(tickformat='%H:%M'))
        h.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displaylogo': False}))

    h.append('<h2>Аномальные циклы за сутки</h2>')
    if an:
        h.append('<div class="tw"><table><tr><th>Время</th><th>Файл</th><th>Сила, σ</th><th>Что не так</th></tr>' +
                 ''.join('<tr><td>%s</td><td>%s</td><td>%.0f</td><td>%s</td></tr>' % (t.strftime('%H:%M:%S'), esc(f), z, esc(w)) for t, f, z, w in an[:50]) +
                 '</table></div>')
    else:
        h.append('<p>Нет.</p>')
    css = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#e3e3e3;--card:#f6f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2c2e33;--card:#1d1f23}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px}
h2{margin-top:30px;border-bottom:1px solid var(--line);padding-bottom:4px}.muted{color:var(--mut)}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{background:var(--card);padding:10px 14px;border-radius:8px;min-width:140px}.kpi b{display:block;font-size:24px}.kpi span{color:var(--mut)}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}th{background:var(--card)}
""" + D.CSS
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                 '<title>Пресс за сутки</title><style>%s</style></head><body>%s</body></html>' % (css, '\n'.join(h)))
    print('Готово:', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features', nargs='?', help='features_OP1.parquet')
    ap.add_argument('--op')
    ap.add_argument('--day', help='ГГГГ-ММ-ДД; по умолчанию — последний день в данных')
    ap.add_argument('--all-days', action='store_true', help='отчёт на каждый день')
    ap.add_argument('-o', '--out', default='daily')
    ap.add_argument('--gap-min', type=float, default=30)
    ap.add_argument('--z', type=float, default=6.0)
    ap.add_argument('--baseline-start'); ap.add_argument('--baseline-end')
    a = ap.parse_args()
    if not a.features or not os.path.exists(a.features):
        raise SystemExit('Укажите файл признаков (результат extract_features.py), напр.: python daily.py features_OP1.parquet --op OP1')
    os.makedirs(a.out, exist_ok=True)
    df = A.load(a.features)
    if a.op and 'op' in df:
        df = df[df['op'].str.upper() == a.op.upper()].reset_index(drop=True)
    days = sorted(df['t_start'].dt.strftime('%Y-%m-%d').unique())
    if not days:
        raise SystemExit('В данных нет циклов')
    todo = days if a.all_days else [a.day or days[-1]]
    for d in todo:
        one_day(df, d, a, a.out)


if __name__ == '__main__':
    main()