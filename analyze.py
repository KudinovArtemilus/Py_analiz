# -*- coding: utf-8 -*-
"""
Шаг 2. Читает features.parquet и строит отчёт:
  * НОРМА — нормальные диапазоны каждого признака (по базовому периоду);
  * АНОМАЛИИ — циклы, сильно отличающиеся от нормы, и что именно в них не так;
  * ДЕГРАДАЦИЯ — признаки, которые медленно и устойчиво уходят от нормы (износ);
  * ОСТАНОВЫ — длинные паузы между циклами и что «плыло» перед каждой;
  * ЗДОРОВЬЕ ДАТЧИКОВ — замершие / пропавшие сигналы.

Запуск:
    python analyze.py features.parquet -o report
    python analyze.py features.parquet -o report --baseline-start 2026-09-01 --baseline-end 2026-09-07
"""
import argparse, os, re, json, datetime as dt, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PROD_FORCE_KN = 5000.0  # цикл считается рабочим (ковка), если усилие выше

# Пояснения к инженерным признакам: (подсистема, описание, что значит рост)
DESC = {
    'cycle_s': ('Цикл', 'Длительность записи цикла, с', 'медленнее работа / ожидания'),
    'press_time_s': ('Ползун', 'Время прессования (усилие > 1000 кН), с', 'насосы/клапаны не дают скорость'),
    'force_max_kN': ('Ползун', 'Максимальное усилие, кН', 'другая заготовка/температура/штамп'),
    'force_rise_s': ('Ползун', 'Время нарастания усилия 10→90%, с', 'падение подачи насосов, утечки'),
    'work_kJ': ('Ползун', 'Работа деформации, кДж', 'холоднее заготовка, износ штампа'),
    'force_impulse_kNs': ('Ползун', 'Импульс усилия, кН·с', ''),
    'pos_bottom_mm': ('Ползун', 'Нижняя точка ползуна, мм', 'износ штампа / смещение датчика'),
    'stroke_mm': ('Ползун', 'Ход ползуна, мм', ''),
    'v_down_max': ('Ползун', 'Макс. скорость вниз, мм/с', ''),
    'v_up_max': ('Ползун', 'Макс. скорость вверх, мм/с', ''),
    'slide_drift_top_mm_s': ('Ползун', 'Дрейф ползуна в покое наверху, мм/с', 'утечка через уплотнения/клапаны'),
    'lr_dev_absmax_mm': ('Ползун', 'Перекос левый-правый (макс.), мм', 'износ направляющих, разбаланс'),
    'lr_dev_press_mean_mm': ('Ползун', 'Перекос при прессовании (среднее), мм', 'износ направляющих'),
    'hp_mean': ('Гидростанция', 'Высокое давление, среднее, бар', ''),
    'hp_min': ('Гидростанция', 'Высокое давление, минимум, бар', 'просадка: насосы/аккумуляторы'),
    'hp_drop': ('Гидростанция', 'Просадка высокого давления за цикл, бар', 'насосы не справляются, азот в аккум.'),
    'hp_recover_s': ('Гидростанция', 'Время восстановления давления после прессования, с', 'износ насосов'),
    'temp_bottom_mean': ('Гидростанция', 'Температура масла, нижний бак, °C', 'охлаждение, внутренние утечки'),
    'temp_upper_mean': ('Гидростанция', 'Температура масла, верхний бак, °C', 'охлаждение, внутренние утечки'),
    'level_bottom_mean': ('Гидростанция', 'Уровень, нижний бак', 'падение = внешняя утечка'),
    'level_upper_mean': ('Гидростанция', 'Уровень, верхний бак', 'падение = внешняя утечка'),
    'accu_rv1_min': ('Аккумуляторы', 'Аккумулятор RV1, минимум за цикл, бар', 'падение = азот/заряд'),
    'accu_rv3_min': ('Аккумуляторы', 'Аккумулятор RV3, минимум за цикл, бар', 'падение = азот/заряд'),
    'fc_speed_err_rms': ('Привод насосов', 'Частотник: ошибка задание-факт оборотов (RMS)', 'перегруз/износ двигателя-насоса'),
    'MB_vel_err_rms': ('Подвижный стол (MB)', 'Ошибка скорости MB (RMS)', 'износ привода/клапана'),
}
for k in range(1, 7):
    DESC['casedrain_md%d_mean' % k] = ('Насосы (утечки)', 'Давление дренажа корпуса MD%d, среднее' % k, 'РОСТ = износ насоса (внутренние утечки)')
    DESC['casedrain_md%d_p95' % k] = ('Насосы (утечки)', 'Давление дренажа корпуса MD%d, p95' % k, 'РОСТ = износ насоса')
for a in range(1, 7):
    for b in (1, 2):
        p = 'pump%d%d' % (a, b)
        DESC[p + '_pivot_err_rms'] = ('Насосы (регулирование)', 'Насос %d.%d: ошибка угла наклона (задание-факт)' % (a, b), 'износ регулятора/люльки')
        DESC[p + '_pivot_err_active'] = ('Насосы (регулирование)', 'Насос %d.%d: ошибка угла под нагрузкой' % (a, b), 'износ регулятора')
        DESC[p + '_pivot_lag'] = ('Насосы (регулирование)', 'Насос %d.%d: запаздывание угла, ×10 мс' % (a, b), 'замедление регулятора')
        DESC[p + '_valve_err_rms'] = ('Насосы (регулирование)', 'Насос %d.%d: ошибка клапана управления' % (a, b), 'износ/загрязнение клапана')
        DESC[p + '_press_max'] = ('Насосы (регулирование)', 'Насос %d.%d: макс. давление, бар' % (a, b), '')
for s in ('v12Y102', 'v12Y069_21', 'v12Y069_22', 'v20Y019', 'v31Y102', 'vCCylL_Y22_2', 'vCCylR_Y22_1',
          'vLCylL_Y21_2', 'vLCylR_Y21_1', 'vMB_Y1', 'vMB_Y2'):
    nm = s[1:]
    DESC[s + '_err_rms'] = ('Сервоклапаны', 'Клапан %s: ошибка задание-золотник (RMS)' % nm, 'загрязнение/износ золотника')
    DESC[s + '_err_absmax'] = ('Сервоклапаны', 'Клапан %s: макс. ошибка' % nm, 'заедание')
    DESC[s + '_lag'] = ('Сервоклапаны', 'Клапан %s: запаздывание, ×10 мс' % nm, 'замедление отклика')
for cyl, cn in (('CCyl', 'Центрирующий цил.'), ('LCyl', 'Нижний цил.')):
    for side, sn in (('Left', 'лев.'), ('Right', 'прав.')):
        DESC['%s%s_posdev_rms' % (cyl, side)] = ('Цилиндры CCyl/LCyl', '%s %s: ошибка позиционирования (RMS)' % (cn, sn), 'износ/утечки/клапан')
        DESC['%s%s_posdev_absmax' % (cyl, side)] = ('Цилиндры CCyl/LCyl', '%s %s: макс. ошибка позиции' % (cn, sn), '')
        DESC['%s%s_force_absmax' % (cyl, side)] = ('Цилиндры CCyl/LCyl', '%s %s: макс. усилие' % (cn, sn), '')
        DESC['%s%s_speed_absmax' % (cyl, side)] = ('Цилиндры CCyl/LCyl', '%s %s: макс. скорость' % (cn, sn), '')

for base, (ss, nm, why) in {
        'lp': ('Гидростанция', 'Низкое давление, бар', ''),
        'temp_bottom': ('Гидростанция', 'Температура масла, нижний бак, °C', 'охлаждение, внутренние утечки'),
        'temp_upper': ('Гидростанция', 'Температура масла, верхний бак, °C', 'охлаждение, внутренние утечки'),
        'level_bottom': ('Гидростанция', 'Уровень, нижний бак', 'падение = внешняя утечка'),
        'level_upper': ('Гидростанция', 'Уровень, верхний бак', 'падение = внешняя утечка'),
        'accu_rv1': ('Аккумуляторы', 'Аккумулятор RV1, бар', 'падение = азот/заряд'),
        'accu_rv3': ('Аккумуляторы', 'Аккумулятор RV3, бар', 'падение = азот/заряд'),
        'dieclamp_c1': ('Зажим штампа', 'Давление зажима, контур 1, бар', 'падение = утечка контура'),
        'dieclamp_c2': ('Зажим штампа', 'Давление зажима, контур 2, бар', 'падение = утечка контура')}.items():
    for suf, sn in (('_mean', 'среднее'), ('_min', 'минимум'), ('_max', 'максимум')):
        DESC.setdefault(base + suf, (ss, '%s, %s' % (nm, sn), why))
DESC.update({'press_start_s': ('Цикл', 'Начало прессования от старта записи, с', 'медленный подвод / ожидание'),
             'pos_top_mm': ('Ползун', 'Верхняя точка ползуна, мм', ''), 'pos_at_force_max_mm': ('Ползун', 'Позиция при макс. усилии, мм', ''),
             'slide_moving_s': ('Ползун', 'Время движения ползуна, с', ''),
             'bej_speed_max': ('Выталкиватели', 'Нижний выталкиватель: макс. скорость', 'падение = износ/утечки'),
             'sej_speed_max': ('Выталкиватели', 'Верхний выталкиватель: макс. скорость', 'падение = износ/утечки'),
             'bej_force_max': ('Выталкиватели', 'Нижний выталкиватель: макс. усилие', 'рост = прилипание к штампу'),
             'scu_force_max': ('Выталкиватели', 'Подушка ползуна: макс. усилие', '')})

META = {'fn_prefix', 'fn_piece', 'fn_heat', 'file', 'op', 't_start', 'n_samples', 'piece_id', 'heat_id', 'die_strokes', 'which_op_max'}


def subsystem(col):
    if col in DESC:
        return DESC[col][0]
    if col.startswith('a'):
        return 'Сырые аналоговые'
    if col.startswith('d'):
        return 'Дискретные'
    return 'Прочее'


def human(col):
    if col in DESC:
        return DESC[col][1]
    m = re.match(r'a\d{3}_(.*)__(\w+)$', col)
    if m:
        return '%s [%s]' % (m.group(1).replace('_', ' '), m.group(2))
    m = re.match(r'd\d{3}_(.*)__(\w+)$', col)
    if m:
        return '%s [%s]' % (m.group(1).replace('_', ' '), 'доля вкл.' if m.group(2) == 'on' else 'кол-во включений')
    return col


def load(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df['t_start'] = pd.to_datetime(df['t_start'])
    num = df.select_dtypes('number').columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).mask(df[num].abs() > 1e15)  # мусор из старых прогонов
    return df.sort_values('t_start').reset_index(drop=True)


def robust_stats(X, ALL=None):
    """Медиана и робастная σ по базовому периоду X. ALL — все циклы: из них берётся «пол» σ,
    чтобы почти-постоянные в базе признаки не давали z в миллионы."""
    med = X.median()
    mad = (X - med).abs().median() * 1.4826
    iqr = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    sig = np.maximum(mad.fillna(0), iqr.fillna(0))
    R = X if ALL is None else ALL
    spread = (R.quantile(0.95) - R.quantile(0.05)).abs()
    floor = np.maximum(spread * 0.1, (R.quantile(0.75) - R.quantile(0.25)).abs() / 1.349 * 0.5)
    floor = np.maximum(floor, med.abs() * 0.01) + 1e-6
    # целочисленные признаки (запаздывания в отсчётах, счётчики): шаг 1 — это не 100σ
    isint = pd.Series([bool(np.all(np.mod(R[c].dropna().values, 1) == 0)) if R[c].notna().any() else False for c in R.columns], index=R.columns)
    floor = np.where(isint, np.maximum(floor, 1.0), floor)
    return med, np.maximum(sig, pd.Series(floor, index=sig.index))


def select_columns(df):
    num = [c for c in df.columns if c not in META and c not in ('production', 'anomaly')
           and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    X = df[num]
    keep = [c for c in num if X[c].notna().mean() > 0.5 and X[c].nunique(dropna=True) > 1]
    eng = [c for c in keep if not re.match(r'[ad]\d{3}_', c)]
    raw = [c for c in keep if re.match(r'a\d{3}_.*__(mean|max|min|std)$', c)]
    dig = [c for c in keep if re.match(r'd\d{3}_', c)]
    return eng, raw, dig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features', nargs='?', help='features_OP1.parquet — результат extract_features.py')
    ap.add_argument('-o', '--out', default='report')
    ap.add_argument('--baseline-start'); ap.add_argument('--baseline-end')
    ap.add_argument('--baseline-frac', type=float, default=0.2, help='доля первых по времени циклов как «норма», если даты не заданы')
    ap.add_argument('--gap-min', type=float, default=30, help='пауза между циклами (мин), считающаяся остановом')
    ap.add_argument('--z', type=float, default=6.0, help='порог робастного z для аномалии')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--op', help='анализировать только эту операцию (OP1, OP2...). Если в данных несколько OP — лучше задать')
    ap.add_argument('--from', dest='date_from'); ap.add_argument('--to', dest='date_to')
    a = ap.parse_args()
    if not a.features or not os.path.exists(a.features):
        print('Не указан или не найден файл признаков: %s\n'
              'Сначала запустите extract_features.py (или просто run.py — он делает оба шага):\n'
              '  python extract_features.py D:\\press --op OP1 -o features_OP1\n'
              '  python analyze.py features_OP1.parquet --op OP1 -o report_OP1' % a.features)
        raise SystemExit(2)
    os.makedirs(a.out, exist_ok=True)

    df = load(a.features)
    if a.op:
        df = df[df['op'].str.upper() == a.op.upper()].reset_index(drop=True)
    elif 'op' in df and df['op'].nunique() > 1:
        print('ВНИМАНИЕ: в данных несколько операций %s — норма будет общей. Лучше запустить отдельно с --op.' % sorted(df['op'].unique()))
    if a.date_from: df = df[df['t_start'] >= pd.Timestamp(a.date_from)]
    if a.date_to: df = df[df['t_start'] < pd.Timestamp(a.date_to) + pd.Timedelta(days=1)]
    df = df.reset_index(drop=True)
    df['production'] = df.get('force_max_kN', pd.Series(np.nan, index=df.index)) > PROD_FORCE_KN
    eng, raw, dig = select_columns(df)
    P = df[df['production']].reset_index(drop=True)
    if len(P) < 10:
        P = df.copy()
    print('Циклов всего: %d, рабочих: %d; признаков: инженерных %d, сырых %d, дискретных %d'
          % (len(df), int(df['production'].sum()), len(eng), len(raw), len(dig)))

    # ---------- базовый период ----------
    if a.baseline_start or a.baseline_end:
        m = pd.Series(True, index=P.index)
        if a.baseline_start: m &= P['t_start'] >= pd.Timestamp(a.baseline_start)
        if a.baseline_end: m &= P['t_start'] <= pd.Timestamp(a.baseline_end)
        B = P[m]
    else:
        B = P.iloc[:max(10, int(len(P) * a.baseline_frac))]
    feats = eng + raw
    med, sig = robust_stats(B[feats], P[feats])
    Z = (P[feats] - med) / sig

    # ---------- НОРМА ----------
    q = B[eng + raw].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
    q.columns = ['p1', 'p5', 'median', 'p95', 'p99']
    q.insert(0, 'описание', [human(c) for c in q.index]); q.insert(0, 'подсистема', [subsystem(c) for c in q.index])
    q.to_csv(os.path.join(a.out, 'normal_profile.csv'), encoding='utf-8-sig')

    # ---------- АНОМАЛИИ ----------
    try:
        from sklearn.ensemble import IsolationForest
        Xi = Z[eng].clip(-20, 20).fillna(0).values
        iso = IsolationForest(n_estimators=200, contamination='auto', random_state=0).fit(Xi[:len(B)] if len(B) > 50 else Xi)
        P['iforest'] = -iso.score_samples(Xi)
    except Exception:
        P['iforest'] = np.nan
    # локальные выбросы: отклонение от скользящей медианы соседних циклов (медленный износ не мешает)
    Rm = P[eng].rolling(51, center=True, min_periods=10).median()
    Zl = ((P[eng] - Rm) / sig[eng]).abs().clip(upper=100)
    P['anom_maxz'] = Zl.max(axis=1).values
    P['anom_score'] = np.sort(Zl.fillna(0).values, axis=1)[:, -3:].mean(axis=1)
    P['anomaly'] = P['anom_maxz'] > a.z

    def explain(i, k=5):
        z = ((P.loc[i, eng] - Rm.loc[i]) / sig[eng]).dropna()
        z = z.reindex(z.abs().sort_values(ascending=False).index)[:k]
        z = z[z.abs() > a.z / 2]
        return '; '.join('%s: %.4g (соседние циклы %.4g, z=%+.1f)' % (human(c), P.loc[i, c], Rm.loc[i, c], z[c]) for c in z.index)

    an = P[P['anomaly']].sort_values('anom_score', ascending=False)
    an_rows = [{'время': P.loc[i, 't_start'], 'файл': os.path.basename(P.loc[i, 'file']),
                'score': round(P.loc[i, 'anom_score'], 1), 'что не так': explain(i)} for i in an.index]
    pd.DataFrame(an_rows).to_csv(os.path.join(a.out, 'anomalies.csv'), index=False, encoding='utf-8-sig')
    # какие признаки чаще всего виноваты
    culprit = (Zl > a.z).sum().sort_values(ascending=False)
    culprit = culprit[culprit > 0]

    # ---------- ДЕГРАДАЦИЯ / ТРЕНДЫ ----------
    span_days = (P['t_start'].max() - P['t_start'].min()).total_seconds() / 86400
    freq = 'D' if span_days >= 5 else 'h'
    G = P.set_index('t_start')[feats].resample(freq).median().dropna(how='all')
    Gz = (G - med) / sig
    n_tail = max(3, int(len(G) * 0.2))
    trows = []
    for c in feats:
        s = Gz[c].dropna()
        if len(s) < 5:
            continue
        rho = pd.Series(np.arange(len(s)), index=s.index).corr(s, method='spearman')
        shift = s.iloc[-n_tail:].median()
        trows.append((c, rho, shift, abs(rho) * min(abs(shift), 50)))
    T = pd.DataFrame(trows, columns=['feature', 'rho', 'shift_sigma', 'score']).sort_values('score', ascending=False)
    T['подсистема'] = T['feature'].map(subsystem); T['описание'] = T['feature'].map(human)
    T['смысл роста'] = T['feature'].map(lambda c: DESC.get(c, ('', '', ''))[2])
    T.to_csv(os.path.join(a.out, 'trends.csv'), index=False, encoding='utf-8-sig')
    Tsig = T[(T['rho'].abs() > 0.6) & (T['shift_sigma'].abs() > 3)]
    # инженерные признаки — первыми; сырые — только если по подсистеме нет инженерного
    Ttop = pd.concat([Tsig[~Tsig.feature.str.match(r'a\d{3}_')], Tsig[Tsig.feature.str.match(r'a\d{3}_')]])
    sigkey = Ttop['feature'].str.replace(r'(__\w+|_(mean|min|max|p95|rms|absmax|active))$', '', regex=True)
    Ttop = Ttop[~sigkey.duplicated()].head(a.top)  # один график на сигнал

    # ---------- ОСТАНОВЫ и предвестники ----------
    allt = df['t_start']
    gaps = allt.diff().dt.total_seconds() / 60
    # если обработана выборка (--every N), обычный интервал между циклами большой — порог поднимаем
    typical = float(gaps.median()) if len(gaps.dropna()) else 0
    gap_thr = max(a.gap_min, 4 * typical)
    if gap_thr > a.gap_min:
        print('Интервал между циклами в данных ~%.0f мин (похоже на выборку) — порог останова поднят до %.0f мин' % (typical, gap_thr))
    stops = []
    pre_counts = pd.Series(0.0, index=eng)
    for i in np.where(gaps > gap_thr)[0]:
        t_stop, t_resume = allt.iloc[i - 1], allt.iloc[i]
        before = P[(P['t_start'] <= t_stop)].tail(50)
        if len(before) < 5:
            continue
        zb = ((before[eng].median() - med[eng]) / sig[eng]).dropna()
        zb = zb.reindex(zb.abs().sort_values(ascending=False).index)
        pre_counts[zb[zb.abs() > 3].index] += 1
        stops.append({'останов с': t_stop, 'пуск': t_resume, 'простой, ч': round(gaps.iloc[i] / 60, 2),
                      'что отклонялось за 50 циклов до': '; '.join('%s z=%+.1f' % (human(c), zb[c]) for c in zb.index[:5] if abs(zb[c]) > 3) or 'ничего заметного'})
    S = pd.DataFrame(stops)
    if len(S):
        S = S.sort_values('простой, ч', ascending=False)
    S.to_csv(os.path.join(a.out, 'stops.csv'), index=False, encoding='utf-8-sig')

    # ---------- ЗДОРОВЬЕ ДАТЧИКОВ ----------
    sens = []
    stdcols = [c for c in df.columns if re.match(r'a\d{3}_.*__std$', c)]
    tail = P.tail(max(10, int(len(P) * 0.2)))
    not_meas = re.compile(r'Date|Day|Time|Soll|Anteil|Set|set_|Mode|mode|Tune|CMD|Cmd|Target|Calc|Needed|Piece|Heat|Stokes|Which|Flow|Out|Dev|Lim|Part|Filter|Local|State|Encoder_?Val|_s$', re.I)
    for c in stdcols:
        if not_meas.search(c.replace('__std', '')):
            continue
        b, t = B[c].median(), tail[c].median()
        mc = c.replace('__std', '__mean')
        scale = abs(B[mc].median()) if mc in B else 0
        if b > 0 and t == 0 and b > 0.002 * (scale + 1e-9):
            sens.append((human(c).replace(' [std]', ''), 'замер (не меняется) в последний период', b, t))
    for c in [c for c in df.columns if c.endswith('__nan')]:
        if tail[c].mean() > 0.01:
            sens.append((human(c), 'NaN/мусор в данных', B[c].mean(), tail[c].mean()))
    SH = pd.DataFrame(sens, columns=['сигнал', 'проблема', 'база', 'сейчас'])

    # ---------- ДИАГНОЗ по узлам ----------
    import diagnose
    span_h = (P['t_start'].max() - P['t_start'].min()).total_seconds() / 3600
    diag, dnotes = diagnose.evaluate(P, B, sig, span_h)
    dstops = diagnose.stops_summary(df, gap_thr)
    short = span_h < 24 * 7
    dtext = diagnose.to_text(diag, a.op or ','.join(sorted(df['op'].dropna().unique())) if 'op' in df else '',
                             '%s – %s' % (df['t_start'].min().strftime('%d.%m.%Y %H:%M'), df['t_start'].max().strftime('%d.%m.%Y %H:%M')),
                             len(P), dstops, short, dnotes)
    with open(os.path.join(a.out, 'diagnosis.txt'), 'w', encoding='utf-8') as fh:
        fh.write(dtext)
    print('\n' + dtext + '\n')
    dhtml = diagnose.to_html(diag, dstops, short, dnotes)

    ctx = dict(P=P, B=B, G=G, med=med, sig=sig, eng=eng, T=T, Ttop=Ttop, S=S, q=q, freq=freq, human=human,
               subsystem=subsystem, an_rows=an_rows, Zl=Zl, z=a.z, pre_counts=pre_counts)
    ctx['diag_html'] = dhtml
    write_html(a, df, P, B, G, med, sig, eng, Ttop, T, an_rows, culprit, S, pre_counts, SH, q, freq, ctx)
    print('Готово: %s' % os.path.join(a.out, 'report.html'))


def write_html(a, df, P, B, G, med, sig, eng, Ttop, T, an_rows, culprit, S, pre_counts, SH, q, freq, ctx=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    figs = []

    def fig_html(fig, first=[True]):
        h = pio.to_html(fig, include_plotlyjs=('inline' if first[0] else False), full_html=False,
                        config={'displaylogo': False})
        first[0] = False
        return h

    def esc(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def table(dfx, maxrows=200):
        if dfx is None or len(dfx) == 0:
            return '<p class="muted">нет</p>'
        h = '<div class="tw"><table><tr>' + ''.join('<th>%s</th>' % esc(c) for c in dfx.columns) + '</tr>'
        for _, r in dfx.head(maxrows).iterrows():
            h += '<tr>' + ''.join('<td>%s</td>' % esc(round(v, 4) if isinstance(v, float) else v) for v in r.values) + '</tr>'
        return h + '</table></div>'

    sec = []
    t0, t1 = df['t_start'].min(), df['t_start'].max()
    sec.append('<h1>Пресс: норма, аномалии, деградация</h1>'
               '<p class="muted">Период: %s — %s · циклов: %d (рабочих %d) · база «нормы»: %s — %s (%d циклов)</p>'
               % (t0, t1, len(df), int(df['production'].sum()), B['t_start'].min(), B['t_start'].max(), len(B)))
    kp = [('Аномальных циклов', int(P['anomaly'].sum())), ('Признаков с трендом', len(Ttop)),
          ('Остановов > %g мин' % a.gap_min, len(S)), ('Проблем датчиков', len(SH))]
    sec.append('<div class="kpis">' + ''.join('<div class="kpi"><b>%s</b><span>%s</span></div>' % (v, k) for k, v in kp) + '</div><!--TOC-->')
    if ctx is not None and ctx.get('diag_html'):
        sec.append('<h2>Диагноз по узлам</h2><p class="muted">Автоматический вывод простым языком. Ниже — графики, на которых он основан. '
                   'Текстовая версия — diagnosis.txt.</p>' + ctx['diag_html'])

    # производительность
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Циклов в %s' % ('день' if freq == 'D' else 'час'), 'Аномальность циклов (score)'))
    cnt = df.set_index('t_start')['file'].resample(freq).count()
    fig.add_bar(x=cnt.index, y=cnt.values, row=1, col=1, name='циклы')
    fig.add_scatter(x=P['t_start'], y=P['anom_score'], mode='markers', marker=dict(size=4,
                    color=np.where(P['anomaly'], '#d62728', '#1f77b4')), row=1, col=2, name='score')
    fig.update_layout(height=320, showlegend=False, margin=dict(l=40, r=10, t=40, b=30))
    sec.append('<h2>Общая картина</h2>' + fig_html(fig))

    # дополнительные графики (charts.py)
    extra = []
    if ctx is not None:
        try:
            import charts
            extra = charts.build(ctx, fig_html)
        except Exception as e:
            import traceback; traceback.print_exc()
            sec.append('<p class="muted">Доп. графики не построены: %s</p>' % esc(e))
    for title, body in extra[:1]:          # карта здоровья — сразу после общей картины
        sec.append('<h2>%s</h2>%s' % (title, body))

    # деградация
    sec.append('<h2>Деградация: что устойчиво уходит от нормы</h2><p class="muted">Медиана по %s; серая полоса — норма (p5–p95 базового периода). '
               'shift — насколько сигм ушло в последние 20%% периода; rho — монотонность тренда.</p>' % ('дням' if freq == 'D' else 'часам'))
    if len(Ttop):
        for _, r in Ttop.iterrows():
            c = r['feature']
            fg = go.Figure()
            fg.add_hrect(y0=q.loc[c, 'p5'], y1=q.loc[c, 'p95'], fillcolor='#999', opacity=0.18, line_width=0)
            fg.add_scatter(x=P['t_start'], y=P[c], mode='markers', marker=dict(size=3, color='#9ecae1'), name='циклы')
            fg.add_scatter(x=G.index, y=G[c], mode='lines+markers', line=dict(color='#d62728', width=2), name='медиана')
            fg.update_layout(title='%s — %s<br><sub>shift %+.1fσ, rho %+.2f. %s</sub>' % (r['подсистема'], r['описание'], r['shift_sigma'], r['rho'], r['смысл роста']),
                             height=300, showlegend=False, margin=dict(l=40, r=10, t=60, b=30))
            sec.append(fig_html(fg))
    else:
        sec.append('<p>Устойчивых трендов не найдено.</p>')
    sec.append('<details><summary>Все признаки по силе тренда (top 100)</summary>' +
               table(T[['подсистема', 'описание', 'shift_sigma', 'rho', 'смысл роста']].head(100)) + '</details>')

    for title, body in extra[1:]:
        sec.append('<h2>%s</h2>%s' % (title, body))

    # аномалии
    sec.append('<h2>Аномальные циклы</h2><p class="muted">Цикл аномален, если хотя бы один инженерный признак отличается от соседних циклов (скользящая медиана ±25 циклов) больше чем на %g робастных σ. Медленный уход от нормы — в разделе «Деградация». Полный список — anomalies.csv.</p>' % a.z)
    if len(culprit):
        cu = pd.DataFrame({'подсистема': [subsystem(c) for c in culprit.index], 'признак': [human(c) for c in culprit.index], 'циклов': culprit.values})
        sec.append('<h3>Чаще всего выбивается</h3>' + table(cu, 30))
    sec.append('<h3>Самые сильные (top 50)</h3>' + table(pd.DataFrame(an_rows), 50))

    # остановы
    sec.append('<h2>Остановы и предвестники</h2><p class="muted">Паузы между циклами > %g мин (самые длинные сверху). Для каждой — какие признаки отклонялись от нормы в 50 рабочих циклах перед ней. Признак, который повторяется перед многими остановами, — кандидат в предвестники отказа.</p>' % a.gap_min)
    pc = pre_counts[pre_counts > 0].sort_values(ascending=False)
    if len(pc):
        sec.append('<h3>Повторяющиеся предвестники</h3>' + table(pd.DataFrame({'признак': [human(c) for c in pc.index], 'перед скольким остановами': pc.values.astype(int)}), 20))
    sec.append(table(S, 300))

    # датчики
    sec.append('<h2>Здоровье датчиков</h2>' + table(SH))

    # норма
    qq = q[q.index.isin(eng)].copy()
    sec.append('<h2>Нормальная работа (базовый период)</h2><p class="muted">Инженерные признаки; все сырые каналы — normal_profile.csv.</p>' +
               table(qq.reset_index(drop=True)[['подсистема', 'описание', 'p1', 'p5', 'median', 'p95', 'p99']], 400))

    css = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#e3e3e3;--card:#f6f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2c2e33;--card:#1d1f23}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:24px}h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:4px}.muted{color:var(--mut)}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{background:var(--card);padding:12px 16px;border-radius:8px;min-width:150px}
.kpi b{display:block;font-size:26px}.kpi span{color:var(--mut)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:12.5px;width:100%}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{background:var(--card);position:sticky;top:0}
""" + __import__('diagnose').CSS + """
.toc{margin:16px 0;line-height:2}.toc a{color:#1f77b4;text-decoration:none;white-space:nowrap}h3{margin-top:24px}"""
    body = '\n'.join(sec)
    heads = re.findall(r'<h2>(.*?)</h2>', body)
    for k, h in enumerate(heads):
        body = body.replace('<h2>%s</h2>' % h, '<h2 id="h%d">%s</h2>' % (k, h), 1)
    toc = '<nav class="toc">' + ' · '.join('<a href="#h%d">%s</a>' % (k, h) for k, h in enumerate(heads)) + '</nav>'
    body = body.replace('<!--TOC-->', toc, 1)
    html = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Отчёт пресса</title><style>%s</style></head><body>%s</body></html>' % (css, body)
    with open(os.path.join(a.out, 'report.html'), 'w', encoding='utf-8') as fh:
        fh.write(html)


if __name__ == '__main__':
    main()
