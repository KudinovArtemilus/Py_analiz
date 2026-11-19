# -*- coding: utf-8 -*-
"""
Диагностика по узлам пресса простым языком.

Для каждого узла: светофор (НОРМА / НАБЛЮДАТЬ / ПРОБЛЕМА), что видно в цифрах, вероятная причина,
что проверить на месте, прогноз (через сколько дней дойдёт до порога, если тренд сохранится).

Три способа найти проблему:
  1. Сравнение с нормой — текущий уровень против базового периода.
  2. Сравнение с «соседями» — 6 одинаковых насосов, 12 регуляторов и т.п. Если один заметно хуже
     остальных, это видно даже на одном дне данных.
  3. Частота выбросов — как часто узел «срывается» в последних циклах против базы.

Пороги в LIMITS — уточните по документации пресса / у механиков.
"""
import re
import numpy as np
import pandas as pd

# ====== Абсолютные пороги (УТОЧНИТЕ по паспорту пресса) ======
LIMITS = {
    'dieclamp_warn': 110.0,   # бар — давление зажима штампа, ниже — «наблюдать»
    'dieclamp_alarm': 100.0,  # бар — ниже — «проблема»
    'oil_temp_warn': 50.0,    # °C
    'oil_temp_alarm': 55.0,
    'hp_min_warn': 265.0,     # бар — минимум высокого давления за цикл
    'hp_min_alarm': 255.0,
}

OK, WARN, ALARM = 0, 1, 2
STATUS_TXT = {OK: 'НОРМА', WARN: 'НАБЛЮДАТЬ', ALARM: 'ПРОБЛЕМА'}


def I(feat, label, bad='up', warn_rel=0.15, alarm_rel=0.30, abs_warn=None, abs_alarm=None, unit=''):
    return dict(feat=feat, label=label, bad=bad, warn_rel=warn_rel, alarm_rel=alarm_rel, abs_warn=abs_warn, abs_alarm=abs_alarm, unit=unit)


def build_units():
    U = []
    for k in range(1, 7):
        U.append(dict(
            name='Насос / привод MD%d' % k, group='Насосы',
            ind=[I('casedrain_md%d_mean' % k, 'давление дренажа корпуса', 'up', 0.20, 0.50, unit='бар')],
            peer=('casedrain', 'casedrain_md%d_mean' % k),
            cause='Рост дренажа = растут внутренние утечки насоса: износ поршневой группы, распределителя, подшипников.',
            check='Замерить расход дренажа (мерной ёмкостью/расходомером), потрогать/пирометр — корпус горячее остальных? '
                  'Вибрация, шум. Проба масла на металл.'))
    for a in range(1, 7):
        for b in (1, 2):
            p = 'pump%d%d' % (a, b)
            U.append(dict(
                name='Регулятор насоса %d.%d' % (a, b), group='Регуляторы насосов',
                ind=[I(p + '_pivot_err_active', 'ошибка угла наклона под нагрузкой', 'up', 0.25, 0.60),
                     I(p + '_pivot_lag', 'запаздывание угла, ×10 мс', 'up', 0.30, 0.80),
                     I(p + '_valve_err_rms', 'ошибка клапана управления', 'up', 0.30, 0.80)],
                peer=('pivot', p + '_pivot_err_active'),
                cause='Насос не успевает за заданием: загрязнение/износ пропорционального клапана регулятора, люфт люльки, падение давления управления.',
                check='Записать задание/факт угла на осциллограф. Фильтр линии управления, чистота масла (ISO 4406). '
                      'Сравнить с соседним насосом той же пары.'))
    servo = [('v12Y102', 'Главный клапан ползуна 12Y102'), ('v12Y069_21', 'Клапан ползуна 12Y069.21'), ('v12Y069_22', 'Клапан ползуна 12Y069.22'),
             ('v20Y019', 'Клапан выталкивателя 20Y019'), ('v31Y102', 'Клапан подушки 31Y102'),
             ('vCCylL_Y22_2', 'Клапан центр. цилиндра лев. Y22.2'), ('vCCylR_Y22_1', 'Клапан центр. цилиндра прав. Y22.1'),
             ('vLCylL_Y21_2', 'Клапан нижн. цилиндра лев. Y21.2'), ('vLCylR_Y21_1', 'Клапан нижн. цилиндра прав. Y21.1'),
             ('vMB_Y1', 'Клапан стола MB Y1'), ('vMB_Y2', 'Клапан стола MB Y2')]
    for s, nm in servo:
        U.append(dict(
            name=nm, group='Сервоклапаны',
            ind=[I(s + '_err_rms', 'ошибка задание–золотник (RMS)', 'up', 0.30, 0.80),
                 I(s + '_lag', 'запаздывание, ×10 мс', 'up', 0.50, 1.00)],
            spike=(s + '_err_absmax', 'выбросы ошибки (золотник не дошёл / рывок)'),
            cause='Золотник не успевает за заданием или «рвётся»: загрязнение, износ втулки, проблемы датчика положения или питания клапана.',
            check='Фильтр пилотной линии, разъём и кабель датчика положения золотника, напряжение питания. '
                  'Если выбросы учащаются — клапан на стенд/замену в ближайший ППР.'))
    U.append(dict(
        name='Зажим штампа (контуры 1 и 2)', group='Зажим штампа',
        ind=[I('dieclamp_c1_min', 'давление зажима, контур 1, минимум', 'down', 0.15, 0.25, LIMITS['dieclamp_warn'], LIMITS['dieclamp_alarm'], 'бар'),
             I('dieclamp_c2_min', 'давление зажима, контур 2, минимум', 'down', 0.15, 0.25, LIMITS['dieclamp_warn'], LIMITS['dieclamp_alarm'], 'бар')],
        clamp=True,
        cause='Давление в замкнутом контуре зажима падает между перезажимами: утечка через уплотнения цилиндров зажима или обратный клапан; '
              'на холоде (ночью) падает дополнительно.',
        check='Тест на падение давления на остановленном прессе (зафиксировать давление и время). Осмотр уплотнений цилиндров зажима, '
              'обратного клапана контура. Проверить, срабатывает ли автоматическая подкачка (reload).'))
    U.append(dict(
        name='Аккумуляторы RV1 / RV3', group='Гидростанция',
        ind=[I('accu_rv1_min', 'аккумулятор RV1, минимум', 'down', 0.05, 0.10, unit='бар'),
             I('accu_rv3_min', 'аккумулятор RV3, минимум', 'down', 0.05, 0.10, unit='бар')],
        cause='Падение минимального давления аккумулятора — потеря азота (предзаправки) или утечка.',
        check='Проверить давление предзаправки азотом, герметичность.'))
    U.append(dict(
        name='Гидростанция: давление', group='Гидростанция',
        ind=[I('hp_min', 'высокое давление, минимум', 'down', 0.03, 0.06, LIMITS['hp_min_warn'], LIMITS['hp_min_alarm'], 'бар'),
             I('hp_drop', 'просадка давления за цикл', 'up', 0.25, 0.60, unit='бар'),
             I('hp_recover_s', 'время восстановления давления', 'up', 0.40, 1.00, unit='с')],
        cause='Насосная станция хуже держит давление: износ насосов, аккумуляторы, перепускной/предохранительный клапан.',
        check='Сопоставить с дренажами насосов и аккумуляторами (выше). Проверить настройку предохранительных клапанов.'))
    U.append(dict(
        name='Масло: температура и уровень', group='Гидростанция',
        ind=[I('temp_bottom_mean', 'температура масла, нижний бак', 'up', 0.10, 0.20, LIMITS['oil_temp_warn'], LIMITS['oil_temp_alarm'], '°C'),
             I('temp_upper_mean', 'температура масла, верхний бак', 'up', 0.10, 0.20, LIMITS['oil_temp_warn'], LIMITS['oil_temp_alarm'], '°C'),
             I('level_bottom_mean', 'уровень, нижний бак', 'down', 0.05, 0.10),
             I('level_upper_mean', 'уровень, верхний бак', 'down', 0.08, 0.15)],
        cause='Рост температуры при той же работе — охлаждение или внутренние утечки (греют масло). Падение уровня — внешняя утечка.',
        check='Теплообменник/вентиляторы, клапан охлаждения 02-M073.1. Осмотр на подтёки.'))
    U.append(dict(
        name='Ползун: направляющие и уплотнения', group='Ползун',
        ind=[I('lr_dev_absmax_mm', 'перекос лево–право, макс.', 'up', 0.25, 0.60, unit='мм'),
             I('slide_drift_top_mm_s', 'сползание ползуна в покое', 'both', 1.0, 3.0, unit='мм/с'),
             I('force_rise_s', 'время нарастания усилия', 'up', 0.20, 0.50, unit='с')],
        cause='Перекос растёт — износ направляющих/планок, разбаланс. Ползун сползает в покое — утечка через уплотнения главного цилиндра или клапаны.',
        check='Зазоры направляющих, параллельность. Тест удержания ползуна при выключенных насосах.'))
    for side, sn in (('Left', 'левый'), ('Right', 'правый')):
        U.append(dict(
            name='Центрирующий цилиндр %s' % sn, group='Цилиндры',
            ind=[I('CCyl%s_posdev_rms' % side, 'ошибка позиционирования', 'up', 0.30, 0.80)],
            cause='Цилиндр хуже держит позицию: утечки, клапан, датчик положения.', check='Датчик положения, уплотнения, клапан.'))
        U.append(dict(
            name='Нижний цилиндр %s' % sn, group='Цилиндры',
            ind=[I('LCyl%s_posdev_rms' % side, 'ошибка позиционирования', 'up', 0.30, 0.80)],
            cause='Цилиндр хуже держит позицию: утечки, клапан, датчик положения.', check='Датчик положения, уплотнения, клапан.'))
    U.append(dict(
        name='Частотник / двигатели насосов', group='Привод',
        ind=[I('fc_speed_err_rms', 'ошибка оборотов задание–факт', 'up', 0.30, 0.80)],
        cause='Двигатель не держит обороты — перегруз (насос тяжелее крутится) или проблема привода.',
        check='Ток двигателей, журнал ошибок частотника, муфта.'))
    return U


# ---------------------------------------------------------------------------------------------
def die_change_mask(ds):
    """True там, где цикл — первый после замены штампа: счётчик ходов упал минимум на 50 и больше чем вдвое."""
    ds = pd.Series(np.asarray(ds, dtype=float))
    prev = ds.shift(1)
    return ((prev - ds) > 50) & (ds < 0.5 * prev)


def _robust_slope(t, y):
    """Наклон по Тейлу–Сену (медиана попарных наклонов). t в часах."""
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 6:
        return np.nan
    if len(t) > 150:
        idx = np.linspace(0, len(t) - 1, 150).astype(int); t, y = t[idx], y[idx]
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 0
    return float(np.median((y[j] - y[i])[ok] / dt[ok])) if ok.any() else np.nan


def _fmt(v, unit=''):
    if v is None or not np.isfinite(v):
        return '—'
    a = abs(v)
    s = '%.0f' % v if a >= 100 else ('%.1f' % v if a >= 10 else ('%.2f' % v if a >= 0.1 else '%.3f' % v))
    return s + (' ' + unit if unit else '')


def evaluate(P, B, sig, span_h, T=None):
    """P — рабочие циклы (по времени), B — базовый период, sig — робастные σ (Series).
    T — оцениваемое окно (напр. циклы одних суток). По умолчанию — последние 15% циклов."""
    units = build_units()
    window = T is not None
    if T is None:
        T = P.tail(max(5, int(len(P) * 0.15)))
    th = (P['t_start'] - P['t_start'].min()).dt.total_seconds().values / 3600.0
    peer_cache = {}

    def peer_stats(kind):
        if kind in peer_cache:
            return peer_cache[kind]
        pat = {'casedrain': r'^casedrain_md\d_mean$', 'pivot': r'^pump\d\d_pivot_err_active$'}[kind]
        cols = [c for c in P.columns if re.match(pat, c) and P[c].notna().sum() > 3]
        cur = T[cols].median()
        peer_cache[kind] = cur
        return cur

    out = []
    notes = []
    for u in units:
        status, facts, forecast = OK, [], []
        have = False
        for ind in u['ind']:
            f = ind['feat']
            if f not in P.columns or P[f].notna().sum() < 5:
                continue
            have = True
            base = B[f].median(); cur = T[f].median()
            s = sig.get(f, np.nan) if hasattr(sig, 'get') else np.nan
            if not np.isfinite(base) or not np.isfinite(cur):
                continue
            rel = (cur - base) / (abs(base) + 1e-9)
            z = (cur - base) / s if s and np.isfinite(s) and s > 0 else 0.0
            sign = {'up': 1, 'down': -1, 'both': 0}[ind['bad']]
            badz = z * sign if sign else abs(z)
            badrel = rel * sign if sign else abs(rel)
            st = OK
            # абсолютные пороги (по минимуму за последний период для 'down')
            worst = T[f].min() if ind['bad'] == 'down' else T[f].max()
            if ind['abs_alarm'] is not None and ((sign < 0 and worst < ind['abs_alarm']) or (sign > 0 and worst > ind['abs_alarm'])):
                st = ALARM
            elif ind['abs_warn'] is not None and ((sign < 0 and worst < ind['abs_warn']) or (sign > 0 and worst > ind['abs_warn'])):
                st = max(st, WARN)
            if badz > 8 and badrel > ind['alarm_rel']:
                st = ALARM
            elif badz > 4 and badrel > ind['warn_rel']:
                st = max(st, WARN)
            # тренд и прогноз
            slope = _robust_slope(th, P[f].values.astype(float))
            rho = pd.Series(th).corr(P[f].reset_index(drop=True), method='spearman')
            eta = None
            target = ind['abs_alarm'] if ind['abs_alarm'] is not None else (base * (1 + sign * ind['alarm_rel']) if sign else None)
            if not u.get('clamp') and target is not None and np.isfinite(slope) and slope != 0 and abs(rho) > 0.4 and sign * slope > 0:
                eta_h = (target - cur) / slope
                if eta_h > 0:
                    eta = eta_h / 24
                    if span_h < 72 and abs(badz) < 3:
                        eta = None          # короткие данные + нет заметного отклонения — прогноз не показываем
                    elif eta < 14 and st < WARN and span_h >= 72:
                        st = WARN
            line = '%s: сейчас %s, норма %s (%+.0f%%)' % (ind['label'], _fmt(cur, ind['unit']), _fmt(base, ind['unit']), rel * 100)
            if ind['bad'] == 'down' and np.isfinite(worst):
                line += ', минимум за последний период %s' % _fmt(worst, ind['unit'])
            if st > OK or abs(badz) > 3:
                facts.append((st, line))
            if eta is not None:
                forecast.append('«%s» дойдёт до %s примерно через %s (если тренд сохранится)' % (
                    ind['label'], _fmt(target, ind['unit']), ('%.0f дн.' % eta) if eta >= 1 else ('%.0f ч' % (eta * 24))))
            status = max(status, st)

        # сравнение с соседями
        if u.get('peer'):
            kind, f = u['peer']
            if f in P.columns:
                cur = peer_stats(kind)
                if f in cur.index and len(cur) >= 3 and np.isfinite(cur[f]):
                    others = cur.drop(f).dropna()
                    om = others.median()
                    if om > 0:
                        ratio = cur[f] / om
                        if kind == 'casedrain' and ratio < 0.25:
                            status = max(status, WARN)
                            facts.append((WARN, 'дренаж %s, у остальных насосов ~%s — в %.0f раз меньше. Насос не может '
                                                'не иметь дренажа: вероятнее, неисправен датчик/канал или насос не работает'
                                          % (_fmt(cur[f], 'бар'), _fmt(om, 'бар'), 1 / max(ratio, 1e-3))))
                            u = dict(u, cause='Подозрение на ДАТЧИК, а не на насос: значение почти нулевое на фоне остальных.',
                                     check='Проверить датчик давления дренажа MD (0 бар при работающем насосе?), кабель, масштабирование канала в ПЛК; убедиться, что насос работает.')
                        elif ratio > 1.5:
                            partner = None
                            m_ = re.match(r'pump(\d)(\d)_', f)
                            if m_:
                                partner = 'pump%s%s_%s' % (m_.group(1), '2' if m_.group(2) == '1' else '1', f.split('_', 1)[1])
                            pr = cur[f] / cur[partner] if partner in cur.index and cur.get(partner, 0) > 0 else None
                            if pr is not None and 0.8 <= pr <= 1.25:
                                notes.append('Пара насосов %s.1/%s.2: оба в %.1f раза «хуже» остальных и одинаковы между собой (%s и %s). '
                                             'Так ведёт себя вся пара — скорее это особенность её контура/режима, а не износ одного насоса. '
                                             'Тревога, если один из пары начнёт отличаться от второго.' % ((m_.group(1), m_.group(1), ratio) + tuple(sorted([_fmt(cur[f]), _fmt(cur[partner])]))))
                            else:
                                st = ALARM if ratio > 2.5 else WARN
                                status = max(status, st)
                                txt = 'хуже соседних в %.1f раза (%s против ~%s у остальных)' % (ratio, _fmt(cur[f]), _fmt(om))
                                if pr is not None:
                                    txt += ', и хуже напарника по паре в %.1f раза' % pr
                                facts.append((st, txt))

        # выбросы (сервоклапаны)
        if u.get('spike'):
            f, lab = u['spike']
            if f in P.columns and P[f].notna().sum() > 5:
                med = P[f].median()
                thr = max(3 * med, med + 6 * (sig.get(f, 0) if hasattr(sig, 'get') else 0))
                W = T if window else P          # в суточном отчёте считаем срывы только за эти сутки
                hits = W[f] > thr
                n_all = int(hits.sum())
                if n_all > 0:
                    rate = n_all / len(W)
                    st = WARN if rate >= 0.02 else OK
                    status = max(status, st)
                    times = ', '.join(W.loc[hits, 't_start'].dt.strftime('%d.%m %H:%M').head(6))
                    facts.append((st, '%s: %d раз(а) из %d циклов (%.0f%%), порог %s, обычно %s. Когда: %s' % (
                        lab, n_all, len(W), rate * 100, _fmt(thr), _fmt(med), times)))

        # зажим штампа: падение с момента перезажима
        if u.get('clamp') and 'die_strokes' in P.columns:
            ds = P['die_strokes'].values
            resets = np.where(die_change_mask(ds).values)[0]
            start = resets[-1] if len(resets) else 0
            seg = P.iloc[start:]
            for f, nm in (('dieclamp_c1_min', 'контур 1'), ('dieclamp_c2_min', 'контур 2')):
                if f in P.columns and len(seg) >= 4:
                    t0 = (seg['t_start'] - seg['t_start'].iloc[0]).dt.total_seconds().values / 3600
                    sl = _robust_slope(t0, seg[f].values.astype(float))
                    if np.isfinite(sl) and sl < 0:
                        facts.append((OK, '%s: после последнего перезажима падает ~%.1f бар/ч' % (nm, -sl)))
            # по каждому периоду между заменами штампа: с какого давления начали, минимум, через сколько часов ниже порогов
            bounds = [0] + list(resets) + [len(P)]
            per, hrs_warn, hrs_alarm = [], [], []
            for a_, b_ in zip(bounds[:-1], bounds[1:]):
                g = P.iloc[a_:b_]
                if len(g) < 3 or 'dieclamp_c2_min' not in g:
                    continue
                th0 = (g['t_start'] - g['t_start'].iloc[0]).dt.total_seconds() / 3600
                txt = '%s–%s: старт %s, минимум %s' % (g['t_start'].iloc[0].strftime('%d.%m %H:%M'), g['t_start'].iloc[-1].strftime('%d.%m %H:%M'),
                                                       _fmt(g['dieclamp_c2_min'].iloc[:2].median(), 'бар'), _fmt(g['dieclamp_c2_min'].min(), 'бар'))
                for thr, acc, nm in ((LIMITS['dieclamp_warn'], hrs_warn, '%g'), (LIMITS['dieclamp_alarm'], hrs_alarm, '%g')):
                    below = g['dieclamp_c2_min'] < thr
                    if below.any():
                        h_ = float(th0[below.values].iloc[0]); acc.append(h_)
                        txt += ', ниже %g бар через %.0f ч после перезажима' % (thr, h_)
                per.append(txt)
            if per:
                facts.append((OK, 'контур 2 по периодам между заменами штампа: ' + '; '.join(per)))
            if hrs_alarm:
                forecast.append('после перезажима штампа контур 2 опускается ниже %g бар примерно через %.0f ч работы (по %d периодам). '
                                'Если штамп стоит дольше — давление уходит ниже порога' % (LIMITS['dieclamp_alarm'], np.median(hrs_alarm), len(hrs_alarm)))
            elif hrs_warn:
                forecast.append('после перезажима штампа контур 2 опускается ниже %g бар примерно через %.0f ч работы' % (LIMITS['dieclamp_warn'], np.median(hrs_warn)))
            facts.append((OK, 'пороги %g/%g бар приняты условно — уточните допустимый минимум давления зажима по паспорту пресса' % (LIMITS['dieclamp_warn'], LIMITS['dieclamp_alarm'])))
            if 'dieclamp_c2_min' in P and 'dieclamp_c1_min' in P:
                d = (T['dieclamp_c1_min'] - T['dieclamp_c2_min']).median()
                if d > 5:
                    facts.append((OK, 'контур 2 ниже контура 1 на ~%.0f бар — утечка, вероятнее, в контуре 2' % d))

        if not have and not facts:
            continue
        facts.sort(key=lambda x: -x[0])
        out.append(dict(name=u['name'], group=u['group'], status=status, facts=[f for _, f in facts], forecast=forecast,
                        cause=u['cause'], check=u['check']))
    out.sort(key=lambda r: (-r['status'], r['group'], r['name']))
    return out, list(dict.fromkeys(notes))


def stops_summary(df, gap_thr_min):
    """Остановы с привязкой к замене штампа."""
    rows = []
    t = df['t_start'].reset_index(drop=True)
    ds = df['die_strokes'].reset_index(drop=True) if 'die_strokes' in df else None
    g = t.diff().dt.total_seconds() / 60
    for i in np.where(g > gap_thr_min)[0]:
        why = 'причина не видна в данных'
        if ds is not None and bool(die_change_mask(ds.iloc[i - 1:i + 1]).iloc[-1]):
            why = 'ЗАМЕНА ШТАМПА (счётчик ходов %d → %d)' % (ds.iloc[i - 1], ds.iloc[i])
        rows.append('%s → %s (%.1f ч): %s' % (t.iloc[i - 1].strftime('%d.%m %H:%M'), t.iloc[i].strftime('%d.%m %H:%M'), g.iloc[i] / 60, why))
    return rows


def to_text(diag, op, period, n, stops, short_data, notes=()):
    L = ['ДИАГНОЗ ПРЕССА %s — %s, циклов: %d' % (op, period, n), '']
    if short_data:
        L += ['ВНИМАНИЕ: данных меньше недели — прогнозы и тренды предварительные; сравнение «насос с насосом» надёжно.', '']
    for s in (ALARM, WARN):
        items = [d for d in diag if d['status'] == s]
        L.append('%s: %s' % (STATUS_TXT[s], ', '.join(d['name'] for d in items) if items else 'нет'))
    L.append('НОРМА: %d узлов' % sum(d['status'] == OK for d in diag))
    L.append('')
    for d in diag:
        if d['status'] == OK:
            continue
        L.append('[%s] %s' % (STATUS_TXT[d['status']], d['name']))
        for f in d['facts']:
            L.append('   • ' + f)
        for f in d['forecast']:
            L.append('   ⏱ ' + f)
        L.append('   Причина: ' + d['cause'])
        L.append('   Проверить: ' + d['check'])
        L.append('')
    if notes:
        L.append('НАБЛЮДЕНИЯ БЕЗ ТРЕВОГИ:')
        L += ['   • ' + x for x in notes]
        L.append('')
    if stops:
        L.append('ОСТАНОВЫ:')
        L += ['   • ' + s for s in stops]
    return '\n'.join(L)


def to_html(diag, stops, short_data, notes=()):
    col = {ALARM: '#c62828', WARN: '#ef6c00', OK: '#2e7d32'}
    bg = {ALARM: 'rgba(198,40,40,.08)', WARN: 'rgba(239,108,0,.08)', OK: 'rgba(46,125,50,.06)'}
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    h = []
    na, nw, no = (sum(d['status'] == s for d in diag) for s in (ALARM, WARN, OK))
    h.append('<div class="verdict"><span style="color:%s">● Проблема: <b>%d</b></span> <span style="color:%s">● Наблюдать: <b>%d</b></span> '
             '<span style="color:%s">● Норма: <b>%d</b></span></div>' % (col[ALARM], na, col[WARN], nw, col[OK], no))
    if short_data:
        h.append('<p class="muted">Данных меньше недели: прогнозы и тренды предварительные. Сравнение одинаковых узлов между собой надёжно уже сейчас.</p>')
    for d in diag:
        if d['status'] == OK:
            continue
        h.append('<div class="dcard" style="border-left:5px solid %s;background:%s"><div class="dh"><b>%s</b> <span class="tag" style="background:%s">%s</span> <span class="muted">%s</span></div>'
                 % (col[d['status']], bg[d['status']], esc(d['name']), col[d['status']], STATUS_TXT[d['status']], esc(d['group'])))
        h.append('<ul>' + ''.join('<li>%s</li>' % esc(f) for f in d['facts']) + ''.join('<li>⏱ %s</li>' % esc(f) for f in d['forecast']) + '</ul>')
        h.append('<p><b>Вероятная причина:</b> %s</p><p><b>Что проверить:</b> %s</p></div>' % (esc(d['cause']), esc(d['check'])))
    oks = [d['name'] for d in diag if d['status'] == OK]
    if oks:
        h.append('<details><summary>В норме (%d): показать</summary><p>%s</p></details>' % (len(oks), esc(', '.join(oks))))
    if notes:
        h.append('<h3>Наблюдения без тревоги</h3><ul>' + ''.join('<li>%s</li>' % esc(x) for x in notes) + '</ul>')
    if stops:
        h.append('<h3>Остановы</h3><ul>' + ''.join('<li>%s</li>' % esc(s) for s in stops) + '</ul>')
    return '\n'.join(h)


CSS = """.verdict{font-size:18px;display:flex;gap:24px;flex-wrap:wrap;margin:8px 0 12px}
.dcard{padding:10px 14px;margin:10px 0;border-radius:6px}.dcard ul{margin:6px 0 6px 18px;padding:0}.dcard p{margin:4px 0}
.tag{color:#fff;font-size:12px;padding:1px 8px;border-radius:10px;margin-left:6px}"""