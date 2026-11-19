# -*- coding: utf-8 -*-
"""
Шаг 1. Проходит по всем .shp файлам (рекурсивно), из каждого файла (= один цикл пресса)
считает признаки и складывает их в таблицу features.parquet (или .csv.gz).

Запуск:
    python extract_features.py  D:\\data\\shp   -o features   -j 8

Повторный запуск дописывает только новые файлы (уже обработанные пропускаются).
"""
import argparse, os, re, sys, time, datetime as dt, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shpreader import read_shp

DT = 0.01                 # шаг записи 10 мс (проверено по clock pulse 10 Гц)
PRESS_KN = 1000.0         # порог "идёт прессование", кН
SKIP_NAME = re.compile(r'spare|^off$|^default$|^$', re.I)


def safe(name):
    return re.sub(r'[^0-9A-Za-zА-Яа-я]+', '_', name).strip('_')


def file_time(path, header):
    """Время начала: из имени (..._YYYYMMDD_HHMMSS.shp — местное), иначе FILETIME из заголовка (UTC)."""
    m = re.search(r'(\d{8})_(\d{6})', os.path.basename(path))
    if m:
        return dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
    ft = int.from_bytes(header[8:16], 'little')
    return dt.datetime(1601, 1, 1) + dt.timedelta(microseconds=ft // 10)


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else np.nan


def lag_samples(sp, pv, maxlag=60):
    """Запаздывание факта от задания (в отсчётах) по максимуму взаимной корреляции."""
    sp = sp - sp.mean(); pv = pv - pv.mean()
    if sp.std() < 1e-6 or pv.std() < 1e-6:
        return np.nan
    best, bl = -np.inf, 0
    for L in range(0, maxlag + 1):
        c = np.dot(sp[:len(sp) - L], pv[L:])
        if c > best:
            best, bl = c, L
    return float(bl)


def features_for_file(path):
    r = read_shp(path)
    A, D = r['ana'], r['dig']
    an, dn = r['ana_names'], r['dig_names']
    n = A.shape[0]
    ix = {}
    for j, nm in enumerate(an):
        ix.setdefault(nm, j)
    def ch(name):
        j = ix.get(name)
        if j is None:
            return None
        x = A[:, j].astype(np.float64)
        x[~np.isfinite(x) | (np.abs(x) >= 1e12)] = np.nan
        return x
    dix = {}
    for j, nm in enumerate(dn):
        dix.setdefault(nm, j)
    def dg(name):
        j = dix.get(name)
        return D[:, j] if j is not None else None

    f = {}
    f['file'] = os.path.abspath(path)
    bn = os.path.basename(path)
    f['op'] = (re.search(r'(OP\d+)', bn) or [None, ''])[1]
    m = re.match(r'(\d+)\.(\d+)\.(\d+)_', bn)   # 26.142380.066563_OP1_... -> год?.заготовка.плавка
    if m:
        f['fn_prefix'] = int(m.group(1)); f['fn_piece'] = int(m.group(2)); f['fn_heat'] = int(m.group(3))
    t0 = file_time(path, r['header'])
    f['t_start'] = t0
    f['n_samples'] = n
    f['cycle_s'] = n * DT

    # ---------- мета ----------
    for nm, key in [('PartTracking_PieceId_Consecutiv', 'piece_id'), ('PartTracking_PieceId_HeatID', 'heat_id'),
                    ('Die.StokesSinceChange', 'die_strokes'), ('Which OP is Pressing', 'which_op_max')]:
        c = ch(nm)
        if c is not None:
            f[key] = float(np.nanmax(c))

    # ---------- слайд (ползун) ----------
    pos, F = ch('slide:average val.encoder [mm]'), ch('slide:total force [kN]')
    if pos is not None and F is not None:
        v = np.gradient(pos, DT)                      # мм/с, отрицательная = вниз
        press = F > PRESS_KN
        f['force_max_kN'] = float(np.nanmax(F))
        f['press_time_s'] = float(press.sum() * DT)
        f['force_impulse_kNs'] = float(np.nansum(F[press]) * DT)
        dpos = -np.diff(pos)
        f['work_kJ'] = float(np.nansum((F[1:] * dpos)[press[1:] & (dpos > 0)]) / 1000.0)  # кН*мм/1000 = кДж
        f['pos_top_mm'] = float(np.nanmax(pos))
        f['pos_bottom_mm'] = float(np.nanmin(pos))
        f['stroke_mm'] = f['pos_top_mm'] - f['pos_bottom_mm']
        f['v_down_max'] = float(-np.nanmin(v))
        f['v_up_max'] = float(np.nanmax(v))
        moving = np.abs(v) > 5
        f['slide_moving_s'] = float(moving.sum() * DT)
        if press.any():
            ip = np.where(press)[0]
            f['press_start_s'] = ip[0] * DT
            f['pos_at_force_max_mm'] = float(pos[np.nanargmax(F)])
            # время подъёма давления: от 10% до 90% максимума
            fm = f['force_max_kN']; i10 = np.argmax(F > 0.1 * fm); i90 = np.argmax(F > 0.9 * fm)
            f['force_rise_s'] = (i90 - i10) * DT
        # дрейф ползуна в покое наверху (утечки / уплотнения)
        top = (pos > f['pos_top_mm'] - 5) & (np.abs(v) < 2) & (~press)
        if top.sum() > 300:
            ti = np.where(top)[0]
            f['slide_drift_top_mm_s'] = float(np.polyfit(ti * DT, pos[ti], 1)[0])
    dev = ch('slide:deviation left-right [mm]')
    if dev is not None:
        f['lr_dev_absmax_mm'] = float(np.nanmax(np.abs(dev)))
        if F is not None and (F > PRESS_KN).any():
            f['lr_dev_press_mean_mm'] = float(np.nanmean(dev[F > PRESS_KN]))

    # ---------- гидростанция ----------
    hp = ch('hyd:high pressure [bar]')
    if hp is not None:
        f['hp_mean'] = float(np.nanmean(hp)); f['hp_min'] = float(np.nanmin(hp))
        f['hp_drop'] = float(np.nanpercentile(hp, 95) - np.nanmin(hp))
        if F is not None and (F > PRESS_KN).any():
            last = np.where(F > PRESS_KN)[0][-1]
            ref = np.nanpercentile(hp, 95)
            rec = np.where(hp[last:] >= ref - 2)[0]
            f['hp_recover_s'] = rec[0] * DT if rec.size else np.nan
    for nm, key in [('hyd:low pressure [bar]', 'lp'), ('hyd:temp bottom tank =03+44M0-B', 'temp_bottom'),
                    ('hyd:temp upper tank =03+93M0-B1', 'temp_upper'),
                    ('hyd:level bottom tank =03+44M0-', 'level_bottom'), ('hyd:level upper tank =03+93M0-B', 'level_upper'),
                    ('pressure accu ring valve1 [bar]', 'accu_rv1'), ('pressure accu ring valve3 [bar]', 'accu_rv3'),
                    ('DieCl: reload press. circuit 1', 'dieclamp_c1'), ('DieCl: reload press. circuit 2', 'dieclamp_c2')]:
        c = ch(nm)
        if c is not None:
            f[key + '_mean'] = float(np.nanmean(c)); f[key + '_min'] = float(np.nanmin(c)); f[key + '_max'] = float(np.nanmax(c))

    # ---------- приводы насосов: утечки (casedrain) и частотник ----------
    for k in range(1, 7):
        c = ch('Oildr: MD%d Casedrain Pressure' % k)
        if c is not None:
            f['casedrain_md%d_mean' % k] = float(np.nanmean(c))
            f['casedrain_md%d_p95' % k] = float(np.nanpercentile(c, 95))
    ns, ni = ch('Oildr:FrequenceConverter NSoll'), ch('Oildr:FrequenceConverter NIst')
    if ns is not None and ni is not None:
        f['fc_speed_err_rms'] = rms(ns - ni)

    # ---------- насосы 1.1..6.2: слежение задания/факта ----------
    for a in range(1, 7):
        for b in (1, 2):
            p = 'Pump %d.%d ' % (a, b)
            sv, av = ch(p + 'svValve'), ch(p + 'avValve')
            sp, ap = ch(p + 'setPivoting'), ch(p + 'actPivoting')
            pr = ch(p + 'actPressure')
            key = 'pump%d%d' % (a, b)
            if sv is not None and av is not None:
                f[key + '_valve_err_rms'] = rms(sv - av)
            if sp is not None and ap is not None:
                f[key + '_pivot_err_rms'] = rms(sp - ap)
                f[key + '_pivot_lag'] = lag_samples(sp, ap)
                act = sp > 20
                f[key + '_pivot_err_active'] = rms((sp - ap)[act]) if act.sum() > 20 else np.nan
            if pr is not None:
                f[key + '_press_max'] = float(np.nanmax(pr))

    # ---------- сервоклапаны: задание vs положение золотника ----------
    servo = [('slide:12Y102 set value [V]', 'slide:12Y102 pistion pos.[V]', 'v12Y102'),
             ('slide:12Y069.21 set value [V]', 'slide:12Y069.21 piston pos.[V]', 'v12Y069_21'),
             ('slide:12Y069.22 set value [V]', 'slide:12Y069.22 piston pos.[V]', 'v12Y069_22'),
             ('BEj:20Y019 set value [V]', 'BEj:20Y019 piston pos.[V]', 'v20Y019'),
             ('SCu:31Y102: set value [V]', 'SCu:31Y102:piston pos.[V]', 'v31Y102'),
             ('CCyl Left Prop Y22.2 U', 'CCyl Left Prop Y22.2 PinF', 'vCCylL_Y22_2'),
             ('CCyl Right Prop Y22.1 U', 'CCyl Right Prop Y22.1 PinF', 'vCCylR_Y22_1'),
             ('LCyl Left Prop Y21.2 U', 'LCyl Left Prop Y21.2 PinF', 'vLCylL_Y21_2'),
             ('LCyl Right Prop Y21.1 U', 'LCyl Right Prop Y21.1 PinF', 'vLCylR_Y21_1'),
             ('MB moValve_Y1SetVal', 'MB moValve_Y1ActVal', 'vMB_Y1'),
             ('MB moValve_Y2SetVal', 'MB moValve_Y2ActVal', 'vMB_Y2')]
    for s, a_, key in servo:
        sp, pv = ch(s), ch(a_)
        if sp is not None and pv is not None:
            f[key + '_err_rms'] = rms(sp - pv)
            f[key + '_err_absmax'] = float(np.nanmax(np.abs(sp - pv)))
            f[key + '_lag'] = lag_samples(sp, pv)

    # ---------- центрирующие / нижние цилиндры, MB ----------
    for side in ('Left', 'Right'):
        for cyl in ('CCyl', 'LCyl'):
            c = ch('%s %s lvPosCtrlDev' % (cyl, side))
            if c is not None:
                f['%s%s_posdev_rms' % (cyl, side)] = rms(c)
                f['%s%s_posdev_absmax' % (cyl, side)] = float(np.nanmax(np.abs(c)))
            c = ch('%s %s Force' % (cyl, side))
            if c is not None:
                f['%s%s_force_absmax' % (cyl, side)] = float(np.nanmax(np.abs(c)))
            c = ch('%s %s Speed' % (cyl, side))
            if c is not None:
                f['%s%s_speed_absmax' % (cyl, side)] = float(np.nanmax(np.abs(c)))
    mv, ms = ch('MB ActVel'), ch('MB SetDynVel')
    if mv is not None and ms is not None:
        moving = np.abs(mv) > 5
        f['MB_vel_err_rms'] = rms((ms - mv)[moving]) if moving.sum() > 20 else np.nan
    for nm, key in [('bed ejector speed [mm/s]', 'bej_speed_max'), ('slide ejector speed [mm/s]', 'sej_speed_max'),
                    ('BEj:netto force [kN]', 'bej_force_max'), ('SCu:netto force [kN]', 'scu_force_max')]:
        c = ch(nm)
        if c is not None:
            f[key] = float(np.nanmax(c))

    # ---------- общие статистики по ВСЕМ аналоговым каналам ----------
    for j, nm in enumerate(an):
        if SKIP_NAME.search(nm):
            continue
        c = A[:, j].astype(np.float64)
        key = 'a%03d_%s' % (j, safe(nm))[:60]
        fin = np.isfinite(c) & (np.abs(c) < 1e12)   # |x| >= 1e12 — мусор в записи, считаем как сбой
        if not fin.all():
            f[key + '__nan'] = float((~fin).mean())
            c = c[fin]
            if c.size == 0:
                continue
        f[key + '__mean'] = float(c.mean()); f[key + '__std'] = float(c.std())
        f[key + '__min'] = float(c.min()); f[key + '__max'] = float(c.max())

    # ---------- дискретные: число включений и доля времени ----------
    di = D.astype(np.int8)
    rises = (np.diff(di, axis=0) == 1).sum(axis=0)
    onfr = D.mean(axis=0)
    for j, nm in enumerate(dn):
        if SKIP_NAME.search(nm) or j == 0:
            continue
        key = 'd%03d_%s' % (j, safe(nm))[:60]
        f[key + '__on'] = float(onfr[j]); f[key + '__rises'] = int(rises[j])
    return f


FN_DT = re.compile(r'_(\d{8})_(\d{6})\.shp$', re.I)


def list_files(root, date_from=None, date_to=None, op=None):
    """Рекурсивно, в т.ч. D:\\press\\ГГГГ\\ММ\\ДД\\*.shp. Сортировка по времени из имени файла.
    Папки дней вне диапазона дат пропускаются, не заходя в них."""
    lo = date_from.replace('-', '') if date_from else None
    hi = date_to.replace('-', '') if date_to else None
    out = []
    for dp, dirs, fs in os.walk(root):
        # отсечение по структуре ГГГГ\ММ\ДД (если она есть)
        parts = re.findall(r'(\d{4})[\\/](\d{2})(?:[\\/](\d{2}))?$', dp)
        if parts and (lo or hi):
            y, mth, d = parts[-1]
            if d:
                day = y + mth + d
                if (lo and day < lo) or (hi and day > hi):
                    dirs[:] = []; continue
        dirs.sort()
        for x in fs:
            if not x.lower().endswith('.shp'):
                continue
            if op and ('_%s_' % op.upper()) not in x.upper():
                continue
            m = FN_DT.search(x)
            day = m.group(1) if m else None
            if day and ((lo and day < lo) or (hi and day > hi)):
                continue
            out.append(((m.group(1) + m.group(2)) if m else x, os.path.join(dp, x)))
    return [p for _, p in sorted(out)]


def _job(p):
    try:
        return features_for_file(p), None
    except Exception:
        return None, (p, traceback.format_exc(limit=2))


def save(df, out):
    try:
        df.to_parquet(out + '.parquet', index=False); return out + '.parquet'
    except Exception:
        df.to_csv(out + '.csv.gz', index=False); return out + '.csv.gz'


def load_existing(out):
    for ext, fn in (('.parquet', pd.read_parquet), ('.csv.gz', pd.read_csv)):
        if os.path.exists(out + ext):
            return fn(out + ext)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='папка с .shp (ищет рекурсивно)')
    ap.add_argument('-o', '--out', default='features', help='имя выходного файла без расширения')
    ap.add_argument('-j', '--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--limit', type=int, default=0, help='обработать только N файлов (для пробы)')
    ap.add_argument('--every', type=int, default=1, help='брать каждый N-й файл (быстрый обзор)')
    ap.add_argument('--from', dest='date_from', help='с даты ГГГГ-ММ-ДД (по имени файла / папкам)')
    ap.add_argument('--to', dest='date_to', help='по дату ГГГГ-ММ-ДД включительно')
    ap.add_argument('--op', help='только эта операция, напр. OP1')
    ap.add_argument('-v', '--verbose', action='store_true', help='печатать каждый обрабатываемый файл')
    a = ap.parse_args()

    files = list_files(a.root, a.date_from, a.date_to, a.op)[::a.every]
    old = load_existing(a.out)
    done = set(old['file']) if old is not None else set()
    todo = [p for p in files if os.path.abspath(p) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print('Найдено файлов: %d, уже обработано: %d, к обработке: %d' % (len(files), len(done), len(todo)))
    rows, errs, t0 = [], [], time.time()
    chunk = 500

    def flush():
        nonlocal old, rows
        if not rows:
            return
        new = pd.DataFrame(rows)
        old = new if old is None else pd.concat([old, new], ignore_index=True, sort=False)
        path = save(old, a.out); rows = []
        print('  сохранено -> %s (%d строк)' % (path, len(old)))

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_job, p): p for p in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            res, err = fu.result()
            if res: rows.append(res)
            if err: errs.append(err)
            if a.verbose:
                print('[%d/%d] %s  %s' % (k, len(todo), 'OK    ' if res else 'ОШИБКА', futs[fu]), flush=True)
            elif err:
                print('ОШИБКА чтения: %s' % futs[fu], flush=True)
            if k % 100 == 0 or k == len(todo):
                el = time.time() - t0
                print('--- %d/%d  %.1f файл/с  осталось ~%.0f мин' % (k, len(todo), k / el, (len(todo) - k) / (k / el) / 60), flush=True)
            if len(rows) >= chunk:
                flush()
    flush()
    if errs:
        with open(a.out + '_errors.txt', 'a', encoding='utf-8') as fh:
            for p, tb in errs:
                fh.write(p + '\n' + tb + '\n')
        print('Ошибок чтения: %d (см. %s_errors.txt)' % (len(errs), a.out))


if __name__ == '__main__':
    main()
