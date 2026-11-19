# -*- coding: utf-8 -*-
"""
Запуск всего конвейера одной кнопкой (PyCharm: правый клик -> Run 'run').
Настройки — ниже. Аргументы командной строки не нужны.
"""
import os, subprocess, sys, webbrowser

# ======================= НАСТРОЙКИ =======================
DATA = r'\\10.130.171.22\BufferShp'          # папка с данными (внутри ГГГГ\ММ\ДД\*.shp)
OPS = ['OP1', 'OP2']        # какие операции анализировать (у каждой — свой отчёт)
QUICK = True                # True  = быстрая проба: каждый EVERY-й файл (минуты)
                            # False = полный прогон всех файлов (можно прервать и запустить снова — продолжит)
EVERY = 20
DATE_FROM = None            # напр. '2026-09-01' или None = с начала
DATE_TO = None              # напр. '2026-09-30' или None = до конца
JOBS = 3                    # потоков чтения; для HDD 2-3, для SSD можно 6-8
SHOW_FILES = True           # True = печатать каждый обрабатываемый файл; False = только прогресс каждые 100 файлов
OUT_DIR = r'D:\press_shp_analysis\results'   # куда класть таблицы признаков и отчёты
FULL_REPORT = True          # отчёт по всем данным (report_OP1\report.html)
DAY_REPORT = 'last'         # отчёт за сутки: 'last' = последний день в данных, '2026-09-28' = конкретный день,
                            # 'all' = отдельный отчёт на каждый день, None = не делать
# =========================================================

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    print('\n>>>', ' '.join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit('Ошибка на шаге: %s' % ' '.join(args))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = 'quick_' if QUICK else ''
    for op in OPS:
        feats = os.path.join(OUT_DIR, 'features_%s%s' % (tag, op))
        rep = os.path.join(OUT_DIR, 'report_%s%s' % (tag, op))
        a = [os.path.join(HERE, 'extract_features.py'), DATA, '--op', op, '-o', feats, '-j', str(JOBS)]
        if QUICK: a += ['--every', str(EVERY)]
        if SHOW_FILES: a += ['--verbose']
        if DATE_FROM: a += ['--from', DATE_FROM]
        if DATE_TO: a += ['--to', DATE_TO]
        run(a)
        if not (os.path.exists(feats + '.parquet') or os.path.exists(feats + '.csv.gz')):
            print('Для %s файлов не найдено — пропускаю.' % op)
            continue
        src = feats + '.parquet' if os.path.exists(feats + '.parquet') else feats + '.csv.gz'
        if FULL_REPORT:
            b = [os.path.join(HERE, 'analyze.py'), src, '--op', op, '-o', rep]
            if DATE_FROM: b += ['--from', DATE_FROM]
            if DATE_TO: b += ['--to', DATE_TO]
            run(b)
            open_html(os.path.join(rep, 'report.html'))
        if DAY_REPORT:
            day_dir = os.path.join(OUT_DIR, 'daily')
            c = [os.path.join(HERE, 'daily.py'), src, '--op', op, '-o', day_dir]
            if DAY_REPORT == 'all':
                c += ['--all-days']
            elif DAY_REPORT != 'last':
                c += ['--day', DAY_REPORT]
            run(c)
            days = sorted(f for f in os.listdir(day_dir) if f.startswith('day_') and f.endswith('_%s.html' % op))
            if days and DAY_REPORT != 'all':
                want = [f for f in days if DAY_REPORT in f] if DAY_REPORT != 'last' else days[-1:]
                for f in want[-1:]:
                    open_html(os.path.join(day_dir, f))


def open_html(html):
    print('Отчёт:', html)
    try:
        webbrowser.open('file:///' + html.replace('\\', '/'))
    except Exception:
        pass


if __name__ == '__main__':
    main()