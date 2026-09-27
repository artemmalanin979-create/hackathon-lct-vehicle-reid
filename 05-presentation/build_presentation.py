"""Assemble supplied content in the original organizer template.

Contact data is read in memory and written only to presentation.pptx.
The validation copy contains only the eleven technical slides.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import math
import re

from PIL import Image, ImageChops, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR as C
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.opc.packuri import PackURI
from pptx.util import Inches, Pt

from check_ui_provenance import require_current_capture, ui_source_repo


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TEMPLATE = REPO / '00-task/assets/ЛЦТ2026 Шаблон презентации.pptx'
TEAM = ROOT / 'team-data.md'
TEAM_LOGO = ROOT / 'assets/logo-prosvet.png'
SLIDES = ROOT / 'slides.md'
OUTPUT = ROOT / 'ЛЦТ2026-задача7-ПРОСВЕТ.pptx'
FONTDIR = Path('/usr/share/fonts/julietaula-montserrat-fonts')
CHECKS = ROOT / 'checks'
require_current_capture(ui_source_repo(REPO), ROOT / 'assets/ui-provenance.json')
CHECKS.mkdir(exist_ok=True)
FIT = []
MEDIA = []
CURRENT = 0


def font(size, bold=False):
    return ImageFont.truetype(str(FONTDIR / ('Montserrat-Bold.otf' if bold else 'Montserrat-Regular.otf')), round(size * 8))


def line_count(text, width_pt, size, bold=False):
    f = font(size, bold)
    max_width = width_pt * 8 * .94
    count = 0
    for line in text.split('\n'):
        words = line.split()
        if not words:
            count += 1
            continue
        cur = ''
        for word in words:
            if cur and f.getlength(cur + ' ' + word) > max_width:
                count += 1
                cur = ''
            if f.getlength(word) > max_width:
                # PowerPoint can wrap long single words; count conservatively.
                if cur:
                    count += 1
                count += max(0, math.ceil(f.getlength(word) / max_width) - 1)
                cur = word
            else:
                cur = (cur + ' ' + word).strip()
        count += 1
    return count


def set_text(shape, content, size=16, color=C.ACCENT_5, bold=False,
             min_size=None, after=4, line=1.08, align=PP_ALIGN.LEFT,
             margin=.025, height=None):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    if isinstance(content, str):
        content = content.split('\n')
    specs = [dict(text=x) if isinstance(x, str) else dict(x) for x in content]
    width_pt = shape.width / 12700 - margin * 144
    height_pt = (height if height is not None else shape.height / 12700) - margin * 144
    minimum = size if min_size is None else min_size
    factor = 1.0
    needed = 0
    while True:
        needed = 0
        for sp in specs:
            fs = sp.get('size', size) * factor
            needed += line_count(sp['text'], width_pt, fs, sp.get('bold', bold)) * fs * sp.get('line', line)
            needed += sp.get('after', after)
        needed -= specs[-1].get('after', after) if specs else 0
        if needed <= height_pt or size * factor <= minimum + .01:
            break
        factor = max(minimum / size, factor - .025)
    for n, sp in enumerate(specs):
        p = tf.paragraphs[0] if n == 0 else tf.add_paragraph()
        p.alignment = sp.get('align', align)
        p.line_spacing = sp.get('line', line)
        p.space_before = Pt(0)
        p.space_after = Pt(sp.get('after', after))
        pp = p._p.get_or_add_pPr()
        pp.set('marL', '0')
        pp.set('indent', '0')
        for ch in list(pp):
            if ch.tag.rsplit('}', 1)[-1].startswith('bu'):
                pp.remove(ch)
        pp.append(OxmlElement('a:buNone'))
        # Add explicit soft line breaks while preserving paragraph formatting.
        parts = sp['text'].split('\n')
        for j, part in enumerate(parts):
            if j:
                p.add_line_break()
            r = p.add_run()
            r.text = part
            r.font.name = 'Montserrat'
            r.font.size = Pt(sp.get('size', size) * factor)
            r.font.bold = sp.get('bold', bold)
            r.font.color.theme_color = sp.get('color', color)
            r._r.get_or_add_rPr().set('lang', 'ru-RU')
        p.font.name = 'Montserrat'
        p.font.size = Pt(sp.get('size', size) * factor)
    FIT.append(dict(slide=CURRENT, shape=shape.shape_id if hasattr(shape, 'shape_id') else 'cell',
                    font_pt=round(size * factor, 2), estimated_pt=round(needed, 2),
                    available_pt=round(height_pt, 2), fits=needed <= height_pt + .1))
    return shape


def box(slide, x, y, w, h, text, size=16, **kw):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    return set_text(sh, text, size=size, **kw)


def by_id(slide, ident):
    return next(sh for sh in slide.shapes if sh.shape_id == ident)


def remove(shape):
    shape._element.getparent().remove(shape._element)


def panel(slide, x, y, w, h, text, size=15, fill=C.ACCENT_2, color=C.ACCENT_5):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.theme_color = fill
    sh.line.fill.background()
    sh.adjustments[0] = .10
    set_text(sh, text, size, color, min_size=size, after=0, align=PP_ALIGN.CENTER, margin=.075)
    sh.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def arrow(slide, x1, y1, x2, y2, both=False):
    sh = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.theme_color = C.ACCENT_5
    sh.line.width = Pt(1.25)
    ln = sh._element.spPr.get_or_add_ln()
    tail = OxmlElement('a:tailEnd')
    tail.set('type', 'triangle')
    tail.set('w', 'sm')
    tail.set('len', 'sm')
    ln.append(tail)
    if both:
        head = OxmlElement('a:headEnd')
        head.set('type', 'triangle')
        head.set('w', 'sm')
        head.set('len', 'sm')
        ln.append(head)
    return sh


class CellBox:
    def __init__(self, cell, w, h):
        self.text_frame = cell.text_frame
        self.width, self.height = Inches(w), Inches(h)


def table(slide, x, y, widths, heights, rows, size=14, header=13):
    gf = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(sum(heights)))
    t = gf.table
    t.first_row = False
    t.horz_banding = False
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for i, h in enumerate(heights):
        t.rows[i].height = Inches(h)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.theme_color = C.ACCENT_5 if i == 0 else (C.ACCENT_2 if i % 2 else C.LIGHT_1)
            c.margin_left = c.margin_right = Inches(.08)
            c.margin_top = c.margin_bottom = Inches(.055)
            tcpr = c._tc.get_or_add_tcPr()
            for tag in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
                ln = OxmlElement(tag)
                ln.append(OxmlElement('a:noFill'))
                tcpr.append(ln)
            set_text(CellBox(c, widths[j], heights[i]), value, header if i == 0 else size,
                     color=C.LIGHT_1 if i == 0 else C.ACCENT_5,
                     bold=i == 0, min_size=header if i == 0 else size,
                     after=0, line=1.07, margin=.07)
    return gf


# Логотип команды нарисован тремя цветами; шум экспорта (±2 в плоских зонах)
# раздувает PNG до 869 КиБ. Фиксированная палитра из этих цветов и переходов между
# ними снимает шум, не трогая ни пиксельную сетку, ни исходный файл в репозитории.
TEAM_LOGO_COLORS = ((251, 250, 246), (24, 25, 29), (148, 223, 51))
TEAM_LOGO_STEPS = 16
# Свободная полоса титула: плашка шаблона кончается на 2,16", буквы названия команды
# начинаются на 4,53". Варианты размещения на выбор — левый край, верх, сторона.
# Поля внутри самого логотипа — 8,3 % стороны слева, поэтому координаты сдвинуты:
# по краю и по строке выравнивается сам знак, а не рамка файла.
TEAM_LOGO_PLACEMENTS = {
    'A': (2.68, 2.59, 1.50),   # по центру над названием команды
    'B': (0.636, 2.59, 1.50),  # знаком по левому краю логотипа постановщика
    'C': (0.677, 4.263, 1.0),  # слева от названия, в одну строку с ним
}
TEAM_LOGO_PLACEMENT = 'A'


def flat_png(path, colors, steps):
    """Пережать плоскую графику фиксированной палитрой, сохранив размер в пикселях."""
    ramp = []
    for i, start in enumerate(colors):
        for end in colors[i + 1:]:
            for step in range(steps + 1):
                shade = tuple(round(a + (b - a) * step / steps) for a, b in zip(start, end))
                if shade not in ramp:
                    ramp.append(shade)
    palette = Image.new('P', (1, 1))
    palette.putpalette([v for shade in ramp for v in shade] + [0] * (768 - 3 * len(ramp)))
    with Image.open(path) as source:
        source = source.convert('RGB')
        packed = source.quantize(palette=palette, dither=Image.Dither.NONE)
        deviation = max(high for _, high in ImageChops.difference(packed.convert('RGB'), source).getextrema())
        size = packed.size
    stream = BytesIO()
    packed.save(stream, format='PNG', optimize=True)
    stream.seek(0)
    return stream, size, len(ramp), deviation


def picture(slide, path, crop, x, y, w, h, label):
    path = Path(path)
    with Image.open(path) as source:
        im = source.crop(crop) if crop else source.copy()
        im = im.convert('RGB')
        stream = BytesIO()
        im.save(stream, format='PNG')
        iw, ih = im.size
    stream.seek(0)
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    sh = slide.shapes.add_picture(stream, Inches(x + (w - dw) / 2), Inches(y + (h - dh) / 2), Inches(dw), Inches(dh))
    sh.name = label
    sh._element.nvPicPr.cNvPr.set('descr', label)
    MEDIA.append(dict(slide=CURRENT, source=str(path.relative_to(REPO)), crop=crop, label=label, embedded=True))
    return sh


def foot(slide, text):
    return box(slide, .66, 7.01, 11.64, .35, text, 10.2, color=C.LIGHT_1, min_size=9.5, after=0, margin=0)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


prs = Presentation(TEMPLATE)
source_content = prs.slides[12]
logo = by_id(prs.slides[5], 2172).image.blob
kept = list(prs.slides)[6:11]
for idx in reversed(range(len(prs.slides))):
    if idx not in range(6, 11):
        entry = prs.slides._sldIdLst[idx]
        prs.part.drop_rel(entry.rId)
        prs.slides._sldIdLst.remove(entry)
for i, s in enumerate(kept, 1):
    s.part._partname = PackURI(f'/ppt/slides/slide{i}.xml')

raw_team = TEAM.read_text()
members = []
for raw in raw_team.splitlines():
    if raw.startswith('|'):
        cells = [c.strip() for c in raw.strip('|').split('|')]
        if len(cells) == 6 and cells[0].isdigit():
            members.append(cells)
assert len(members) == 5
team_name = re.search(r'Команда \*\*(.+?)\*\*', raw_team).group(1)
captain = re.search(r'Капитан — ([^.]+)\.', raw_team).group(1)
speciality = members[0][2].split('Специальность — ', 1)[1]
def quote_block(start, end=None):
    part = raw_team.split(start, 1)[1]
    if end:
        part = part.split(end, 1)[0]
    lines = [line[1:].strip() for line in part.splitlines() if line.startswith('>')]
    return ' '.join(lines).strip()
short_history = quote_block('**Вариант покороче, если на слайде мало места:**', '### Почему')
why = quote_block('### Почему выбрали эту задачу', '### Что было сложного')
difficulty = quote_block('### Что было сложного', '## Чего ещё не хватает')

# 1: mandatory cover, original slide 7.
CURRENT = 1
s = kept[0]
set_text(by_id(s, 3), team_name, 46, C.LIGHT_1, True, min_size=42, align=PP_ALIGN.CENTER, after=0)
set_text(by_id(s, 5), 'Задача №7. Сервис формирования цифрового признака транспортного средства для сопоставления снимков одного автомобиля из разных локаций без использования государственного номера', 18.2, C.LIGHT_1, min_size=16, align=PP_ALIGN.CENTER, after=0)
ph = by_id(s, 4)
with Image.open(BytesIO(logo)) as im:
    iw, ih = im.size
width = 4.82
height = width * ih / iw
pic = s.shapes.add_picture(BytesIO(logo), Inches(.76), Inches(.68), Inches(width), Inches(height))
pic.name = 'Фалькон Тех — официальный логотип из слайда 6 шаблона'
remove(ph)
MEDIA.append(dict(slide=1, source='00-task/assets/ЛЦТ2026 Шаблон презентации.pptx', shape=2172, label=pic.name, embedded=True))
left, top, side = TEAM_LOGO_PLACEMENTS[TEAM_LOGO_PLACEMENT]
stream, (logo_w, logo_h), shades, deviation = flat_png(TEAM_LOGO, TEAM_LOGO_COLORS, TEAM_LOGO_STEPS)
packed_bytes = stream.getbuffer().nbytes
team_pic = s.shapes.add_picture(stream, Inches(left), Inches(top), Inches(side), Inches(side * logo_h / logo_w))
team_pic.name = 'ПРОСВЕТ — логотип команды'
team_pic._element.nvPicPr.cNvPr.set('descr', 'Логотип команды ПРОСВЕТ')
MEDIA.append(dict(slide=1, source='05-presentation/assets/logo-prosvet.png', crop=None, label=team_pic.name,
                  embedded=True, placement=TEAM_LOGO_PLACEMENT, inches=round(side, 3), palette=shades,
                  max_deviation=deviation, bytes=packed_bytes, source_bytes=TEAM_LOGO.stat().st_size))
notes(s, 'Титульный слайд на основе страницы 7 оригинального шаблона. Официальный логотип постановщика взят непосредственно из страницы 6 этого же шаблона. Логотип команды пережат фиксированной палитрой без изменения размера в пикселях; исходный файл в репозитории не меняется.')

# 2: mandatory team and solution, original slide 8.
CURRENT = 2
s = kept[1]
remove(by_id(s, 2))  # No team photograph was supplied; do not invent one.
set_text(by_id(s, 18), f'КОМАНДА\n«{team_name}»', 23, bold=True, min_size=21, after=0)
set_text(by_id(s, 14), [
    {'text': f'Капитан: {captain}', 'bold': True},
    {'text': f'Специальность: {speciality}'},
    {'text': 'Кол-во участников: 5 человек'},
    {'text': short_history},
    {'text': 'Город и регион: Чебоксары, Чувашская Республика'},
], 12.4, min_size=11.5, after=5, line=1.08)
set_text(by_id(s, 5), 'Сервис строит компактный цифровой признак автомобиля по внешнему виду и находит те же машины на снимках с других камер. Распознавание номера в вычислительный путь не входит.', 14.3, min_size=12.8, after=0)
set_text(by_id(s, 8), 'Для d1_j48 проверены 6 контрастов маски номера и контрольной области: 3 маски × 2 режима. Значимой разницы не обнаружено. Наша валидация.', 14, min_size=12.2, after=0)
for ident in (6, 15, 4):
    sh = by_id(s, ident)
    set_text(sh, sh.text, 16, bold=True, after=0)
notes(s, 'История команды — локальный team-data.md. Проверка зоны номера: 04-solution/plate-ablation/d1_j48/plate_ablation_d1_j48.json, 3 маски × 2 режима, интервалы включают ноль; полной независимости это не доказывает. OCR отсутствует в сервисном пути: 04-solution/service/app/core/model.py. Фотография команды не предоставлена, необязательный фотоплейсхолдер удалён.')

# 3: mandatory participant matrix in the five original cards, original slide 9.
CURRENT = 3
s = kept[2]
set_text(by_id(s, 7), f'КОМАНДА «{team_name}»', 20, color=C.LIGHT_1, bold=True, after=0)
name_ids = (15, 58, 61, 64, 67)
body_ids = (9, 57, 60, 63, 66)
for pid in (2, 3, 4, 5, 6):
    remove(by_id(s, pid))
for member, nid, bid in zip(members, name_ids, body_ids):
    name, role, handle, phone, university = member[1:]
    # Specialty is already shown in the mandatory team overview; preserve full role here.
    role = role.split(' Специальность — ', 1)[0]
    set_text(by_id(s, nid), name.replace(' ', '\n'), 13.2, bold=True, min_size=12.5, after=0, line=1.02, margin=.015)
    set_text(by_id(s, bid), [role, handle, phone, university], 11.2, min_size=10.1, after=5, line=1.05, margin=.02)
notes(s, 'Страница 9 шаблона: сохранены пять колонок участников и их поля. Фотографии не предоставлены, необязательные фотоплейсхолдеры удалены. Специальность капитана приведена в обзоре команды. Контактные данные находятся только на этом слайде.')

# 4: mandatory team story, original slide 10.
CURRENT = 4
s = kept[3]
set_text(by_id(s, 7), 'ИСТОРИЯ КОМАНДЫ', 20, C.LIGHT_1, True, after=0)
STORY = 12.6  # общий кегль трёх блоков страницы 10 шаблона; сжатие запрещено
set_text(by_id(s, 37), short_history, STORY, min_size=STORY, after=0)
# Поле блока 02 кончается на 4,929", а разделительная линия шаблона лежит на 4,82":
# размещение считаем по расстоянию до линии с зазором 0,06". Оценка в set_text —
# кегль x интервал, реальная строка выше примерно в 1,2 раза, отсюда поправка.
to_line = 4.82 - .06 - by_id(s, 43).top / 914400  # дюймы от верха поля до линии
set_text(by_id(s, 43), why, STORY, min_size=STORY, after=0, height=(to_line / 1.2 + .05) * 72)
set_text(by_id(s, 40), difficulty, STORY, min_size=STORY, after=0)
for ident in (38, 41, 44, 9, 11, 12):
    sh = by_id(s, ident)
    set_text(sh, sh.text.replace('\v', '\n'), 14 if ident in (38, 41, 44) else 17, bold=True, min_size=13 if ident in (38, 41, 44) else 17, after=0, align=PP_ALIGN.RIGHT if ident in (9, 11, 12) else PP_ALIGN.LEFT)
notes(s, 'Использован предоставленный короткий вариант истории команды. Тексты мотивации и сложностей перенесены без содержательного сокращения. Текст о сложностях в источнике помечен как черновик: команде следует подтвердить его перед выступлением.')

# 5: mandatory overview, original slide 11, with the three requested content fields.
CURRENT = 5
s = kept[4]
set_text(by_id(s, 14), 'КОРОТКО О РЕШЕНИИ', 20, C.LIGHT_1, True, after=0)
for ident in (15, 17):
    sh = by_id(s, ident)
    set_text(sh, sh.text.strip(), 20, bold=True, min_size=18.5, after=0)
set_text(by_id(s, 3), [
    'Две модели + whitening → вектор из 512 чисел → кандидаты или отказ.',
    'd1_j48: mAP 0,7741, Rank-1 0,7308 с переранжированием; 832 запроса с парой.',
], 18, min_size=17, after=15, margin=.055)
set_text(by_id(s, 7), [
    {'text': 'Помогает оператору сопоставить автомобиль между камерами по внешности.', 'after': 10},
    {'text': 'Порог делает цену отказа явной: принимаем 606 из 832 запросов с парой и 60 из 278 без пары.', 'after': 18},
    {'text': 'Идеи по дальнейшему развитию', 'bold': True, 'size': 18, 'after': 10},
    {'text': 'Проверить локальные приметы двойников и расширение данных заказчика.', 'after': 10},
    {'text': 'Перенести измеренный ANN-поиск на d1_j48 и перекалибровать отказ.'},
], 15.8, min_size=15, after=8, margin=.05)
foot(s, 'd1_j48; наша валидация, market / presence')
notes(s, 'Источники чисел: 04-solution/training/combined/s02_metrics.json → d1_j48.kr; 04-solution/service/calib-d1_j48/summary.json → rerank_market_presence.selected.robust_balanced (tp=606, fp=60, fn=226, tn_unknown=218). В исходном шаблоне две белые панели; третье смысловое поле «Идеи по дальнейшему развитию» расположено внутри правой панели с отдельным заголовком, без изменения фона, панелей и двухколоночной структуры.')

# Technical content is authored as a separate visual story. Mandatory slides above
# retain the exact organizer fields, theme, geometry and participants contract.
text_source = SLIDES.read_text()
sections = re.split(r'\n## (?=\d+\.)', text_source)[1:]
assert len(sections) == 11

INK, PAPER, PINK, MUTED = C.DARK_2, C.LIGHT_1, C.ACCENT_1, C.ACCENT_5
DARK_SLIDES = {6, 16}
CASE_ROOT = REPO / '04-solution/error-analysis/d1_j48'
CASES = {r['case']: r for r in json.loads((CASE_ROOT / 'examples/manifest.json').read_text())}
IMAGE_SOURCES = json.loads((CASE_ROOT / 'out/image_sources.json').read_text())
FINAL_QUERIES = json.loads((CASE_ROOT / 'out/per_query_d1_j48.json').read_text())


def txt(s, x, y, w, h, text, size=18, **kwargs):
    kwargs.setdefault('color', PAPER if CURRENT in DARK_SLIDES else INK)
    return box(s, x, y, w, h, text, size, after=0, margin=0, **kwargs)


def rule(s, x1, y, x2, color=C.LIGHT_2):
    sh = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y))
    sh.line.color.theme_color = color
    sh.line.width = Pt(.6)
    return sh


def frame(s, sh, color=PINK):
    border = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, sh.left, sh.top, sh.width, sh.height)
    border.fill.background()
    border.line.color.theme_color = color
    border.line.width = Pt(1.1)


def technical(title, source_note):
    s = prs.slides.add_slide(source_content.slide_layout)
    for sh in list(s.shapes):
        remove(sh)
    s._element.set('showMasterSp', '0')
    s.background.fill.solid()
    s.background.fill.fore_color.theme_color = INK if CURRENT in DARK_SLIDES else PAPER
    txt(s, .65, .35, 8, .25, 'ПРОСВЕТ   /   ЛЦТ 2026   /   ЗАДАЧА 7', 10.5,
        color=C.ACCENT_2 if CURRENT in DARK_SLIDES else MUTED, bold=True)
    txt(s, .65, .83, 12, 1.07, title, 31, bold=True, line=1.02)
    txt(s, .65, 7.00, 11.3, .28, source_note, 10.3,
        color=C.ACCENT_2 if CURRENT in DARK_SLIDES else MUTED)
    txt(s, 12.13, 6.96, .55, .32, str(CURRENT), 11,
        align=PP_ALIGN.RIGHT, color=C.ACCENT_2 if CURRENT in DARK_SLIDES else MUTED)
    notes(s, sections[CURRENT - 6].strip())
    return s


def case_picture(s, case, column, x, y, w, h, label):
    """Reuse only the exact photograph rectangle from a verified case sheet.

    Caption/metrics remain editable. Geometry mirrors the documented ImageOps
    contain operation, without cropping the actual car or manufacturing pixels.
    """
    spec = CASES[case]
    double = spec['mode'] in ('false_accept', 'false_reject')
    ids = [spec['query_image_id'], spec['top_gallery_image_id'], spec['best_positive_image_id']]
    if spec['mode'] == 'fixed' and column == 2:
        ident = spec['top_gallery_image_id']
    else:
        ident = ids[column]
    _, _, bw, bh = IMAGE_SOURCES[ident]['bbox']
    cw, step = (872, 920) if double else (568, 608)
    if bw / bh > cw / 532:
        iw, ih = cw, round(bh * cw / bw)
    else:
        iw, ih = round(bw * 532 / bh), 532
    left = 64 + column * step + (cw - iw) // 2
    top = 245 + (532 - ih) // 2
    return picture(s, CASE_ROOT / spec['path'], (left, top, left + iw, top + ih),
                   x, y, w, h, f'{case}: {label}')


def diagram_node(s, x, y, w, h, heading, sub='', emphasis=False):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.theme_color = C.ACCENT_6 if emphasis else PAPER
    sh.line.color.theme_color = C.ACCENT_6 if emphasis else C.LIGHT_2
    sh.line.width = Pt(.8)
    txt(s, x + .17, y + .17, w - .34, .38, heading, 15.5, bold=True,
        color=PAPER if emphasis else INK)
    if sub:
        txt(s, x + .17, y + .66, w - .34, h - .76, sub, 11.7,
            color=PAPER if emphasis else MUTED)
    return sh


# 6. An actual successful pair of the submitted model sets the visual premise.
CURRENT = 6
s = technical('Один автомобиль. Две камеры',
              'Валидация, случай F03. d1_j48 + KR, полный сплит 1110 × 750. Кадры и bbox без ретуши.')
txt(s, .65, 1.81, 11.8, .64, 'Ракурс меняется. Задача остаётся: найти ту же машину по внешности.', 20)
txt(s, .65, 2.65, 5.8, .37, 'ЗАПРОС   /   КАМЕРА 39', 12, bold=True, color=C.ACCENT_2)
txt(s, 6.97, 2.65, 5.7, .37, 'ВЕРНАЯ ПАРА   /   КАМЕРА 33', 12, bold=True, color=C.ACCENT_2)
for col, x in [(0, .65), (2, 6.97)]:
    sh = case_picture(s, 'F03', col, x, 3.08, 5.71, 2.96, 'запрос' if col == 0 else 'верный top-1')
    frame(s, sh, C.ACCENT_2)
txt(s, .65, 6.32, 11.6, .44, 'Исходная OSNet: верная пара на 2-м месте. d1_j48: на 1-м.', 19, bold=True)

# 7. The actual UI screenshot is supplied after the independent UI workstream.
CURRENT = 7
s = technical('Поиск начинается с кадра и рамки',
              'HTTP API: точный cosine. Score — близость, не вероятность. Ошибка API отдельно от отказа модели.')
UI = ROOT / 'assets/ui-search.png'
# A missing real screenshot is a build error, never a presentational placeholder.
ui_provenance = json.loads((ROOT / 'assets/ui-provenance.json').read_text())
import hashlib
assert hashlib.sha256(UI.read_bytes()).hexdigest() == ui_provenance['source_sha256']
picture(s, UI, None, .65, 1.92, 8.05, 4.85, 'Работающий сервис: актуальный поиск')
for n, y, heading, body in [
    ('01', 2.0, 'Выделить автомобиль', 'Демокадр или JPEG/PNG.\nРамка мышью, касанием\nили точными числами.'),
    ('02', 3.55, 'Сравнить кандидатов', 'Крупная пара кадров.\nЯвный порог и оценки\nниже него.'),
    ('03', 5.13, 'Сохранить результат', 'CSV / JSON текущего поиска.\nОтказ тоже сохраняется.'),
]:
    txt(s, 9.15, y, .52, .33, n, 13, bold=True, color=PINK)
    txt(s, 9.67, y-.025, 3.03, .68, heading, 16, bold=True)
    txt(s, 9.67, y+.69, 3.03, .78, body, 12.8)

# 8. Every box and connector is an editable presentation object.
CURRENT = 8
s = technical('Две ветви. Один признак из 512 чисел',
              'd1_j48 @208. Нормировка каждого вектора, среднее, L2, whitening rho=0,5, L2. Обучение whitening: train_fit.')
diagram_node(s, .65, 2.18, 1.75, 1.38, 'Кроп', 'RGB\n208 × 208')
diagram_node(s, 2.90, 1.96, 2.2, .78, 'OSNet-AIN')
diagram_node(s, 2.90, 3.04, 2.2, 1.20, 'combined_v1', 'Наше LP-FT\nдообучение', emphasis=True)
diagram_node(s, 5.62, 2.18, 2.13, 1.38, 'Объединение', 'L2 ветвей\nСреднее → L2')
diagram_node(s, 8.26, 2.18, 2.18, 1.38, 'Whitening', 'Матрица P\n→ L2')
diagram_node(s, 10.95, 2.18, 1.75, 1.38, '512', 'float32', emphasis=True)
for args in [(2.42,2.72,2.83,2.36),(2.42,3.10,2.83,3.43),(5.13,2.36,5.55,2.72),(5.13,3.43,5.55,3.10),(7.80,2.87,8.18,2.87),(10.50,2.87,10.88,2.87)]:
    arrow(s,*args)
rule(s,.65,4.63,12.70)
for x,w,heading,body in [
    (.65,3.74,'Почему OSNet','Перенос на наши данные:\nmAP 0,657 против 0,238\nу R50-IBN, cosine.'),
    (4.95,3.4,'Смена ракурса','Межкамерный MCNL,\nбаланс камер, MixStyle\nи аугментации при обучении.'),
    (9.0,3.7,'Два режима поиска','API: точный cosine.\nBatch: KR(6, 3, 0,3),\nзависит от состава batch.'),
]:
    txt(s,x,4.96,w,.45,heading,19,bold=True)
    txt(s,x,5.58,w,1.13,body,15.7)

# 9. Native PowerPoint chart with a common zero baseline and exact values in notes.
CURRENT = 9
s = technical('Качество поиска зависит от режима',
              'Валидация: 1110 query × 750 gallery; 832 query с парой. market, same-camera исключены. Full-ranking mAP.')
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_TICK_MARK
metric_data = json.loads((REPO / '04-solution/training/combined/s02_metrics.json').read_text())
release = metric_data['d1_j48']
cd = CategoryChartData()
cd.categories = ['mAP', 'Rank-1']
cd.add_series('Cosine / API', (release['cos']['mAP'], release['cos']['Rank-1']))
cd.add_series('KR / batch', (release['kr']['mAP'], release['kr']['Rank-1']))
chart = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(.52), Inches(2.00), Inches(8.07), Inches(3.37), cd).chart
chart.has_legend=True
chart.legend.position=XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout=False
chart.legend.font.name='Montserrat'; chart.legend.font.size=Pt(12)
chart.category_axis.tick_labels.font.name='Montserrat'; chart.category_axis.tick_labels.font.size=Pt(16)
chart.value_axis.minimum_scale=0; chart.value_axis.maximum_scale=1; chart.value_axis.major_unit=.25
chart.value_axis.tick_labels.font.name='Montserrat'; chart.value_axis.tick_labels.font.size=Pt(11)
chart.value_axis.major_tick_mark=XL_TICK_MARK.NONE
chart.value_axis.has_major_gridlines=True
chart.value_axis.major_gridlines.format.line.color.theme_color=C.LIGHT_2
chart.plots[0].has_data_labels=True
chart.plots[0].data_labels.position=XL_LABEL_POSITION.OUTSIDE_END
chart.plots[0].data_labels.number_format='0.0000'
chart.plots[0].data_labels.font.name='Montserrat'; chart.plots[0].data_labels.font.size=Pt(14)
for series,color in zip(chart.series,[C.ACCENT_4,PINK]):
    series.format.fill.solid();series.format.fill.fore_color.theme_color=color
    series.format.line.fill.background()
chart.plots[0].gap_width=65
chart.plots[0].overlap=-15
txt(s,9.12,2.06,3.5,.95,'608 / 832',36,bold=True,color=MUTED)
txt(s,9.12,3.06,3.5,.81,'верных первых ответов\nу d1_j48 + KR',17)
txt(s,9.12,4.21,3.5,1.18,'78 ошибок исправлены,\n22 добавлены\nпротив OSNet + KR.',17)
rule(s,.65,5.76,12.7)
txt(s,.65,6.02,11.9,.72,'Одна многократно использованная валидация. Закрытый test без меток.\nЭти числа не обещают такое же качество на новых камерах.',16.5)

# 10. Real errors on both sides of the operating threshold, not a fabricated demo.
CURRENT = 10
s = technical('Порог имеет цену в обе стороны',
              'd1_j48 + KR, t=0,5282812306342437. В API другой порог: 0,5141976914190476. Оценки не вероятности.')
for x,case,heading,sub in [
    (.65,'P07','Чужая машина проходит порог','Пары нет по разметке. Score 0,839679.'),
    (6.95,'N01','Верная машина получает отказ','Правильный top-1. Score 0,520430.'),
]:
    txt(s,x,1.96,5.72,.65,heading,20,bold=True)
    for col,dx in [(0,0),(1,2.96)]:
        case_picture(s,case,col,x+dx,2.82,2.72,1.87,'запрос' if col==0 else 'top-1')
        txt(s,x+dx,4.81,2.72,.3,'Запрос' if col==0 else 'Первый кандидат',12,color=MUTED)
    txt(s,x,5.32,5.7,.72,sub,17)
txt(s,.65,6.25,5.7,.48,'60 из 278 без пары приняты',20,bold=True,color=MUTED)
txt(s,6.95,6.25,5.7,.48,'226 из 832 с парой отклонены',20,bold=True,color=MUTED)

# 11. A residual failure of the released model at a readable image size.
CURRENT = 11
s = technical('Одинаковая ливрея скрывает различия',
              'E17, query 516. d1_j48 + KR на валидации. Верная пара осталась третьей; это ошибка финальной сборки.')
for col,x,heading,caption in [
    (0,.65,'Запрос','v451 · камера 19'),
    (1,4.94,'Ошибочный top-1','v1158 · score 0,929677'),
    (2,9.23,'Верная пара','v451 · score 0,278172'),
]:
    txt(s,x,2.08,3.45,.42,heading,18.5,bold=True)
    sh=case_picture(s,'E17',col,x,2.76,3.45,2.70,heading)
    if col==1:frame(s,sh,PINK)
    txt(s,x,5.65,3.45,.38,caption,13.7,color=MUTED)
txt(s,.65,6.28,11.9,.5,'Похожий ракурс чужой машины оказался сильнее различий конкретного автомобиля.',17)

# 12. Two further mechanisms, each with the complete query / wrong / positive triplet.
CURRENT = 12
s = technical('Свет и перекрытие остаются трудными',
              'E08/E44, d1_j48 + KR, val. Причины по одному наблюдателю — гипотезы.')
for case,y,heading,detail in [
    ('E08',2.05,'Свет фар','Верная пара: ранг 7'),
    ('E44',4.48,'Перекрытие кузова','Верная пара ниже top-1'),
]:
    txt(s,.65,y,2.15,.64,heading,19,bold=True)
    txt(s,.65,y+.73,2.1,.95,detail,14.4,color=MUTED)
    for col,x in [(0,3.17),(1,6.43),(2,9.69)]:
        txt(s,x,y-.03,2.95,.33,['Запрос','Ошибочный top-1','Верная пара'][col],13.2,bold=True)
        case_picture(s,case,col,x,y+.42,2.95,1.65,['запрос','top-1','верная пара'][col])
rule(s,.65,4.27,12.7)

# 13. Separate measured descriptor cost from a research ANN index experiment.
CURRENT = 13
s = technical('Цена дескриптора и масштаб поиска',
              'CPU-замер branches3: batch=1, ORT=2 потока, 50 кадров. ANN — отдельный опыт на синтетических векторах OSNet.')
txt(s,.65,2.04,5.4,.4,'d1_j48 / ИЗВЛЕЧЕНИЕ ПРИЗНАКА',12,bold=True,color=MUTED)
txt(s,.65,2.66,5.55,1.0,'372,5 мс',42,bold=True)
txt(s,.65,3.77,5.55,.9,'Среднее на кадр\nМедиана 392,7 мс; p90 421,7 мс',17)
txt(s,7.17,2.04,5.5,.4,'ИССЛЕДОВАТЕЛЬСКИЙ ANN / ТОЛЬКО ИНДЕКС',11.5,bold=True,color=MUTED)
txt(s,7.17,2.66,5.5,1.0,'2,98 мс',42,bold=True)
txt(s,7.17,3.77,5.5,1.08,'Медиана, p95 5,14 мс\n99,96 % полноты top-10\nПик RSS 2908 МиБ; 1 CPU-поток',16.3)
rule(s,.65,5.14,12.7)
for x,n,heading in [(.65,'1','d1_j48 на реальных данных'),(4.92,'2','ANN top-K + локальный KR'),(9.18,'3','Новая калибровка отказа')]:
    txt(s,x,5.49,.36,.44,n,20,bold=True,color=PINK)
    txt(s,x+.53,5.49,3.05,.94,heading,16.5,bold=True)
txt(s,.65,6.56,11.9,.3,'Миллион — синтетический индекс. Сквозной путь не измерен. Локальный API: p95 1,81 с, 20 запросов.',11.2,color=MUTED)

# 14. The trained prototype is judged against the predeclared no-loss gate.
CURRENT = 14
core = json.loads((ROOT / 'assets/owncore-distill-summary.json').read_text())
assert core['status'] == 'research_only; release d1_j48'
assert (core['query_count'], core['gallery_count'], core['known_queries']) == (1110, 750, 832)
def metric(value, digits):
    return f'{value:.{digits}f}'.replace('-', '−').replace('.', ',')

s = technical('Собственное ядро быстрее, но ищет хуже',
              'Повторно использованная val: 1110 × 750, 832 запроса с парой; это не новый holdout. KR full-ranking mAP; парный bootstrap по vehicle_id.')
txt(s,.65,2.08,5.45,.36,'КАЧЕСТВО · KR / BATCH',12,bold=True,color=MUTED)
txt(s,.65,2.62,5.55,.60,f"{metric(core['baseline_kr_map'], 4)} → {metric(core['core_kr_map'], 4)}",30,bold=True)
txt(s,.65,3.46,5.57,.76,f"d1_j48 → CrossViewCore\nΔmAP {metric(core['core_kr_delta'], 4)}",18.2)
txt(s,.65,4.51,5.62,.75,
    f"95 % ДИ [{metric(core['core_kr_delta_ci95'][0], 4)}; {metric(core['core_kr_delta_ci95'][1], 4)}]"
    f"\np_Holm(8) = {metric(core['core_kr_p_holm8'], 3)}",17.3,color=MUTED)
txt(s,7.13,2.08,5.5,.36,'СКОРОСТЬ · CPU, BATCH-1',12,bold=True,color=MUTED)
txt(s,7.13,2.62,5.6,.60,
    f"{metric(core['baseline_cpu_p95_ms'], 2)} → {metric(core['core_cpu_p95_ms'], 2)} мс",30,bold=True)
txt(s,7.13,3.46,5.57,.76,'p95, готовый тензор → признак\nОдин CPU, 2 потока',18.2)
txt(s,7.13,4.51,5.6,.75,
    f"Веса {metric(core['baseline_weights_bytes']/1e6, 2)} → {metric(core['core_weights_bytes']/1e6, 2)} МБ"
    f"\n{core['benchmark_samples']} парных кадров, {core['benchmark_warmup']} прогревов",17.3,color=MUTED)
rule(s,.65,5.50,12.7)
txt(s,.65,5.82,7.2,.78,'Критерий качества и отказа не выполнен.\nВ релизе остаётся d1_j48.',19.5,bold=True)
txt(s,8.08,5.82,4.55,.78,
    f"Fusion: KR {metric(core['fusion_kr_map'], 4)}; ДИ Δ включает 0."
    f"\nTNR ядра = {metric(core['core_kr_tnr'], 1)}.",15.5)

# 15. Table wording is an acceptance contract, retained literally and still editable.
CURRENT = 15
s = technical('Один проверяемый комплект сдачи',
              'Readme.md и SOLUTION.md описывают фактическую сборку. Изображения организатора передаются отдельно от Git.')
table(s,.65,1.99,[3.12,8.91],[.54,.91,1.04,1.20],[
    ['Артефакт','Контракт результата одного прогона'],
    ['submission.csv','Первые 10 кандидатов при галерее ≥ 10, независимо от отказа'],
    ['embeddings.npy','По 512 чисел: сначала запросы, затем галерея, в порядке входных CSV'],
    ['candidates.csv','Все пары с исходным score ≥ порога, без лимита 10. Нет строк — отказ при успешном batch и уникальных query ID'],
],size=16.0,header=14.5)
txt(s,.65,6.05,5.73,.62,'manifest.json: SHA входов и выходов.\nrun_info.json: режим и параметры.',16.0,bold=True)
txt(s,7.17,6.05,5.49,.62,'Обе ONNX, whitening, wheels\nи базовые офлайн-образы в комплекте.',16.0)

# 16. The readiness claim remains bounded to the verified local candidate.
CURRENT = 16
s = technical('Демонстрация готовится к внешнему доступу',
              'Локальная сборка не означает публикацию. Финальные ссылки проверяются из внешней сети после развёртывания.')
txt(s,.65,2.07,6.7,1.62,'Кадр\nРамка\nСравнение',34,bold=True,line=1.14)
txt(s,.65,4.6,5.9,1.21,'CrossViewCore обучен и проверен.\nПо критерию качества в релизе\nостаётся d1_j48.',18)
txt(s,8.03,2.10,4.63,.42,'27–28 СЕНТЯБРЯ',13,bold=True,color=C.ACCENT_2)
txt(s,8.03,2.78,4.63,1.18,'Развёртывание VPS\nи проверка доступа жюри',24,bold=True)
txt(s,8.03,4.40,4.63,.91,'Сейчас\nDEPLOY PENDING',21,color=C.ACCENT_2,bold=True)
txt(s,8.03,5.70,4.63,.98,'Репетиция демонстрации\nи резерв без сети',18)

# Normalize visible page numbers, keeping each original number field's position.
for n, s in enumerate(prs.slides, 1):
    CURRENT = n
    for sh in s.shapes:
        if sh.is_placeholder and str(sh.placeholder_format.type).startswith('SLIDE_NUMBER'):
            set_text(sh, str(n), 10.0, C.LIGHT_1 if n in (3, 5, 6, 16) else C.ACCENT_5,
                     after=0, align=PP_ALIGN.CENTER, margin=0)
        elif sh.has_text_frame and 'Номер слайда' in sh.name:
            set_text(sh, str(n), 10.0, C.LIGHT_1 if n in (3, 5, 6, 16) else C.ACCENT_5,
                     after=0, align=PP_ALIGN.CENTER, margin=0)

prs.core_properties.title = f'{team_name} — цифровой признак транспортного средства'
prs.core_properties.subject = 'ЛЦТ 2026, задача №7'
prs.core_properties.author = 'Команда'
prs.core_properties.last_modified_by = 'Команда'
prs.core_properties.comments = ''
prs.save(OUTPUT)

# Check readback and sensitive values without disclosing them in logs or reports.
readback = Presentation(OUTPUT)
assert len(readback.slides) == 16
for member in members:
    for secret in member[3:5]:
        hits = [n for n, sl in enumerate(readback.slides, 1) if any(secret in sh.text for sh in sl.shapes if sh.has_text_frame)]
        assert hits == [3], 'A contact field is absent or outside the participant slide'

# Only public technical content is allowed in the temporary render copy.
for idx in reversed(range(5)):
    entry = readback.slides._sldIdLst[idx]
    readback.part.drop_rel(entry.rId)
    readback.slides._sldIdLst.remove(entry)
readback.core_properties.title = 'Проверка технических слайдов'
readback.save(CHECKS / 'content_preview.pptx')

(CHECKS / 'fit_check.json').write_text(json.dumps(FIT, ensure_ascii=False, indent=2))
(CHECKS / 'illustrations.json').write_text(json.dumps(MEDIA, ensure_ascii=False, indent=2))
overflows = [x for x in FIT if not x['fits']]
print(json.dumps({'slides': 16, 'embedded_illustrations': len(MEDIA), 'contact_check': 'passed',
                  'text_boxes_checked': len(FIT), 'fit_warnings': overflows}, ensure_ascii=False, indent=2))
