"""Assemble supplied content in the original organizer template.

Contact data is read in memory and written only to presentation.pptx.
The validation copy contains only the ten technical slides.
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TEMPLATE = REPO / '00-task/assets/ЛЦТ2026 Шаблон презентации.pptx'
TEAM = ROOT / 'team-data.md'
TEAM_LOGO = ROOT / 'assets/logo-prosvet.png'
SLIDES = ROOT / 'slides.md'
OUTPUT = ROOT / 'ЛЦТ2026-задача7-ПРОСВЕТ.pptx'
FONTDIR = Path('/usr/share/fonts/julietaula-montserrat-fonts')
CHECKS = ROOT / 'checks'
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
set_text(by_id(s, 5), 'Сервис строит компактный цифровой признак автомобиля по внешнему виду и находит те же машины на снимках с других камер — без использования государственного номера.', 14.3, min_size=12.8, after=0)
set_text(by_id(s, 8), 'Заметного вклада зоны номера не обнаружили. Это проверка исходных весов и косинусного поиска. Для переранжирования и дообученной модели измерение нужно повторить. Наша валидация.', 14, min_size=12.2, after=0)
for ident in (6, 15, 4):
    sh = by_id(s, ident)
    set_text(sh, sh.text, 16, bold=True, after=0)
notes(s, 'Использован готовый короткий вариант истории команды. Формулировка о зоне номера соответствует ограниченному выводу в slides.md, раздел 5; утверждение о доказанной полной независимости не используется. Фотография команды не предоставлена, необязательный фотоплейсхолдер удалён.')

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
    'Кадр и рамка автомобиля → вектор из 512 чисел → поиск кандидатов или отказ.',
    'Пакетное переранжирование: mAP 0,694 на 832 запросах с парой в нашей валидации.',
], 18, min_size=17, after=15, margin=.055)
set_text(by_id(s, 7), [
    {'text': 'Помогает оператору сопоставить автомобиль между камерами по внешности.', 'after': 10},
    {'text': 'Порог делает цену отказа явной: принимаем 603 из 832 запросов с парой и 62 из 278 без пары.', 'after': 18},
    {'text': 'Идеи по дальнейшему развитию', 'bold': True, 'size': 18, 'after': 10},
    {'text': 'Дообучить на смене ракурса и освещения — ≈58 % ошибок базовой модели.', 'after': 10},
    {'text': 'Проверить локальные приметы двойников и поиск в галерее порядка 10⁶ объектов.'},
], 15.8, min_size=15, after=8, margin=.05)
foot(s, 'Наша валидация, market / presence')
notes(s, 'Источник: slides.md, дополнение об обязательном слайде 11. В исходном шаблоне две белые панели; третье смысловое поле «Идеи по дальнейшему развитию» расположено внутри правой панели с отдельным заголовком, без изменения фона, панелей и двухколоночной структуры.')

text_source = SLIDES.read_text()
sections = re.split(r'\n## (?=\d+\.)', text_source)[1:]
titles = [re.search(r'### Заголовок\s+\*\*(.*?)\*\*', sec, re.S).group(1) for sec in sections[:10]]


def clone_content():
    s = prs.slides.add_slide(source_content.slide_layout)
    for sh in list(s.shapes):
        remove(sh)
    rid_map = {}
    for rel in source_content.part.rels.values():
        if rel.reltype.endswith('/slideLayout') or rel.reltype.endswith('/notesSlide'):
            continue
        rid_map[rel.rId] = s.part.relate_to(rel.target_ref if rel.is_external else rel.target_part, rel.reltype, rel.is_external)
    bg = source_content._element.cSld.find('{http://schemas.openxmlformats.org/presentationml/2006/main}bg')
    if bg is not None:
        bg = deepcopy(bg)
        for e in bg.iter():
            for key, val in list(e.attrib.items()):
                if key.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}') and val in rid_map:
                    e.set(key, rid_map[val])
        s._element.cSld.insert(0, bg)
    for sh in source_content.shapes:
        e = deepcopy(sh._element)
        for item in e.iter():
            for key, val in list(item.attrib.items()):
                if key.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}') and val in rid_map:
                    item.set(key, rid_map[val])
        s.shapes._spTree.insert_element_before(e, 'p:extLst')
    remove(by_id(s, 3))
    title = by_id(s, 2)
    title.left, title.top, title.width, title.height = Inches(.58), Inches(.37), Inches(7.02), Inches(.76)
    set_text(title, titles[CURRENT - 6], 22, C.LIGHT_1, True, min_size=20, line=1.05, after=0, margin=0)
    sec = sections[CURRENT - 6].split('\n## Дополнение:', 1)[0]
    speaker = re.search(r'Для докладчика:(.*?)(?=\n\nИсточники:|\n---|\Z)', sec, re.S)
    extra = '' if not speaker else '\n\nДля докладчика:' + speaker.group(1).strip()
    notes(s, f'Источник: slides.md, раздел {CURRENT - 5}.' + extra)
    return s


# 6 / custom 1.
CURRENT = 6
s = clone_content()
box(s, .72, 1.68, 11.82, 1.24, [
    '96 камер: одна машина меняет ракурс, масштаб и освещение.',
    'Номера в данных размыты. ТЗ запрещает использовать и остаточные признаки знака.',
    'Одной модели и цвета недостаточно: базовая модель взаимно путает 28 пар машин.',
], 17.1, min_size=16.4, after=5)
box(s, 1.4, 3.05, 4.2, .35, 'Запрос', 16, bold=True, after=0)
box(s, 7.6, 3.05, 4.65, .35, 'Та же машина на другой камере', 15.5, bold=True, after=0)
path = REPO / '04-solution/error-analysis/sheets/success_viewpoint_01.jpg'
picture(s, path, (9, 681, 237, 850), 1.4, 3.5, 4.25, 3.05, 'v825 — запрос, четвёртая строка, левая панель')
picture(s, path, (247, 681, 475, 850), 7.65, 3.5, 4.25, 3.05, 'v825 — та же машина на другой камере, средняя панель')
foot(s, 'Пример из нашей валидации, исходные веса; 28 пар — результат разбора базовой модели.')

# 7 / custom 2.
CURRENT = 7
s = clone_content()
box(s, .74, 1.67, 6.08, .48, 'Оператор передаёт кадр и рамку автомобиля через браузер.', 14.2, min_size=13.5, after=0)
ys = (2.2, 2.86, 3.52, 4.18, 4.84)
nodes = ('Оператор: кадр + рамка', 'Кроп машины', 'Вектор из 512 чисел', 'Поиск в галерее', 'Есть кандидат выше порога?')
for j, (y, label) in enumerate(zip(ys, nodes)):
    panel(s, 1.6, y, 4.15, .43, label, 13.9)
    if j:
        arrow(s, 3.68, ys[j-1] + .44, 3.68, y - .035)
arrow(s, 3.4, 5.31, 2.02, 5.83)
arrow(s, 3.94, 5.31, 5.27, 5.83)
box(s, 2.15, 5.39, .6, .3, 'Да', 12, bold=True, after=0)
box(s, 4.68, 5.39, .75, .3, 'Нет', 12, bold=True, after=0)
panel(s, .77, 5.87, 2.64, .72, 'Кандидаты с оценкой близости', 13.3)
panel(s, 3.72, 5.87, 3.10, .72, 'Пустой ответ:\n«Уверенного совпадения нет»', 12.1)
table(s, 7.15, 1.78, [1.72, 3.54], [.53, .83, 1.32, 1.42], [
    ['Артефакт', 'Что получает жюри'],
    ['submission.csv', 'Первые 10 кандидатов для каждого запроса'],
    ['embeddings.npy', 'Векторы по 512 чисел; сначала запросы, затем галерея, в порядке входных CSV'],
    ['candidates.csv', 'Кандидаты с оценкой близости; отсутствие строк для запроса означает отказ'],
], size=13, header=12.5)
box(s, 7.15, 6.0, 5.28, .68, 'Пакетный прогон создаёт 3 сдаваемых файла одной командой в готовом контейнере, без интернета.', 13.2, min_size=12.3, after=0)
foot(s, 'Браузер/API: косинусный поиск. Пакетная сдача: переранжирование. Одна команда — после подготовки образа и данных; сборка требует сети.')

# 8 / custom 3.
CURRENT = 8
s = clone_content()
box(s, .74, 1.68, 11.85, 1.79, [
    'OSNet-AIN: вход 208 × 208, вектор 512, веса 8,4 МиБ; инференс работает на CPU.',
    'На одном сплите OSNet даёт mAP 0,657, FastReID R50-IBN — 0,238.',
    'Переранжирование учитывает общих соседей: mAP растёт до 0,694 с теми же весами.',
    'На парах «ночь ↔ день» Rank-1 растёт на 0,117. Смена ракурса и света остаётся главной причиной ошибок: 58 % по разбору.',
], 16.2, min_size=15.4, after=6)
xs = (.74, 3.77, 6.8, 9.82)
for x, label in zip(xs, ('Кроп', 'RGB, 208 × 208', 'OSNet-AIN', 'Нормированный\nвектор 512')):
    panel(s, x, 3.9, 2.63, .71, label, 15)
for i in range(3):
    arrow(s, xs[i] + 2.67, 4.255, xs[i+1] - .05, 4.255)
arrow(s, 11.1, 4.65, 11.1, 4.97)
panel(s, 6.8, 5.03, 5.65, .56, 'Пакетный поиск по взаимным соседям', 14.6)
arrow(s, 9.63, 5.63, 9.63, 5.93)
panel(s, 6.8, 5.98, 5.65, .55, 'Ранжирование и порог отказа', 14.8)
box(s, .79, 4.98, 5.72, .98, [
    {'text': 'mAP 0,657 → 0,694', 'size': 27, 'bold': True, 'after': 7},
    {'text': '+0,037 с теми же весами', 'size': 19, 'bold': True},
], after=0)
box(s, .79, 6.08, 5.68, .45, 'OSNet-AIN выбран как компактный и быстрый.', 13.5, after=0)
foot(s, 'Наша валидация; mAP по 832 запросам с парой, вся галерея, протокол market. Качество пакетного режима не приписывается API.')

# 9 / custom 4.
CURRENT = 9
s = clone_content()
table(s, .75, 1.86, [4.23, 3.17, 4.4], [.67, 1.14, 1.39, 1.49], [
    ['Проверка', 'Что получили', 'Что это меняет'],
    ['Подставили случайные векторы', 'mAP 0,0099 при ожидаемых 0,0103', 'Контур показывает ожидаемый случайный уровень'],
    ['Разрешили совпадения той же машины с той же камеры', 'Rank-1 вырос 0,631 → 0,971', 'Такие совпадения исключаем из основной оценки'],
    ['Сравнили подбор на проверочной выборке и на других машинах', 'Завышение 0,001–0,012 mAP', 'Параметры переранжирования выбираем на отдельных идентичностях'],
], size=16, header=16)
foot(s, 'Контрольные опыты исходной модели. Основной mAP: 832 запроса с парой из 1110. Это исторические проверки.')

# 10 / custom 5.
CURRENT = 10
s = clone_content()
box(s, .75, 1.73, 6.30, 1.10, [
    {'text': '−0,003 mAP', 'size': 28, 'bold': True, 'after': 5},
    {'text': '95 % интервал [−0,011; +0,004]', 'size': 18, 'bold': True},
], after=0)
box(s, .75, 2.84, 6.35, .7, 'Закрыли пластину и такой же прямоугольник рядом на кузове: сравнили потерю качества при равной площади.', 14.5, min_size=13.8, after=0)
box(s, .75, 3.55, 6.35, .58, 'Расширенная маска: верхняя граница интервала +0,016 mAP.', 15.4, min_size=14.6, bold=True, after=0)
table(s, .78, 4.28, [4.65, 1.55], [.45, .43, .43, .77], [
    ['Что закрыто', 'mAP'], ['Ничего', '0,6566'], ['Зона пластины', '0,6513'],
    ['Тот же прямоугольник со сдвигом по вертикали', '0,6480'],
], size=12.7, header=12.4)
box(s, .78, 6.38, 6.26, .27, 'Для переранжирования и дообученной модели измерение нужно повторить.', 10.8, min_size=10.2, after=0)
box(s, 7.48, 1.78, 4.95, .42, 'Контроль покрытия пластины', 16, bold=True, after=0)
path = REPO / '04-solution/plate-ablation/pics/detector_vs_labels_0.png'
picture(s, path, (330, 286, 659, 540), 7.49, 2.33, 2.46, 2.58, 'Пластина: ячейка #5, такси')
picture(s, path, (662, 557, 974, 810), 10.10, 2.33, 2.46, 2.58, 'Пластина: ячейка #10, белый Lexus')
box(s, 7.49, 5.18, 5.01, .65, 'Красная рамка — детектор.\nЗелёная — ручная разметка.', 14.2, after=2)
box(s, 7.49, 6.0, 4.95, .62, 'Второй пример показывает неполное покрытие и объясняет, зачем проверяли расширенную маску.', 12.2, min_size=11.9, after=0)
foot(s, 'Наша валидация. Исходные веса, косинус; перенос на переранжирование пока не проверен.')

# 11 / custom 6.
CURRENT = 11
s = clone_content()
picture(s, REPO / '05-presentation/assets/f1_tnr_threshold.png', None, .62, 1.67, 7.82, 4.45, 'Готовый график F1 и TNR по порогу; без пересчёта')
box(s, 8.58, 1.77, 3.97, 1.05, 'Контроль на косинусе: максимум F1 при 75 % запросов с парой принимает 86 % запросов без пары.', 13.3, min_size=12.8, after=0)
box(s, 8.58, 2.92, 3.97, 1.14, 'Порог 0,4994 для переранжирования выбираем по худшему из F1 и TNR при долях запросов без пары 10 / 25 / 40 %.', 13.1, min_size=12.5, after=0)
box(s, 8.58, 4.13, 3.97, .35, 'Калиброванный порог; один протокол', 10.7, after=0)
table(s, 8.58, 4.54, [1.70, 1.12, 1.16], [.65, .54, .59], [
    ['Режим', 'Принято\nс парой', 'Ложно\nбез пары'],
    ['Косинус', '530 / 832', '84 / 278'],
    ['Переранжирование', '603 / 832', '62 / 278'],
], size=10.0, header=10.1)
box(s, .78, 6.05, 7.57, .40, 'Принимаем 603 из 832 запросов с парой; 229 отклоняем. Ошибочно принимаем 62 из 278 без пары.', 10.7, min_size=10.2, after=0)
box(s, .78, 6.48, 11.8, .39, 'TNR — доля корректных отказов среди запросов без пары. F1 здесь оценивает решение «принять / отказать» по наличию пары, а не правильность каждого кандидата.', 10.8, min_size=10.2, after=0)
foot(s, 'Наша калибровочная валидация, market / presence; независимого замера нового порога нет.')

# 12 / custom 7.
CURRENT = 12
s = clone_content()
box(s, .74, 1.67, 11.82, .94, [
    'Базовая модель ошиблась в первом кандидате на 307 из 832 запросов с парой.',
    '28 пар путаются взаимно: A выдаёт B, B выдаёт A. Это 84 из 307 ошибок первого кандидата — 27 %.',
    'Переранжирование исправляет 6 % ошибок двойников против 18 % остальных. Мелкие приметы требуют отдельной проверки.',
], 13.2, min_size=12.7, after=3, line=1.05)
table(s, .75, 2.9, [2.65, .65], [.53, .48, .96, .59, .56], [
    ['Главная причина', 'Доля'],
    ['Смена ракурса', '47 %'],
    ['Та же модель, цвет или ливрея у чужой машины', '36 %'],
    ['Смена освещения', '11 %'],
    ['Испорченный кроп', '6 %'],
], size=11.9, header=10.7)
box(s, .77, 6.15, 3.26, .60, 'Ручной просмотр: 64 случайно выбранных ошибки из 307; доли округлены.', 10.7, min_size=10.3, after=0)
path = REPO / '04-solution/error-analysis/sheets/twins_zoom_01.jpg'
for y, rownum, caption, top, bottom in [
    (3.19, 2, 'Запрос A → первой выдана B', 428, 765),
    (4.99, 4, 'Запрос B → первой выдана A', 1192, 1529),
]:
    box(s, 4.26, y - .34, 4.09, .27, caption, 11.8, bold=True, after=0)
    picture(s, path, (9, top, 477, bottom), 4.26, y, 1.96, 1.43, f'Двойники, строка {rownum}: запрос')
    picture(s, path, (488, top, 955, bottom), 6.33, y, 1.96, 1.43, f'Двойники, строка {rownum}: ошибочный первый кандидат')
box(s, 8.61, 2.86, 3.93, 1.27, 'В случайной выборке ошибок 4,7 % случаев связаны с несогласованной разметкой; интервал оценки — 0–10 %. На примере рамка выделяет соседнюю машину. Эти случаи ограничивают точность измерения качества.', 12.0, min_size=11.0, after=0, line=1.05)
path = REPO / '04-solution/error-analysis/sheets/label_check_B.jpg'
picture(s, path, (10, 443, 710, 836), 8.61, 4.39, 1.89, 1.10, 'v1012: полный кадр запроса, рамка на соседней машине')
picture(s, path, (720, 443, 1420, 836), 10.62, 4.39, 1.89, 1.10, 'v1012: полный кадр с той же меткой')
box(s, 8.61, 5.62, 3.93, .60, 'Одна метка, разные машины внутри рамок', 12.4, bold=True, after=0)
box(s, 4.26, 6.54, 8.26, .21, '27 % и 36 % — разные способы группировать те же ошибки первого кандидата; доли не складываются.', 9.5, after=0)
foot(s, 'Базовая модель, до переранжирования; наша валидация.')
notes(s, s.notes_slide.notes_text_frame.text + '\n\nВключён мягкий вариант А о разметке: решение о включении зафиксировано в structure.md. На слайде показаны обе стороны двух взаимных ошибок оранжевых машин и полные кадры нижней пары label_check_B.jpg. Это ограничение текущего признака, а не доказанная невозможность различить машины любым методом.')

# 13 / custom 8.
CURRENT = 13
s = clone_content()
for x, number, desc in [
    (.79, '39–64 мс / объект', 'Извлечение признака на CPU, batch 1: 39–64 мс на объект в локальных прогонах.'),
    (4.82, '8,4 МиБ / 2 ГБ', 'Веса — 8,4 МиБ при лимите ТЗ 2 ГБ.'),
    (8.85, '1,8 ГБ на 7440', 'Плотное переранжирование требует около 1,8 ГБ рабочих матриц на 7440 объектах; практический предел — порядка 10⁴.'),
]:
    box(s, x, 1.83, 3.72, .59, number, 21.6, bold=True, min_size=20, after=0)
    box(s, x, 2.57, 3.72, 1.26, desc, 14, min_size=13.4, after=0)
box(s, .79, 3.87, 11.75, .40, 'Сейчас API ищет точно; пакетный режим хранит плотные матрицы.', 14.7, after=0)
labels = ('Галерея около\n10⁶ объектов', 'Приближённый поиск\nсоседей — план', 'Ограниченный список\nкандидатов', 'Уточнение порядка\nи проверка порога')
for x, label in zip((.77, 3.78, 6.79, 9.8), labels):
    panel(s, x, 4.66, 2.64, .83, label, 13.7)
for x in (3.44, 6.45, 9.46):
    arrow(s, x, 5.08, x + .30, 5.08)
box(s, .79, 5.87, 11.72, .79, [
    {'text': '10⁶ объектов — целевой масштаб, [НЕТ ЗАМЕРА]', 'size': 18.6, 'bold': True, 'after': 7},
    {'text': 'Задержка поиска / RAM / полнота приближённого поиска — после проверки', 'size': 14.4},
], after=0)
foot(s, 'Наша валидация; локальный справочный тайминг. Сопоставимый замер оборудования жюри в исходном кратком отчёте не приведён.')

# 14 / custom 9.
CURRENT = 14
s = clone_content()
for x, label in ((.78, 'Браузер'), (4.82, 'API / OpenAPI'), (8.86, 'Qdrant')):
    panel(s, x, 1.86, 3.57, .62, label, 18)
arrow(s, 4.40, 2.17, 4.75, 2.17, both=True)
arrow(s, 8.43, 2.17, 8.78, 2.17, both=True)
box(s, 4.82, 2.57, 3.57, .32, 'контейнер 1', 12.5, align=PP_ALIGN.CENTER, after=0)
box(s, 8.86, 2.57, 3.57, .32, 'контейнер 2', 12.5, align=PP_ALIGN.CENTER, after=0)
for y, label in ((3.42, 'Готовый образ сервиса + CSV + изображения'), (4.7, 'Пакетный офлайн-прогон'), (5.94, '3 сдаваемых файла')):
    panel(s, .8, y, 5.9, .61, label, 15.3)
arrow(s, 3.75, 4.08, 3.75, 4.64)
arrow(s, 3.75, 5.36, 3.75, 5.89)
box(s, 7.14, 3.22, 5.35, 3.40, [
    'API и Qdrant работают в 2 контейнерах; браузерный клиент получает данные через API с OpenAPI.',
    'Офлайн-прогон обработал 1860 объектов; 3 сдаваемых файла побайтово совпали с запуском на хосте.',
    'Измерительный контур прошёл 60 тестов. Отключение сети подтверждено неудачной попыткой внешнего запроса.',
    'Аудит зафиксировал 18 проблем воспроизведения. Чистый git-клон пока не содержит полную поставку.',
], 14.2, min_size=13.5, after=9, line=1.08)
foot(s, 'Проверен основной офлайн-инференс. Для передачи нужны веса и готовые образы; полное воспроизведение исследований ещё не закрыто.')

# 15 / custom 10.
CURRENT = 15
s = clone_content()
table(s, .76, 1.79, [4.85, 6.96], [.58, 1.26, 1.10, .92, .89], [
    ['Что ограничивает решение сейчас', 'Что проверяем следующим'],
    ['≈58 % ошибок — ракурс и освещение', 'Две попытки дообучения на нашей обучающей части прироста не дали: 0,700 против 0,694, разброс 0,0134, p = 0,52, Rank-1 хуже. Дальше — данные заказчика в большем объёме, а не повтор того же'],
    ['27 % ошибок первого кандидата — взаимные двойники', 'Проверить локальные детали высокого разрешения и отказ при малом отрыве; показывать оператору различимые приметы'],
    ['Целевой масштаб 10⁶ объектов', 'Проверить приближённый поиск, память, задержку и потерю качества относительно точного поиска'],
    ['≈5 % в случайной выборке ошибок связаны с разметкой', 'Передать примеры на проверку; дообучение не исправляет неверную цель'],
], size=14.9, header=14.5)
box(s, .79, 6.63, 11.76, .23, 'Дообучение закрыто: две попытки прироста не дали, остаёмся на готовых весах. Проверка зоны номера и порог отказа в силе.', 10.6, min_size=10.2, after=0)
foot(s, 'Наша валидация. Приоритеты по разбору базовой модели; эффект будущих изменений ещё не измерен.')

# Normalize visible page numbers, keeping each original number field's position.
for n, s in enumerate(prs.slides, 1):
    CURRENT = n
    for sh in s.shapes:
        if sh.is_placeholder and str(sh.placeholder_format.type).startswith('SLIDE_NUMBER'):
            set_text(sh, str(n), 10.0, C.LIGHT_1 if n in (3, 5) or n >= 6 else C.ACCENT_5,
                     after=0, align=PP_ALIGN.CENTER, margin=0)
        elif sh.has_text_frame and 'Номер слайда' in sh.name:
            set_text(sh, str(n), 10.0, C.LIGHT_1 if n in (3, 5) or n >= 6 else C.ACCENT_5,
                     after=0, align=PP_ALIGN.CENTER, margin=0)

prs.core_properties.title = f'{team_name} — цифровой признак транспортного средства'
prs.core_properties.subject = 'ЛЦТ 2026, задача №7'
prs.core_properties.author = 'Команда'
prs.core_properties.last_modified_by = 'Команда'
prs.core_properties.comments = ''
prs.save(OUTPUT)

# Check readback and sensitive values without disclosing them in logs or reports.
readback = Presentation(OUTPUT)
assert len(readback.slides) == 15
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
print(json.dumps({'slides': 15, 'embedded_illustrations': len(MEDIA), 'contact_check': 'passed',
                  'text_boxes_checked': len(FIT), 'fit_warnings': overflows}, ensure_ascii=False, indent=2))
