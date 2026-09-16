"""Validate the deliverable without exporting any participant values."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

from lxml import etree
from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/artem/projects/hackathon-lct-vehicle-reid')
SOURCE = REPO / '00-task/assets/ЛЦТ2026 Шаблон презентации.pptx'
OUTPUT = ROOT / 'presentation.pptx'
prs = Presentation(OUTPUT)
template = Presentation(SOURCE)
assert len(prs.slides) == 15
empty = []
fonts = set()
pictures = 0
for index, slide in enumerate(prs.slides, 1):
    assert any(s.has_text_frame and s.text.strip() for s in slide.shapes)
    for shape in slide.shapes:
        if shape.is_placeholder and shape.has_text_frame and not shape.text.strip():
            empty.append((index, shape.shape_id))
        frames = [shape.text_frame] if shape.has_text_frame else [c.text_frame for row in shape.table.rows for c in row.cells] if shape.has_table else []
        for tf in frames:
            for paragraph in tf.paragraphs:
                for run in paragraph.runs:
                    if run.text.strip():
                        fonts.add(run.font.name or '(inherited)')
        if shape.shape_type == 13:
            pictures += 1
            with Image.open(BytesIO(shape.image.blob)) as image:
                image.verify()
assert not empty
assert fonts == {'Montserrat'}

# All retained fields of the mandatory block keep their exact template geometry.
allowed_removed = {1: {4}, 2: {2}, 3: {2, 3, 4, 5, 6}, 4: set(), 5: set()}
geometry_checked = 0
for index in range(1, 6):
    src = template.slides[index + 5]
    dst = prs.slides[index - 1]
    current = {s.shape_id: s for s in dst.shapes}
    for shape in src.shapes:
        if shape.shape_id in allowed_removed[index]:
            continue
        target = current[shape.shape_id]
        assert tuple(getattr(shape, k) for k in ('left', 'top', 'width', 'height')) == tuple(getattr(target, k) for k in ('left', 'top', 'width', 'height'))
        geometry_checked += 1

with ZipFile(SOURCE) as original, ZipFile(OUTPUT) as result:
    assert result.testzip() is None
    themes = [f for f in result.namelist() if f.startswith('ppt/theme/') and f.endswith('.xml')]
    assert all(original.read(f) == result.read(f) for f in themes)
    xml = {f: result.read(f) for f in result.namelist() if f.endswith('.xml')}
    for blob in xml.values():
        etree.fromstring(blob)
    for f in result.namelist():
        if f.endswith('.rels'):
            for rel in etree.fromstring(result.read(f)):
                assert not (rel.get('Type', '').endswith('/image') and rel.get('TargetMode') == 'External')

    # Read contact values only in memory and never echo them or put them in metadata.
    contacts = []
    for line in (ROOT / 'TEAM-DATA-персональные.md').read_text().splitlines():
        if line.startswith('|'):
            cells = [x.strip() for x in line.strip('|').split('|')]
            if len(cells) == 6 and cells[0].isdigit():
                contacts.extend(cells[3:5])
    assert len(contacts) == 10
    for value in contacts:
        parts = [name for name, blob in xml.items() if value.encode() in blob]
        assert parts == ['ppt/slides/slide3.xml'], 'Contact is missing or occurs outside the participant slide'

# Compare embedded illustration pixels with the exact selected source regions.
media = json.loads((ROOT / 'checks/illustrations.json').read_text())
for item in media:
    slide = prs.slides[item['slide'] - 1]
    if item['source'].endswith('.pptx'):
        src_pic = next(s for s in template.slides[5].shapes if s.shape_id == item['shape'])
        dst_pic = next(s for s in slide.shapes if s.shape_type == 13)
        assert src_pic.image.blob == dst_pic.image.blob
        continue
    shape = next(s for s in slide.shapes if s.name == item['label'])
    with Image.open(REPO / item['source']) as source:
        expected = source.crop(item['crop']) if item.get('crop') else source.copy()
        expected = expected.convert('RGB')
    with Image.open(BytesIO(shape.image.blob)) as actual:
        actual = actual.convert('RGB')
        assert actual.size == expected.size and actual.tobytes() == expected.tobytes()

fit = json.loads((ROOT / 'checks/fit_check.json').read_text())
assert all(item['fits'] for item in fit)
render = json.loads((ROOT / 'checks/render_check.json').read_text())
assert render['rendered_pages'] == 10 and not render['issues']

summary = {
    'slide_count': len(prs.slides),
    'text_present_on_each_slide': True,
    'original_template_pages': [7, 8, 9, 10, 11],
    'mandatory_original_shape_geometries_checked': geometry_checked,
    'embedded_pictures': pictures,
    'source_image_pixel_comparisons_passed': len(media),
    'empty_active_placeholders': [],
    'broken_pictures': [],
    'external_image_links': [],
    'explicit_fonts': sorted(fonts),
    'original_theme_bytes_unchanged': True,
    'all_xml_parsed': True,
    'zip_error': None,
    'contact_fields_only_in_participant_slide': True,
    'text_layout_estimates_checked': len(fit),
    'text_layout_estimate_failures': 0,
    'rendered_technical_pages': render['rendered_pages'],
    'rendered_native_text_blocks_matched': render['text_blocks_checked'],
    'bytes': OUTPUT.stat().st_size,
    'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
}
(ROOT / 'checks/validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
