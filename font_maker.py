from tqdm import tqdm
import os
import json
import glob


def bmp_to_svg(save_bmp):
    prBar = tqdm(glob.glob(f"{save_bmp}/*.bmp"))
    save_svg = save_bmp + '_svg'
    os.makedirs(save_svg, exist_ok=True)
    for file in prBar:
        filename = os.path.basename(file)[:-3] + 'svg'
        filepath = os.path.join(save_svg, filename)
        os.system(f'potrace {file} --svg -o {filepath}')
        prBar.set_description(f'filepath : {filepath}')
    return save_svg

def make_font(save_svg):
    save_font = 'static/results/fonts'
    os.makedirs(save_font, exist_ok=True)
    x = json.load(open('example.json'))
    print(len(os.listdir(save_svg)))
    for idx, i in enumerate(range(0xac00, 0xd7a4)):
        x['glyphs'][hex(i)] = {'src':f'{save_svg}/{idx}.svg', 'width':128}
    del x['# vim: set et sw=2 ts=2 sts=2:']
    del x['glyphs']['0x3f']
    del x['glyphs']['0xab']
    del x['glyphs']['0x263a']
    del x['glyphs']['0x1f304']
    del x['glyphs']['0x2723']
    fontname = os.path.basename(save_svg)[:-4]
    x['sfnt_names'] = [['Korean', 'Copyright', 'Copyright (c) 2022 by MyFontStyler'],
            ['Korean', 'Family', fontname],
            ['Korean', 'SubFamily', 'Regular'],
            ['Korean', 'UniqueID', fontname + ' 2022-05-27'],
            ['Korean', 'Fullname', fontname + 'Regular'],
            ['Korean', 'Version', 'Version 001.000'],
            ['Korean', 'PostScriptName', fontname + '-Regular']]
    fontname_short = fontname.split('_')[-1]
    x['output'] = [f'{save_font}/{fontname_short}.ttf', '{save_font}/{fontname_short}.otf'] # name to convert ttf
    try:
        f = open(f'{fontname}.json', 'w')
        json.dump(x, f)
    except:
        # 에러 발생할 경우 한 번 더 실행
        f = open(f'{fontname}.json', 'w')
        json.dump(x, f)
    return fontname

def run_make_font(fontname):
    os.system(f"fontforge -lang=py -script svgs2ttf.py {fontname}.json") 
