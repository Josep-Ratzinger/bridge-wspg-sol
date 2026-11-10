"""Figura comparativa de Lurín a partir de las predicciones completas del M3."""
import argparse
from pathlib import Path
import numpy as np
from .rutas import DATOS
from PIL import Image, ImageDraw, ImageFont

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--resultados', type=Path, default=DATOS / 'experimento/resultados')
    args = p.parse_args()
    out = args.resultados
    spg = args.repo / 'M1/SPG/resultados'
    with np.load(spg / 'superpuntos/superpuntos.npz') as f:
        groups = f['celda_a_superpunto']
    with np.load(spg / 'entrada/nube_completada.npz') as f:
        xyz, origin = f['xyz'], f['origen']
    with np.load(args.repo / 'M1/clases/resultados/clases.npz') as f:
        arrays = [f['clase_propuesta']]
    for arm in ('sin_aumento', 'uniforme', 'dirigido'):
        with np.load(out / arm / 'predicciones.npz') as f:
            arrays.append(f['clase_por_superpunto'][groups])
    names = ('Columnas', 'Vigas/cabezales', 'Tablero', 'Barandas', 'Accesos',
             'Separador', 'Suelo/base', 'Estribos', 'Vegetación', 'Señales')
    palette = np.array([(230,65,60),(245,155,35),(45,125,225),(150,70,200),
                        (55,175,105),(35,200,215),(155,105,65),(200,170,120),
                        (40,110,40),(245,210,30),(173,173,173)], dtype='u1')
    titles = ('Anotación asistida', 'Sin aumento adicional', 'Aumento uniforme', 'Aumento dirigido')
    image = Image.new('RGB', (1700, 1300), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    heading = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 32)
    draw.text((850, 26), 'Lurín · WSPG + SOL con tres estrategias de entrenamiento', fill='#152439', font=heading, anchor='mt')
    rng = np.random.default_rng(42)
    views = [(0,1,np.ones(len(xyz),bool),'Planta XY'),
             (1,2,(xyz[:,0]>5.8)&(xyz[:,0]<10.7),'Apoyos YZ')]
    for col, (a,b,region,view) in enumerate(views):
        ids = np.flatnonzero(region)
        idx = rng.choice(ids, min(len(ids), 200000), replace=False)
        coords = xyz[idx][:,[a,b]]
        low, high = np.percentile(coords, [.05,99.95], axis=0)
        scale = min(770/(high[0]-low[0]),220/(high[1]-low[1]))
        px = np.rint((coords-(high+low)/2)*scale*[1,-1]+[400,120]).astype(int)
        inside = ((px>=0)&(px<[800,240])).all(1)
        x,y = px[inside].T
        for row,(title,labels) in enumerate(zip(titles,arrays)):
            left, top = 35 + col*830, 77 + row*283
            draw.text((left+400,top),f'{title} · {view}',fill='#152439',font=font,anchor='mt')
            canvas = np.full((240,800,3), 250, dtype='u1')
            c = labels[idx[inside]]
            canvas[y,x] = palette[np.where(c<0,10,c)]
            image.paste(Image.fromarray(canvas),(left,top+32))
            draw.rectangle((left,top+32,left+799,top+271),outline='#d4dbe3')
    for i,name in enumerate(names):
        x,y = 55+(i%5)*330, 1221+(i//5)*32
        draw.rectangle((x,y+3,x+19,y+21),fill=tuple(palette[i]))
        draw.text((x+30,y),name,fill='#152439',font=font)
    image.save(out / 'comparacion_lurin.png')

if __name__ == '__main__':
    main()





