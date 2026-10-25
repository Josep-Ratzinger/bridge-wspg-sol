"""Exportación con conservación exacta de puntos, RGB e identificadores."""
import numpy as np

DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('origen','u1'),('punto_id','<i4'),('reduccion_id','<u4'),('hipotesis_seccion','u1'),('forma_propuesta','u1'),('apoyo_id','<i2')])

def global_points(local,context):
    xy=context['origin_xy']+local[:,0,None]*context['axis_xy']+local[:,1,None]*context['normal_xy']
    return np.column_stack([xy,local[:,2]]).astype('f4')

def write_bridge(path,xyz,rgb,reduction,new,labels,forms,supports):
    header='ply\nformat binary_little_endian 1.0\ncomment APROXIMACION GEOMETRICA; no etiquetas semanticas de referencia\ncomment hipotesis_seccion 0 incierta/sin asignar 1 rectangular 2 cilindrica 3 viga\ncomment forma_propuesta indica geometria usada aunque hipotesis sea incierta\nelement vertex '+str(len(xyz)+len(new))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar origen\nproperty int punto_id\nproperty uint reduccion_id\nproperty uchar hipotesis_seccion\nproperty uchar forma_propuesta\nproperty short apoyo_id\nend_header\n'
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        for begin in range(0,len(xyz),500000):
            p=xyz[begin:begin+500000];block=np.zeros(len(p),dtype=DTYPE)
            for i,k in enumerate(('x','y','z')):block[k]=p[:,i]
            for i,k in enumerate(('red','green','blue')):block[k]=rgb[begin:begin+len(p),i]
            block['punto_id']=np.arange(begin,begin+len(p));block['reduccion_id']=reduction[begin:begin+len(p)];block['apoyo_id']=-1;block.tofile(stream)
        block=np.zeros(len(new),dtype=DTYPE)
        for i,k in enumerate(('x','y','z')):block[k]=new[:,i]
        block['red']=240;block['green']=141;block['blue']=35;block['origen']=1;block['punto_id']=-1;block['reduccion_id']=np.iinfo('u4').max;block['hipotesis_seccion']=labels;block['forma_propuesta']=forms;block['apoyo_id']=supports;block.tofile(stream)
