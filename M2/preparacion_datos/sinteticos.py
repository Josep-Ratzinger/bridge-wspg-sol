"""Escenas de superficies variadas para segmentación; no planos calculados.

Tipologías de apoyos y secciones documentadas por las fuentes del corpus.
Las dimensiones, combinaciones y defectos son propios, no mediciones civiles.
"""
import numpy as np


def generar(seed):
    r=np.random.default_rng(seed);parts=[];labs=[]
    L=r.uniform(20,85);W=r.uniform(7,20);H=r.uniform(3,10)
    spans=int(r.integers(2,5));depth=r.uniform(.55,1.6);slab=r.uniform(.22,.55)
    family=str(r.choice(['multicolumna','martillo','muro','pilotes']))
    section=str(r.choice(['circular','rectangular','octogonal','eliptica','achaflanada','ensanchada','oval_variable']))
    beam=str(r.choice(['T','I','trapezoidal','cajon_exterior']))
    rail=str(r.choice(['metal_horizontal','metal_diagonal','mixta','barrera']))
    curve=str(r.choice(['recto','arco','S'],p=[.3,.4,.3]));amplitude=r.uniform(.025,.16)*L
    phase=r.uniform(-.3,.3);slope=r.uniform(-.06,.06);camber=r.uniform(-.025,.025)
    def add(p,k):parts.append(np.asarray(p));labs.append(np.full(len(p),k,dtype='i4'))
    def box(c,e,n,k):
        e=np.asarray(e);p=r.uniform(-.5,.5,(n,3))*e
        area=e[[1,0,0]]*e[[2,2,1]];axis=r.choice(3,n,p=area/area.sum())
        p[np.arange(n),axis]=r.choice([-1,1],n)*e[axis]/2;add(p+c,k)
    def extrude(poly,a,b,n,k,axis=0,offset=(0,0,0),taper=0):
        poly=np.asarray(poly);end=np.roll(poly,-1,axis=0);length=np.linalg.norm(end-poly,axis=1)
        ed=r.choice(len(poly),n,p=length/length.sum());t=r.random((n,1))
        q=poly[ed]+t*(end[ed]-poly[ed]);p=np.empty((n,3));p[:,axis]=r.uniform(a,b,n)
        # Variation along the station, continuous and bounded.
        q[:,1]*=1+taper*np.sin(np.pi*(p[:,axis]-a)/max(b-a,1e-6))**2
        p[:,[j for j in range(3) if j!=axis]]=q;add(p+offset,k)
    def bar(a,b,radius,n,k):
        a=np.asarray(a);b=np.asarray(b);v=b-a;v=v/np.linalg.norm(v)
        u=np.cross(v,[0,0,1])
        if np.linalg.norm(u)<1e-6:u=np.array([1.,0,0])
        u/=np.linalg.norm(u);w=np.cross(v,u);theta=r.uniform(0,2*np.pi,n)
        add(a+r.random((n,1))*(b-a)+radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*w),k)
    box([0,0,-slab/2],[L,W,slab],9500,2)
    for sign in [-1,1]:box([sign*(L/2+4),0,-slab/2],[8,W,slab],2000,4)
    b=r.uniform(.35,.7);web=r.uniform(.10,.22)
    if beam=='T':poly=[[-b,0],[b,0],[b,-.18],[web,-.18],[web,-depth],[-web,-depth],[-web,-.18],[-b,-.18]]
    elif beam=='I':poly=[[-b,0],[b,0],[b,-.16],[web,-.16],[web,-depth+.16],[b,-depth+.16],[b,-depth],[-b,-depth],[-b,-depth+.16],[-web,-depth+.16],[-web,-.16],[-b,-.16]]
    else:poly=[[-b,0],[b,0],[b*.55,-depth],[-b*.55,-depth]]
    nbeams=int(r.integers(2,6))
    for y in np.linspace(-W*.38,W*.38,nbeams):extrude(poly,-L/2,L/2,1800,1,offset=(0,y,-slab),taper=r.uniform(0,.4))
    captop=-slab-depth;capthick=r.uniform(.5,1.15);capbottom=captop-capthick
    for x in np.linspace(-L/2,L/2,spans+1)[1:-1]:
        half=W*r.uniform(.43,.49);tip=r.uniform(.55,.92)
        cap=[[-half,captop],[half,captop],[half*tip,capbottom],[-half*tip,capbottom]]
        extrude(cap,x-r.uniform(.4,.8),x+r.uniform(.4,.8),2300,1)
        cols=1 if family in ['martillo','muro'] else int(r.integers(2,10))
        for y in (np.linspace(-W*.36,W*.36,cols) if cols>1 else [0.]):
            n=1500;z=r.uniform(-H+capbottom,capbottom,n);t=(z+H-capbottom)/H
            rad=r.uniform(.3,.6);theta=r.uniform(0,2*np.pi,n)
            if family=='muro':
                extrude([[-.45,-W*.25],[.45,-W*.25],[.45,W*.25],[-.45,W*.25]],-H+capbottom,capbottom,n,0,axis=2,offset=(x,0,0));continue
            if section in ['rectangular','achaflanada']:
                p=np.array([[-1,-.8],[1,-.8],[1,.8],[-1,.8]])
                if section=='achaflanada':p=np.array([[-.7,-1],[.7,-1],[1,-.7],[1,.7],[.7,1],[-.7,1],[-1,.7],[-1,-.7]])
                ed=r.integers(len(p),size=n);xy=rad*(p[ed]+r.random((n,1))*(np.roll(p,-1,axis=0)[ed]-p[ed]))
            elif section=='octogonal':
                a=np.floor(theta/(np.pi/4))*np.pi/4;p=np.column_stack([np.cos(a),np.sin(a)]);q=np.column_stack([np.cos(a+np.pi/4),np.sin(a+np.pi/4)])
                xy=rad*(p+r.random((n,1))*(q-p))
            else:
                xy=rad*np.column_stack([np.cos(theta),np.sin(theta)])
                if section in ['eliptica','oval_variable']:xy[:,1]*=r.uniform(1.2,1.8)
            flare=(r.uniform(.4,1.3)*np.maximum((t-.65)/.35,0)**2 if section in ['ensanchada','oval_variable'] or family=='martillo' else r.uniform(-.15,.3)*t)
            xy*=1+flare[:,None]
            lean=r.uniform(-.15,.15);xy[:,0]+=lean*H*(t-.5)
            add(np.column_stack([xy[:,0]+x,xy[:,1]+y,z]),0)
    bottom=capbottom-H
    for sign in [-1,1]:box([sign*(L/2-.4),0,(captop+bottom)/2],[.8,W*.95,captop-bottom],1900,7)
    barrier=[[-.28,0],[.28,0],[.12,.85],[-.12,.85]]
    for y in [-W/2,W/2]:
        if rail=='barrera':extrude(barrier,-L/2,L/2,2500,3,offset=(0,y,0));continue
        zbase=.35 if rail=='mixta' else 0.
        if rail=='mixta':box([0,y,.175],[L,.3,.35],1300,3)
        stations=np.linspace(-L/2,L/2,int(L/r.uniform(1.5,2.7))+1)
        for x in stations:bar([x,y,zbase],[x,y,1.2],.055,100,3)
        for z in [.65,1.2]:bar([-L/2,y,z],[L/2,y,z],.045,int(L*55),3)
        if rail=='metal_diagonal':
            for a,b in zip(stations[:-1],stations[1:]):bar([a,y,zbase],[b,y,1.2],.035,100,3)
    median=bool(r.integers(2))
    if median:extrude(barrier,-L/2,L/2,2000,5,offset=(0,r.uniform(-.5,.5),0))
    box([0,0,bottom-.15],[L+12,W+8,.3],6500,6)
    xyz=np.concatenate(parts);labels=np.concatenate(labs);station=xyz[:,0].copy();t=station/L
    if curve=='arco':
        bend=amplitude*(1-np.cos(np.pi*t));deriv=amplitude*np.pi/L*np.sin(np.pi*t)
    elif curve=='S':
        bend=amplitude*np.sin(2*np.pi*t+phase);deriv=amplitude*2*np.pi/L*np.cos(2*np.pi*t+phase)
    else:bend=np.zeros(len(xyz));deriv=np.zeros(len(xyz))
    angle=np.arctan(deriv);lateral=xyz[:,1].copy()
    xyz[:,0]=station-lateral*np.sin(angle);xyz[:,1]=bend+lateral*np.cos(angle)
    xyz[:,2]+=slope*station+camber*L*(t*t-.25)
    # Occlusion of patches, missing faces, scan-density imbalance and ghosts.
    density=np.clip(.75+.22*np.sin(station/L*5+r.uniform(-3,3)),.3,.98)
    keep=r.random(len(xyz))<density
    for _ in range(int(r.integers(2,7))):
        center=xyz[r.integers(len(xyz))];scale=np.array([r.uniform(.5,3),r.uniform(.5,2),r.uniform(.5,3)])
        inside=np.square((xyz-center)/scale).sum(1)<1;keep &= ~(inside&(r.random(len(xyz))<.8))
    xyz=xyz[keep];labels=labels[keep]
    xyz+=r.normal(0,r.uniform(.004,.025),(len(xyz),3))
    ghost=r.choice(len(xyz),int(.025*len(xyz)),replace=False)
    xyz=np.concatenate([xyz,xyz[ghost]+r.normal(0,.06,(len(ghost),3))]);labels=np.r_[labels,labels[ghost]]
    return xyz.astype('f4'),labels,dict(semilla=seed,longitud=L,ancho=W,altura=H,vanos=spans,
        familia_apoyo=family,seccion_columna='muro_rectangular' if family=='muro' else section,seccion_viga=beam,baranda=rail,trazado=curve,
        amplitud_curva=amplitude if curve!='recto' else 0,separador=median,combinacion_independiente=True,
        dimensiones='propias, no copiadas de planos',validacion_estructural=False)

