# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:00:16 2022

@author: José
"""

import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#%% funciones
def dibuja_truss(nudo1, nudo2, color):
    """
    Function for drawing each truss of the model.

    Parameters
    ----------
    nudo1 : int, label of the first joint of the truss
    nudo2 : int, label of the second joint of the truss
    color : str, color code according to matplotlib library
    """
    x1, y1, z1 = ops.nodeCoord(nudo1)
    x2, y2, z2 = ops.nodeCoord(nudo2)
    
    plt.plot([x1,x2],[y1,y2],[z1,z2],color)

def dibuja_equals(nudo1, nudo2, color):
    """
    Function for marking the equal constraints of the model

    Parameters
    ----------
    nudo1 : int, master joint label
    nudo2 : int, constrained joint label
    color : str, color code according to matplotlib library
    """
    x1, y1, z1 = ops.nodeCoord(nudo1)
    x2, y2, z2 = ops.nodeCoord(nudo2)
    plt.plot([x1,x2],[y1,y2],[z1,z2],color)

def dibuja_pared(nudos_pared, nudo_ventana):
    """
    Function to draw the joints of the model

    Parameters
    ----------
    nudos_pared : list of joint labels of the model
    nudo_ventana : list of joint labels of the window/door opening

    Returns
    -------
    ax : axes for additional drawings
    px : list of floats, x coordinate of each joint
    py : list of floats, y coordinate of each joint
    pz : list of floats, z coordinate of each joint
    """
    px, py, pz = [], [], []
    for i in range(nudos_pared[0][0][0],nudos_pared[-1][-1][-1]+1):
        if i not in nudo_ventana:
            px.append(ops.nodeCoord(i)[0])
            py.append(ops.nodeCoord(i)[1])
            pz.append(ops.nodeCoord(i)[2])
    
    ax.scatter(px,py,pz,s=5)
    minimo = np.min([px,py,pz])
    maximo = np.max([px,py,pz])
    ax.set_xlim3d([minimo,maximo])
    ax.set_ylim3d([minimo,maximo])
    ax.set_zlim3d([minimo,maximo])
    plt.show()
    
    return ax, px, py, pz

def dibuja_portico(col1,col2,vig,ax):
    """
    Function to draw the frames of the model

    Parameters
    ----------
    col1 : list of labels of joints of the left column of the model
    col2 : list of labels of joints of the right column of the model
    vig : list of labels of joints of the upper beam of the model
    ax : axes to keep drawing the model

    Returns
    -------
    px : list of floats, x coordinate of each joint
    py : list of floats, y coordinate of each joint
    pz : list of floats, z coordinate of each joint
    """
    px, py, pz = [], [], []   
    for i in range(len(vig)):
        px.append(ops.nodeCoord(vig[i])[0])
        py.append(ops.nodeCoord(vig[i])[1])
        pz.append(ops.nodeCoord(vig[i])[2])
        
    for i in range(len(col1)):
        px.append(ops.nodeCoord(col1[i])[0])
        py.append(ops.nodeCoord(col1[i])[1])
        pz.append(ops.nodeCoord(col1[i])[2])
    
    for i in range(len(col2)):
        px.append(ops.nodeCoord(col2[i])[0])
        py.append(ops.nodeCoord(col2[i])[1])
        pz.append(ops.nodeCoord(col2[i])[2])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.scatter(px,py,pz,s=5)
    minimo = np.min([px,py,pz])
    maximo = np.max([px,py,pz])
    ax.set_xlim3d([minimo,maximo])
    ax.set_ylim3d([minimo,maximo])
    ax.set_zlim3d([minimo,maximo])
    plt.show()
    
    return px, py, pz

# def dibuja_deformada(px, py, pz, archivo, factor,xyz):
#     dx = np.loadtxt(archivo)

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     if xyz == 'x':
#         p = ax.scatter(np.array(px)+dx[-1]*factor,py,pz,c=dx[-1,:],cmap = 'Oranges')
#     elif xyz == 'y':
#         p = ax.scatter(px,np.array(py)+dx[-1]*factor,pz,c=dx[-1,:],cmap = 'Oranges')
#     else:
#         p = ax.scatter(px,py,np.array(pz)+dx[-1]*factor,c=dx[-1,:],cmap = 'Oranges')
#     ax.set_xlim3d([0,750])
#     ax.set_ylim3d([0,750])
#     ax.set_zlim3d([0,750])
#     fig.colorbar(p)
#     plt.show()



#%% construcción del modelo
def modelo(direccion, nlong, nz, long, esp, alt, ti, parametros_mamposteria,
           parametros_portico, secciones, nudo0, elemento0, material0, 
           xov, zov, Lv, Hv, x0=0, y0=0, z0=0, nesp=1, col_izq=1, 
           nudos_ci=[], col_der=1, nudos_cd=[], nudo_vig_izq=None, nudo_vig_der=None):
    """
    Función para generar de manera rápida la geometría de un pórtico relleno de mampostería,
    utilizando el modelo de puntales de Pirsaheb et al (2020)
    Parameters
    ----------
    direccion : string "x" o "y", indicando dirección paralela al plano del muro
    nlong : número de espacios en el mallado de la dirección paralela al plano del muro
    nz : número de espacios en el mallado de la dirección de la altura del muro
    long : longitud del muro
    esp : espesor del muro
    alt : altura del muro
    ti : ubicación de puntales verticales en referencia a cara externa de la pared
    parametros_mamposteria : [fc, fcu, ft, ft2, E, nu, G, Gf]
        parámetros para los puntales del modelo de mampostería, referidos al Concrete02 de opensees
        fc, fcu : resistencia máxima y última
        ft, ft2 : resistencia a tracción para elementos diagonales y horizontales/verticales, respectivamente
        E, nu, G : módulo de elasticidad, de Poisson y de corte de la mampostería
        Gf : valor de energía de fractura, usado (se supone) para regularizar el mallado
    parametros_portico : [b1, h1, b2, h2, b3, h3, rec_col, recv, Fc, Fy]
        b1, h1, b2, h2 : dimensiones de las columnas 1 y 2, respectivamente, en dirección perpendicular y paralela al muro, respectivamente
        b3, h3 : dimensiones de la viga
        rec_col, recv : recubrimientos en columnas y vigas
        Fc, Fy : resistencia a la compresión en el hormigón y fluencia en el acero de refuerzo de vigas y columnas
    secciones : tags de los beamIntegration de las secciones de la columna, viga, columna que conforman el pórtico
    
    nudo0 : tag del nudo inicial del modelo
    elemento0 : tag del elemento inicial del modelo
    material0 : tag del material inicial del modelo
    xov, zov : coordenadas locales del punto inicial de la ventana
    Lv, Hv : longitud y altura de la ventana
    
    x0, y0, z0 : coordenadas del punto de referencia para insertar la pared
    nesp : número de espacios en el mallado de la dirección perpendicular al plano del muro... por ahora sólo funciona si nesp = 1
    col_izq, col_der : variable con valor 1 si la columna respectiva debe modelarse y 0 si no debe modelarse
    nudos_ci, nudos_cd : listas con los tags de los nudos que se utilizarán para unir la pared a otra columna
    nudo_vig_izq, nudo_vig_der : tags de los nudos de las columnas izq y der, respectivamente, a la altura de la viga (primer o último término de la lista "vig" de la pared de conexión)
                                El valor es distinto de None si col_izq o col_der, respectivamente, son cero.
    
    Returns
    -------
    None.

    """
    ini = time.process_time()
    if direccion == 'x':
        geom = 1
    elif direccion == 'y':
        geom = 2
    [fc, fcu, ft, ft2, epsc0, nu] = parametros_mamposteria
    [b1, h1, b2, h2, b3, h3, rec_col, recv, Fc, Fy] = parametros_portico
    # ops.wipe()
    ops.model('basic','-ndm',3,'-ndf',3)
    
    slong = long/nlong
    sesp = esp/nesp
    sz = alt/nz
    
    Lb = np.sqrt(slong**2 + sz**2)
    LbOv = np.sqrt(sesp**2 + sz**2)
    LbOh = np.sqrt(slong**2 + sesp**2)
    
    peso = slong*sesp*sz*0.8*9810/1000000000
    masa = peso/9.81/1000
    thick = [ti,sesp-ti]
    
    E = np.abs(2*fc/epsc0)
    G = E/(2*(1+nu))
    Gf = 73*np.abs(fc)**0.18 /1000# para ceb fib, hormigón, ec 5.1-9
    eps_20 =-( Gf/(0.6*np.abs(fc)*Lb) -0.8*np.abs(fc)/E + np.abs(epsc0))
    epscu = eps_20*5
    Ets = np.abs(0.01*fc/epsc0)
    print('DATOS INICIALES',flush=True)
    
    ###########################################################
    # Creación de nudos
    nudo = nudo0
    
    # pared
    nudo_i = []
    base_nodes = []
    nudo_ventana = []
    for k in range(nz+1):
        nudo_i.append([])
        for j in range(nesp+1):
            nudo_i[k].append([])
            for i in range(nlong+1):
                if i*slong <= xov or i*slong >= xov+Lv or k*sz <= zov or k*sz >= zov+Hv:
                    if direccion == 'x':
                        ops.node(nudo,x0+i*slong,y0+thick[j],z0+k*sz,'-mass',masa/2,masa/2,masa/10000)
                    elif direccion == 'y':
                        ops.node(nudo,x0+thick[j],y0+i*slong,z0+k*sz,'-mass',masa/2,masa/2,masa/10000)
                    else:
                        print('Error en definición del parámetro "direccion"')
                    
                    if k == 0:
                        base_nodes.append(nudo)
                else:
                    nudo_ventana.append(nudo)
                nudo_i[k][j].append(nudo)
                nudo += 1
    
    ops.fixZ(0.0,1,1,1)
    print('NUDOS INGRESADOS')
    
    ax, pxp, pyp, pzp = dibuja_pared(nudo_i, nudo_ventana)
    ###########################################################
    # Creación de elementos tipo truss
    # áreas
    Ab1 = sesp*sz *.24 # horizontales x
    Ab2 = slong*sz /3 # horizontales y
    Ab3 = slong*G*LbOv**3/(4.8*E*sesp**2) /3 # diagonal horizontal
    Ab4 = slong*sesp *.24 # verticales
    Ab5 = sz*G*LbOh**3/(4.8*E*sesp**2) /3 # diagonal vertical
    Ab6 = sesp*G*Lb**3/(4.8*E*slong**2) *.9 # diagonal in plane
    
    # materiales
    ops.uniaxialMaterial('Concrete02',material0+1,fc*Ab1,epsc0*slong/sesp,fcu*Ab1,epscu*slong/sesp,0.1,ft2*Ab1,Ets*Ab1)
    ops.uniaxialMaterial('Concrete02',material0+2,fc*Ab2,epsc0*sesp/sesp,fcu*Ab2,epscu*sesp/sesp,0.1,ft2*Ab2,Ets*Ab2)
    ops.uniaxialMaterial('Concrete02',material0+3,fc*Ab3,epsc0*LbOv/sesp,fcu*Ab3,epscu*LbOv/sesp,0.1,ft*Ab3,Ets*Ab3)
    ops.uniaxialMaterial('Concrete02',material0+4,fc*Ab4,epsc0*sz/sesp,fcu*Ab4,epscu*sz/sesp,0.1,ft2*Ab4,Ets*Ab4)
    ops.uniaxialMaterial('Concrete02',material0+5,fc*Ab5,epsc0*LbOh/sesp,fcu*Ab5,epscu*LbOh/sesp,0.1,ft*Ab5,Ets*Ab5)
    ops.uniaxialMaterial('Concrete02',material0+6,fc*Ab6,epsc0*Lb/sesp,fcu*Ab6,epscu*Lb/sesp,0.1,ft*Ab6,Ets*Ab6)
    
    # horizontales en x
    elem = elemento0
    for k in range(nz+1):
        for j in range(nesp+1):
            for i in range(nlong):
                nudo1 = nudo_i[k][j][i]
                nudo2 = nudo_i[k][j][i+1]
                if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                    ops.element('Truss', elem, nudo1, nudo2, 1,1)
                    dibuja_truss(nudo1, nudo2, 'c')
                    elem += 1
                
    # horizontales en y
    for k in range(nz+1):
        for i in range(nlong+1):
            for j in range(nesp):
                nudo1 = nudo_i[k][j][i]
                nudo2 = nudo_i[k][j+1][i]
                if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                    ops.element('Truss', elem, nudo1, nudo2, 1,2)
                    elem += 1
                    dibuja_truss(nudo1, nudo2, 'y')
                if i < nlong:
                    # diagonal 1 OOP horizontal
                    nudo1 = nudo_i[k][j][i]
                    nudo2 = nudo_i[k][j+1][i+1]
                    if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                        ops.element('Truss', elem, nudo1, nudo2, 1,3)
                        elem += 1
                        dibuja_truss(nudo1, nudo2, 'g')
                    # diagonal 2 OOP horizontal
                    nudo1 = nudo_i[k][j+1][i]
                    nudo2 = nudo_i[k][j][i+1]
                    if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                        ops.element('Truss', elem, nudo1, nudo2, 1,3)
                        elem += 1
                        dibuja_truss(nudo1, nudo2, 'g')
                   
    # verticales
    for j in range(nesp+1):
        for i in range(nlong+1):
            for k in range(nz):
                nudo1 = nudo_i[k][j][i]
                nudo2 = nudo_i[k+1][j][i]
                if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                    ops.element('Truss', elem, nudo1, nudo2, 1,4)
                    elem += 1
                    dibuja_truss(nudo1, nudo2, 'b')
                if j < nesp:
                    # diagonal 1 OOP horizontal
                    nudo1 = nudo_i[k][j][i]
                    nudo2 = nudo_i[k+1][j+1][i]
                    if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                        ops.element('Truss', elem, nudo1, nudo2, 1,5)
                        elem += 1
                        dibuja_truss(nudo1, nudo2, 'm')
                    
                    # diagonal 2 OOP horizontal
                    nudo1 = nudo_i[k+1][j][i]
                    nudo2 = nudo_i[k][j+1][i]
                    if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana:
                        ops.element('Truss', elem, nudo1, nudo2, 1,5)
                        elem += 1
                        dibuja_truss(nudo1, nudo2, 'm')
    
    for k in range(nz):
        for j in range(nesp+1):
            for i in range(nlong):
                # diagonales IP izq abajo a der arriba
                nudo1 = nudo_i[k][j][i]
                nudo2 = nudo_i[k+1][j][i+1]
                nudo3 = nudo_i[k+1][j][i]
                nudo4 = nudo_i[k][j][i+1]
                if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana and nudo3 not in nudo_ventana and nudo4 not in nudo_ventana:
                    ops.element('Truss', elem, nudo1, nudo2, 1,6)
                    elem += 1
                    dibuja_truss(nudo1, nudo2, 'k')
                
                # diagonales IP izq arriba a der abajo
                
                if nudo1 not in nudo_ventana and nudo2 not in nudo_ventana and nudo3 not in nudo_ventana and nudo4 not in nudo_ventana:
                    ops.element('Truss', elem, nudo3, nudo4, 1,6)
                    elem += 1
                    dibuja_truss(nudo3, nudo4, 'k')
    print('MODELO TRUSS COMPLETO')   
    
    ###########################################################
    # pórtico
    ops.model('basic','-ndm',3,'-ndf',6) 
    
    # viga
    vig = []
    if col_izq == 1:
        if direccion == 'x':
            ops.node(nudo,x0-b1/2,y0+esp/2,z0+alt+h3/2)
        elif direccion == 'y':
            ops.node(nudo,x0+esp/2,y0-b1/2,z0+alt+h3/2)
        vig.append(nudo)
        nudo += 1
    else:
        vig.append(nudo_vig_izq)
    for i in range(nlong+1):
        if direccion == 'x':
            ops.node(nudo,x0+slong*i,y0+esp/2,z0+alt+h3/2)
        elif direccion == 'y':
            ops.node(nudo,x0+esp/2,y0+slong*i,z0+alt+h3/2)
        for j in range(len(nudo_i[nz])):
            ops.equalDOF(nudo,nudo_i[-1][j][i],1,2,3)
            dibuja_equals(nudo, nudo_i[-1][j][i], 'c')
        vig.append(nudo)
        nudo += 1
    if col_der == 1:
        if direccion == 'x':
            ops.node(nudo,x0+long+b2/2,y0+esp/2,z0+alt+h3/2)
        elif direccion == 'y':
            ops.node(nudo,x0+esp/2,y0+long+b2/2,z0+alt+h3/2)
        vig.append(nudo)
        nudo += 1
    else:
        vig.append(nudo_vig_der)
    
    # columna 1
    if col_izq == 1:
        col1 = []
        for i in range(nz+1):
            if direccion == 'x':
                ops.node(nudo,x0-b1/2,y0+esp/2,z0+i*sz)
            elif direccion == 'y':
                ops.node(nudo,x0+esp/2,y0-b1/2,z0+i*sz)
            for j in range(len(nudo_i[i])):
                ops.equalDOF(nudo,nudo_i[i][j][0],1,2,3)
                dibuja_equals(nudo, nudo_i[i][j][0], 'c')
            col1.append(nudo)
            nudo += 1
        ops.fix(col1[0],1,1,1,1,0,1)
        base_nodes.append(col1[0])
    else:
        for i in range(nz+1):
            for j in range(len(nudo_i[i])):
                ops.equalDOF(nudos_ci[i],nudo_i[i][j][0],1,2,3)
                dibuja_equals(nudos_ci[i], nudo_i[i][j][0], 'c')
        col1 = nudos_ci
    
    
    # columna 2
    if col_der == 1:
        col2 = []
        for i in range(nz+1):
            if direccion == 'x':
                ops.node(nudo,x0+long+b2/2,y0+esp/2,z0+i*sz)
            elif direccion == 'y':
                ops.node(nudo,x0+esp/2,y0+long+b2/2,z0+i*sz)
            for j in range(len(nudo_i[i])):
                ops.equalDOF(nudo,nudo_i[i][j][-1],1,2,3)
                dibuja_equals(nudo, nudo_i[i][j][-1], 'c')
            col2.append(nudo)
            nudo += 1
        ops.fix(col2[0],1,1,1,1,0,1)
        base_nodes.append(col2[0])
    else:
        for i in range(nz+1):
            for j in range(len(nudo_i[i])):
                ops.equalDOF(nudos_cd[i],nudo_i[i][j][-1],1,2,3)
                dibuja_equals(nudos_cd[i], nudo_i[i][j][-1], 'c')
        col2 = nudos_cd
    
     
    # condiciones de borde de pórtico
    if direccion == 'x':
        ops.node(nudo+1,x0-b1/2,y0+esp/2,z0)
        ops.node(nudo+2,x0+long+b2/2,y0+esp/2,z0)
    elif direccion == 'y':
        ops.node(nudo+1,x0+esp/2,y0-b1/2,z0)
        ops.node(nudo+2,x0+esp/2,y0+long+b2/2,z0)
    
    
    base_nodes.append(nudo+1)
    base_nodes.append(nudo+2)
    
    ops.fix(nudo+1,1,1,1,1,1,1)
    ops.fix(nudo+2,1,1,1,1,1,1)
    
    # BASE DE COLUMNAS => OJO, PENDIENTE ACTUALIZAR CON ALGÚN MODELO ESPECÍFICO
    teta = [1.52e6, 1.51e8, 4.87e6, 8.58, 0.06, 0.29, 0, 0, 0.27]
    ops.uniaxialMaterial('Hysteretic', material0+1000, teta[0], teta[0]/teta[1], teta[0]*teta[3] , teta[0]/teta[1] + teta[0]/teta[2], -teta[0], -teta[0]/teta[1], -teta[0]*teta[3] ,-teta[0]/teta[1]-teta[0]/teta[2], teta[4], teta[5], teta[6], teta[7], teta[8])
    if direccion == 'x':
        ops.element('zeroLength', elem, nudo+1, col1[0], '-mat', 1000, '-dir', 5)
        elem += 1
        ops.element('zeroLength', elem, nudo+2, col2[0], '-mat', 1000, '-dir', 5)
    elif direccion == 'y':
        ops.element('zeroLength', elem, nudo+1, col1[0], '-mat', 1000, '-dir', 4)
        elem += 1
        ops.element('zeroLength', elem, nudo+2, col2[0], '-mat', 1000, '-dir', 4)
    elem += 1
    
    
    # elementos col1
    if col_izq == 1:
        for i in range(nz):
            ops. element('dispBeamColumn', elem, col1[i], col1[i+1], geom, secciones[0])
            dibuja_truss(col1[i], col1[i+1], 'b')
            elem += 1
        ops.element('elasticBeamColumn', elem, col1[i], vig[0], b1*h1, 20000, 7000, b1*h1**3/12 + h1*b1**3/12, b1*h1**3/12, h1*b1**3/12, geom)
        dibuja_truss(col1[i+1], vig[0], 'r')
        elem += 1
    
    # elementos col2
    if col_der == 1:
        for i in range(nz):
            ops. element('dispBeamColumn', elem, col2[i], col2[i+1], geom, secciones[2])
            dibuja_truss(col2[i], col2[i+1], 'b')
            elem += 1
        ops.element('elasticBeamColumn', elem, col2[i], vig[-1], b2*h2, 20000, 7000, b2*h2**3/12 + h2*b2**3/12, b2*h2**3/12, h2*b2**3/12, geom)
        dibuja_truss(col2[i+1], vig[-1], 'r')
        elem += 1
    
    # elementos viga => la dejo elástica por el posible uso de diafragma rígido
    for i in range(nlong+2):
        if i > 0 and i < nlong+1:
            ops.element('elasticBeamColumn', elem, vig[i], vig[i+1], b3*h3, 20000, 7000, b3*h3**3/12 + h3*b3**3/12, b3*h3**3/12, h3*b3**3/12, geom)
            dibuja_truss(vig[i], vig[i+1], 'r')
        else:
            ops.element('elasticBeamColumn', elem, vig[i], vig[i+1], b3*h3, 20000, 7000, b3*h3**3/12 + h3*b3**3/12, b3*h3**3/12, h3*b3**3/12, geom)
            dibuja_truss(vig[i], vig[i+1], 'r')
        elem += 1
    
    px, py, pz = dibuja_portico(col1,col2,vig,ax)
    print('VIGA Y COLUMNAS INGRESADAS')
    
    fin = time.process_time()
    print(f'TIEMPO DE GENERACIÓN DEL MODELO = {fin-ini}')
    
    return nudo+2, nudo_i, base_nodes, elem, vig, col1, col2


