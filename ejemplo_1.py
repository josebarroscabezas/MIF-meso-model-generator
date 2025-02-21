# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:17:39 2025

@author: jo_cb
"""
import MIF_MMG as mm
import matplotlib.pyplot as plt
import openseespy.opensees as ops

#%% INGRESO DE DATOS
## Pared de mampostería
direccion = 'x'
nlong, nz = 6, 6 # mallado en dirección longitudinal y altura de la pared
long, esp, alt = 600, 50, 600 # longitud, espesor y altura de la pared
ti = 8 # posición hacia dentro del espesor del bloque donde se desea colocar el puntal
x0, y0, z0 = 75,75,0 # posición inicial del borde de la pared
xov, zov, Lv, Hv = 200, 200, 200, 200 # posición y geometría de ventana

fc, fcu, ft = -1.1, -0.2, 0.05 # parámetros de resistencia de los puntales
ft2 = ft*20
epsc0 = -0.002 # parámetro de capacidad de deformación de los puntales
nu = 0.2 # módulo de poisson para estimar E y G
parametros_mamposteria = [fc, fcu, ft, ft2, epsc0, nu]

## Pórtico de hormigón armado
Fc, Fy = 20 , 500
As=33.18

b1, b2 = 75, 75 # dimensión de columnas en dirección paralela al plano del muro
h1, h2 = 75, 75 # dimensión de columnas en dirección perpendicular al plano del muro
b3, h3 = 75, 85 # base y altura de viga superior
rec_col, recv = 10, 10 # recubrimiento en columnas y viga, respectivamente
parametros_portico = [b1, h1, b2, h2, b3, h3, rec_col, recv, Fc, Fy]

ops.wipe()
# definición de secciones de vigas y columnas
ops.model('basic','-ndm',3,'-ndf',6)
ops.uniaxialMaterial('Concrete01', 13, -Fc, -0.002, -5,-0.01)
ops.uniaxialMaterial('Steel02', 17, Fy, 200000, 0.14)

#COLUMNAS
ops.section('Fiber', 1,'-GJ',4.6142578e10)
ops.patch( 'rect' , 13 , 20 , 20, -b1/2. , -h1/2 , b1/2 , h1/2 )
ops.layer('straight', 17, 2, As, b1/2 -rec_col, h1/2 -rec_col, b1/2 -rec_col,-(h1/2 -rec_col))
ops.layer('straight', 17, 2, As, rec_col-b1/2, h1/2 -rec_col, rec_col-b1/2,rec_col-h1/2)

ops.section('Fiber', 3,'-GJ',4.6142578e10)
ops.patch( 'rect' , 13 , 20 , 20, -b2/2. , -h2/2 , b2/2 , h2/2 )
ops.layer('straight', 17, 2, As, b2/2 -rec_col, h2/2 -rec_col, b2/2 -rec_col,-(h2/2 -rec_col))
ops.layer('straight', 17, 2, As, rec_col-b2/2, h2/2 -rec_col, rec_col-b2/2,rec_col-h2/2)

#VIGAS
ops.section('Fiber', 2,'-GJ',4.6107e10)
ops.patch( 'rect' , 13 , 20 , 20, -b3/2. , -h3/2 , b3/2 , h3/2 )
ops.layer('straight', 17, 2, 15.9, -b3/2 +recv, -h3/2 +recv, b3/2 -recv, -h3/2 +recv)
ops.layer('straight', 17, 2, 23.76, -b3/2 +recv, h3/2 -recv, b3/2 -recv, h3/2 -recv)

ops.beamIntegration('Lobatto', 1, 1, 2)
ops.beamIntegration('Lobatto', 2, 2, 2)
ops.beamIntegration('Lobatto', 3, 3, 2)
secciones = [1, 2, 3] # columna, viga, columna

ops.geomTransf('PDelta', 1,0,1,0)
ops.geomTransf('PDelta', 2,1,0,0)

nudo1, nudo_i1, base_nodes1, elem1, vig1, col11, col21 = mm.modelo(direccion, nlong, nz, long, esp, alt, ti, parametros_mamposteria,
                                                         parametros_portico, secciones, 0, 0, 0, xov, zov, Lv, Hv, x0, y0, z0)
nudo2, nudo_i2, base_nodes2, elem2, vig2, col12, col22 = mm.modelo('y', 8, nz, 800, esp, alt, ti, parametros_mamposteria,
                                                          parametros_portico, secciones, nudo1+1, elem1+1, 6, xov, zov, Lv, Hv, x0-25-75/2, y0+25+75/2, z0, col_izq=0,nudos_ci=col11,nudo_vig_izq=vig1[0])
nudo3, nudo_i3, base_nodes3, elem3, vig3, col13, col23 = mm.modelo('y', 12, nz, 1200, esp, alt, ti, parametros_mamposteria,
                                                          parametros_portico, secciones, nudo2+1, elem2+1, 18, 500, 0, 200, 500, x0+long-25+75/2, y0+25+75/2, z0, col_izq=0, nudos_ci=col21, nudo_vig_izq=vig1[-1])
nudo4, nudo_i4, base_nodes4, elem4, vig4, col14, col24 = mm.modelo('y', 12, nz, 1200, esp, alt, ti, parametros_mamposteria,
                                                          parametros_portico, secciones, nudo3+1, elem3+1, 24, 0, 0, 0, 0, x0+long-25+75/2, y0+25-75/2-1200, z0, col_der=0, nudos_cd=col21, nudo_vig_der=vig1[-1])