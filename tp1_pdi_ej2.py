import cv2
import numpy as np


f_vacio = cv2.imread('formulario_vacio.png', cv2.IMREAD_GRAYSCALE)

f1 = cv2.imread('formulario_01.png', cv2.IMREAD_GRAYSCALE)
f2 = cv2.imread('formulario_02.png', cv2.IMREAD_GRAYSCALE)
f3 = cv2.imread('formulario_03.png', cv2.IMREAD_GRAYSCALE)
f4 = cv2.imread('formulario_04.png', cv2.IMREAD_GRAYSCALE)
f5 = cv2.imread('formulario_05.png', cv2.IMREAD_GRAYSCALE)

#---------------------Datos columnas-----------------------------------

#Obtenemos los índices de las columnas de cada celda

def indices_columnas(form):   
    img_cols = np.sum(form,0)
    cant_cols = 2
    unique_values = np.unique(img_cols)
    top_cols_largest = unique_values[-cant_cols:]
    img_cols_th = img_cols >= min(top_cols_largest) 
    col_indices = np.where(img_cols_th)      
    list_col_indices = col_indices[0].tolist()
    col_ind_start = {}
    for i in range(len(list_col_indices)-2):
        # Verifica si el índice es impar
        if i % 2 != 0:
            # Asigna la clave como un número entero del 1 en adelante
            clave = len(col_ind_start) + 1
            # Asigna el valor como la posición impar de la lista más uno
            valor = list_col_indices[i] + 2
            # Agrega la entrada al diccionario
            col_ind_start[clave] = valor
    col_ind_end = {}
    for i in range(len(list_col_indices)):
        # Verifica si el índice es impar
        if i % 2 == 0 and i > 1:
            # Asigna la clave como un número entero del 1 en adelante
            clave = len(col_ind_end) + 1
            # Asigna el valor como la posición impar de la lista más uno
            valor = list_col_indices[i] - 2
            # Agrega la entrada al diccionario
            col_ind_end[clave] = valor
    valor_int = int((col_ind_end[2] - col_ind_start[2]) / 2)
    col_ind_start[3] = col_ind_start[2] + valor_int +3
    col_ind_end[3] = col_ind_end[2] - (valor_int +3)
    return col_ind_start, col_ind_end

#---------------------Datos filas---------------------------------

#Obtenemos los índices de las filas de cada celda

def indices_filas(form):
    img_rows = np.sum(form,1)
    cant_rows = 1
    unique_values = np.unique(img_rows)
    top_rows_largest = unique_values[-cant_rows:]
    img_rows_th = img_rows >= top_rows_largest 
    row_indices = np.where(img_rows_th)         
    list_row_indices = row_indices[0].tolist()
    row_ind_start = {}
    for i in range(len(list_row_indices)-2):
        # Verifica si el índice es impar
        if i % 2 != 0:
            # Asigna la clave como un número entero del 1 en adelante
            clave = len(row_ind_start) + 1
            # Asigna el valor como la posición impar de la lista más uno
            valor = list_row_indices[i] + 2
            # Agrega la entrada al diccionario
            row_ind_start[clave] = valor   
    row_ind_end = {}
    for i in range(len(list_row_indices)):
        # Verifica si el índice es impar
        if i % 2 == 0 and i > 1:
            # Asigna la clave como un número entero del 1 en adelante
            clave = len(row_ind_end) + 1
            # Asigna el valor como la posición impar de la lista más uno
            valor = list_row_indices[i] - 2
            # Agrega la entrada al diccionario
            row_ind_end[clave] = valor    
    return row_ind_start, row_ind_end 

#-------------------------Campos formulario---------------------------------------

#Obtenemos las celdas de interés

def campos_formulario(form, row_ind_start, row_ind_end, col_ind_start, col_ind_end):
    tipo_form = form[row_ind_start[1]:row_ind_end[1] , col_ind_start[1]:col_ind_end[2]]
    nom_ape = form[row_ind_start[2]:row_ind_end[2] , col_ind_start[2]:col_ind_end[2]]
    edad = form[row_ind_start[3]:row_ind_end[3] , col_ind_start[2]:col_ind_end[2]]
    mail = form[row_ind_start[4]:row_ind_end[4] , col_ind_start[2]:col_ind_end[2]]
    legajo = form[row_ind_start[5]:row_ind_end[5] , col_ind_start[2]:col_ind_end[2]]
    preg1_si = form[row_ind_start[7]:row_ind_end[7] , col_ind_start[2]:col_ind_end[3]]
    preg2_si = form[row_ind_start[8]:row_ind_end[8] , col_ind_start[2]:col_ind_end[3]]
    preg3_si = form[row_ind_start[9]:row_ind_end[9] , col_ind_start[2]:col_ind_end[3]]
    preg1_no = form[row_ind_start[7]:row_ind_end[7] , col_ind_start[3]:col_ind_end[2]]
    preg2_no = form[row_ind_start[8]:row_ind_end[8] , col_ind_start[3]:col_ind_end[2]]
    preg3_no = form[row_ind_start[9]:row_ind_end[9] , col_ind_start[3]:col_ind_end[2]]
    coment = form[row_ind_start[10]:row_ind_end[10] , col_ind_start[2]:col_ind_end[2]]
    campos = [tipo_form, nom_ape, edad, mail, legajo, preg1_si, preg2_si, preg3_si, preg1_no, preg2_no, preg3_no, coment]
    return campos


#Obtenemos cantidad de palabras y caracteres
def comp_conectados_espacios(celda):    
    f_point = celda < 150
    f_point = f_point.astype(np.uint8)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_point, connectivity, cv2.CV_32S)
    caracteres = 0
    palabras = 0
    if  num_labels <= 2:
        caracteres = num_labels - 1
        palabras = num_labels - 1
    else:   
        ind_ord = np.argsort(stats[:,0])
        stats_ord = stats[ind_ord]
        resultados = []
        for i in range(2,num_labels):
            fila_actual = stats_ord[i]
            fila_anterior = stats_ord[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])        
            resultados.append(suma)    
        espacios = 0
        for valor in resultados:
            if valor >= 9:
                espacios += 1
        palabras = espacios + 1
        caracteres = num_labels + espacios - 1
    return caracteres, palabras


#Obtenemos la cantidad de píxeles de de la letra del formulRIO
def num_pix_letra_for(f): 
    img_th = f < 200    
    row_inicio, row_final = indices_filas(img_th)
    col_inicio, col_final = indices_columnas(img_th) 
    campos = campos_formulario(f,row_inicio, row_final, col_inicio, col_final)
    f_point = campos[0] < 150
    f_point = f_point.astype(np.uint8)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_point, connectivity, cv2.CV_32S)
    ind_ord = np.argsort(stats[:,0])
    stats_ord = stats[ind_ord]
    valor_letra = stats_ord[-1, -1]
    return valor_letra





def formularios(f, nro):  
    
    
    def tipo_formulario(form):
        letra = num_pix_letra_for(form)
        tipo_form = "A" if letra == letra_a  else "B"
        return tipo_form        
    def nombre_apellido(nom_ape):
        caracteres, palabras = comp_conectados_espacios(nom_ape)
        result_nomb_ape = "OK" if 1 < caracteres <= 25 and palabras > 1 else "MAL"
        return result_nomb_ape
    def edad(edad):
        caracteres, palabras = comp_conectados_espacios(edad)
        result_edad = "OK" if 0 < caracteres <= 3 and palabras == 1 else "MAL"
        return result_edad
    def mail(mail):
        caracteres, palabras = comp_conectados_espacios(mail)
        result_mail = "OK" if 1 < caracteres <= 25 and palabras == 1 else "MAL"
        return result_mail  
    def legajo(legajo):
        caracteres, palabras = comp_conectados_espacios(legajo)
        result_legajo = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return result_legajo
    def preg1(preg_si, preg_no):
        caracteres_si, palabras_si = comp_conectados_espacios(preg_si)
        caracteres_no, palabras_no = comp_conectados_espacios(preg_no)
        if (caracteres_si == 1 and caracteres_no != 1) or (caracteres_si != 1 and caracteres_no == 1):
            result_preg1 = "OK"
        else:
            result_preg1 = "MAL"
        return result_preg1   
    def preg2(preg_si, preg_no):
        caracteres_si, palabras_si = comp_conectados_espacios(preg_si)
        caracteres_no, palabras_no = comp_conectados_espacios(preg_no)
        if (caracteres_si == 1 and caracteres_no != 1) or (caracteres_si != 1 and caracteres_no == 1):
            result_preg2 = "OK"
        else:
            result_preg2 = "MAL"
        return result_preg2   
    def preg3(preg_si, preg_no):
        caracteres_si, palabras_si = comp_conectados_espacios(preg_si)
        caracteres_no, palabras_no = comp_conectados_espacios(preg_no)
        if (caracteres_si == 1 and caracteres_no != 1) or (caracteres_si != 1 and caracteres_no == 1):
            result_preg3 = "OK"
        else:
            result_preg3 = "MAL"
        return result_preg3   
    def comentario(coment):
        caracteres, palabras = comp_conectados_espacios(coment)
        result_comentario = "OK" if 1 < caracteres <= 25 and palabras >= 1 else "MAL"
        return result_comentario   
    
    img_th = f < 200
    
    row_inicio, row_final = indices_filas(img_th)
    col_inicio, col_final = indices_columnas(img_th)
    
    campos = campos_formulario(f,row_inicio, row_final, col_inicio, col_final)
    
    resultados = {
        "Tipo de Formulario": tipo_formulario(f),
        "Nombre y Apellido": nombre_apellido(campos[1]),
        "Edad": edad(campos[2]),
        "Mail": mail(campos[3]),
        "Legajo": legajo(campos[4]),
        "Pregunta 1": preg1(campos[5],campos[8]),
        "Pregunta 2": preg2(campos[6],campos[9]),
        "Pregunta 3": preg3(campos[7],campos[10]),
        "Comentario": comentario(campos[11])
    }
    print("+------------------------+-----------+")
    print(f"|         Formulario {nro}               |")
    print("+------------------------+-----------+")
    print("| Campo                  | Resultado |")
    print("+------------------------+-----------+")
    for campo, resultado in resultados.items():
        print(f"| {campo:<22} |   {resultado:<7} |")
    print("+------------------------+-----------+")

letra_a = num_pix_letra_for(f_vacio)

formularios(f1,1)
formularios(f2,2)
formularios(f3,3)
formularios(f4,4)
formularios(f5,5)