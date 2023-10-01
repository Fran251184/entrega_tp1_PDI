import cv2
import numpy as np

def ecualizacion_histograma_local(imagen, tamano_ventana):
    # Hacer una copia de la imagen original para preservarla
    imagen_resultado = imagen.copy()

    # Obtener las dimensiones de la imagen
    altura, ancho = imagen.shape

    # Calcular la mitad del tamaño de la ventana
    mitad_ventana = tamano_ventana // 2

    # Calcular el tamaño de los bordes (la mitad del tamaño de la ventana)
    tamano_bordes = mitad_ventana

    # Crear una copia de la imagen con replicación de valores en los bordes
    imagen_bordes = cv2.copyMakeBorder(imagen, tamano_bordes, tamano_bordes, tamano_bordes, tamano_bordes, cv2.BORDER_REPLICATE)

    for y in range(tamano_bordes, altura + tamano_bordes):
        for x in range(tamano_bordes, ancho + tamano_bordes):
            # Definir la región local usando la ventana
            region_local = imagen_bordes[y - mitad_ventana:y + mitad_ventana + 1, x - mitad_ventana:x + mitad_ventana + 1]

            # Verificar si la región local contiene píxeles con el mismo valor (evitar división por cero)
            if np.all(region_local == region_local[0, 0]):
                continue

            # Calcular el histograma de la región local
            hist, _ = np.histogram(region_local, bins=256, range=(0, 256))

            # Calcular la función de distribución acumulativa (CDF)
            cdf = hist.cumsum()

            # Normalizar el CDF para el rango completo de intensidades
            cdf_normalizado = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

            # Mapear los valores de intensidad de la región local a través del CDF
            resultado_local = cdf_normalizado[region_local]

            # Asignar el resultado de ecualización a la imagen resultante
            imagen_resultado[y - tamano_bordes, x - tamano_bordes] = resultado_local[mitad_ventana, mitad_ventana]
            
    imagen_resultado = cv2.medianBlur(imagen_resultado, 3)        

    return imagen_resultado


# Cargar la imagen
imagen = cv2.imread('imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Llamar a la función de ecualización de histograma local sin modificar la imagen original
imagen_resultado = ecualizacion_histograma_local(imagen, 15)

# Mostrar la imagen original y la imagen ecualizada
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Ecualizada por Histograma Local', imagen_resultado)

# Esperar hasta que se presione una tecla (0 significa esperar indefinidamente)
cv2.waitKey(0)

# Cerrar todas las ventanas de visualización
cv2.destroyAllWindows()
