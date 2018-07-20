# CNN_Crops

Las CNN pasan la imagen a través de una serie de varias capas de pequeñas colecciones de neuronas/kernels donde cada una de ellas mira una pequeña porción de una imagen y obtiene un resultado. El resultado puede ser una clase única o un conjunto de probabilidades de clase que mejor describa la imagen. Típicamente, las CNN usan pequeños núcleos convolucionales. En consecuencia, las CNN implican menos parámetros que las redes neuronales completamente conectadas [1]. Las arquitecturas CNN más simples se muestran en la figura 

<img src="/images/basic.png" alt="Screenshot1"/>

y se componen de varias capas. Estas capas pueden ser de uno de los siguientes tipos:

  1) Convolutional: durante la convolución, los núcleos se deslizan sobre todos los píxeles de la imagen de entrada. Este     kernel / weights es una matriz con la misma profundidad de entrada. Cada uno de estos núcleos se puede ver como             identificadores de características.
  2) Pooling: esta es una capa de reducción de resolución cuya entrada es la salida de una capa convolucional. La capa de     agrupamiento reduce la cantidad de datos en el dominio del espacio (la profundidad permanece sin cambios) para reducir el   número de parámetros de las capas siguientes, para reducir el costo de cálculo y también para controlar el sobreajuste.
  3) Fully-Connected: toma una capa de volumen de entrada (capa previa) y la convierte en una capa unidimensional,              conectando todas las neuronas de la capa anterior a cada neurona de la capa totalmente conectada.

La capa final en un CNN contiene un solo nodo para cada clase. Las arquitecturas CNN más complejas suelen tener muchas capas convolucionales y de agrupación [2][3]. El enfoque tradicional consiste en apilar todas las imágenes de la secuencia multitemporal para ensamblar, para cada ubicación de píxel, un descriptor que comprenda las características de todas las etapas. Las representaciones construidas de esta manera se utilizan para entrenar un clasificador que asigna una etiqueta de clase a cada píxel a lo largo de la secuencia. En este enfoque, no se tiene en cuenta el contexto del espacio. La figura ilustra el flujo del proceso de la generación del vector característico.

<img src="/images/vector.png" alt="Screenshot2"/>


# EXPERIMENTOS

## DATASETS:
Las imágenes satelitales para ambos datasets consisten de 26 imágenes SENTINEL 2A obtenidas del ESA SCIENTIFIC DATA HUB. Los datos se corrigieron atmosféricamente utilizando el paquete de software SEN2COR estándar. Para asegurar la comparabilidad con la serie LANDSAT, seleccionamos las bandas de resolución de distancia de muestreo de tierra de 10 m (GSD) (es decir, 2 azules, 3 verdes, 4 rojas, 8 infrarrojas cercanas) junto con las bandas GSD de 20 m (es decir, 11 de onda corta-infrarrojo-1, 12 de onda corta-infrarrojo-2) muestreados a 10 m GSD por KNN usando QGIS. La data puede ser encontrada en el siguiente link: http://weegee.vision.ucmerced.edu/datasets/landuse.html

## IMPLEMENTACION:
Dentro del preprocesamiento, se aplicará una corrección atmosférica utilizando el paquete de software SEN2COR estándar. Para asegurar la comparabilidad con la serie LANDSAT, se seleccionará las bandas de resolución de distancia de muestreo de tierra de 10 m (GSD) (es decir, 2 azules, 3 verdes, 4 rojas, 8 infrarrojas cercanas) junto con las bandas GSD de 20 m (es decir, 11 de onda corta-infrarrojo-1, 12 de onda corta-infrarrojo-2) muestreados a 10 m GSD por KNN usando QGIS.

<img src="/images/pipline.png" alt="Screenshot3"/>

Configuración de hyper parametros θc = (lc, rc) 
Los hiperparámetros se eligieron mediante una búsqueda de cuadrícula, de modo que todas las combinaciones de la cantidad de capas de red s lc ∈ {2, 3, 4} y número de celdas por capa rc ∈ {110, 165, 220, 330, 440} testeadas.

Aunque cada capa se podría inicializar con un número diferente de celdas, lo que podría beneficiar a la clasificación, optamos por mantener la complejidad de la búsqueda de grillas moderada y aplicar el mismo número de celdas a cada capa. Para reducir el sobreajuste en los datos de entrenamiento presentados, agregamos la normalización drop_out para mantener probabilidad pkeep = 0.5.
 θCNN = (3, 440) 
 
 ## RESULTADOS:
 
 <img src="/images/entropia_cnn.png" alt="Screenshot4"/>
 
 <img src="/images/cnn_all.png" alt="Screenshot5"/>
 
 <img src="/images/metricas_cnn.png" alt="Screenshot6"/>
 
# REFERENCIAS
 
[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT press, 2016

[2] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European conference on computer vision. Springer, 2014, pp. 818–833.

[3] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions, in Proceedings of the IEEE C
