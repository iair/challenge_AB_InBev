# challenge_AB_InBev

## Introducción

Este es el desafío técnico que se le planteó a la cervecera AB InBev (https://www.abinbev.cl/) para el cual se desarrolló un preprocesamiento inicial y EDA, y 4 POC's de modelos de recomendación. El problema fue planteado como uno de predicción de la siguiente canasta en donde se debe ofrecer una canasta que balancee dos objetivos:

1. Desplegar recomendaciones que quepan en una pantalla de aplicación  
2. Ser lo suficientemente completa para que el cliente escoja el máximo posible de los elementos ofrecidos

Para escoger adecuadamente los algoritmos con los cuales se hicieron los prototipos, se recurrió a un análisis de dos papers recientes (2023) que compilan los resultados obtenidos hasta ahora por los distintos tipos de modelos y que proponen un marco de evaluación con métricas que nos permiten comprender el balance entre repetición y exploración en el cual nos encontramos. Los documentos se pueden encontrar en la carpeta: documents/bibliographic review

## Las preguntas a contestar son las siguientes

Evaluación 
1. “EDA”. La idea de este punto es que integre los dataset compartidos, se plantee una hipótesis como por ejemplo “que variables me ayudarían a realizar predicciones” y como el Feature engineering podría mejorar la performance del modelo. Desarrolle algunas visualizaciones que fundamenten dichas hipótesis.

Respuesta:

2. “Data wrangling y modelado” desarrolle el preprocesing y modelado, justifique asunciones que tome durante el proceso, y el o los modelos 
elegidos.

Respuesta:

3. Evalué el o los modelos desarrollados en distintos periodos, además, justifique las métricas que selecciono y el periodo de tiempo definido para la misma.

Respuesta:


4. “Output”: Defina cual es el output de su modelo. Que columnas va a tener. Para esta parte, va a tener que tomar 2 definiciones: 
    * ¿Cuántas recomendaciones de productos va a hacer para cada cliente? 
    * ¿Cuál va a ser el criterio para definir esta cantidad?


Respuesta:


## Descripción del proceso realizado

A continuación describimos el proceso de lo realizado, los resultados de cada proceso se puden encontrar en los notebooks correspondientes

### Preprocessing (preprocesssing_EDA.ipynb)

En los dataset de atributos y transacciones se ejecutan los siguientes análisis:
        * Calidad de los datos:
            * Revisar e imputar valores nulos cuando es necesario
            * Revisar valores fuera de rango y analizar posible impacto cuando es necesario
            * Cuantificar outliers y analizar posible impacto cuando es necesario
        * Transformaciones:
            * Se ejecutan algunas transformaciones de formato de fechas, números y textos
            * Se generan agregaciones para disponer de datasets que simplifiquen el EDA

### EDA -- Se contestan estas preguntas para tener una radiografía clara de mis clientes en base a su comportamiento de consumo y atributos entregados (preprocesssing_EDA.ipynb)

El objetivo de este análisis es entender la magnitud del problema, las características del comportamiento transaccional y de los atributos asignados a los clientes. Uno de los aspectos más importantes en este análisis es entender la factibilidad de ejecutar algoritmos de clusterización para segmentar a los usuarios, así como también comprender si los primeros algoritmos que debemos usar en estas POC con frecuentistas, de filtro colaborativo , secuenciales , factorización de matrices, aprendizaje profundo y/o híbrido

**Sobre las clientes**

* **Análisis a la caracterización de los clientes**
    * ¿A cúantos clientes se les debe hacer recomendación? ¿Cuánto compran los clientes en cada compra? ¿Con qué frecuencia los clientes realizan compras?
    * Si los categorizo ¿Hay diferencias significativas en la distribución de clientes por categoría? ¿Las variables de consumo son distintas entre las categorías?  ¿Estas categorías son independientes entre si? 
  
* **Análisis transaccional general**
    * Con qué frecuencia
    * Cuántos productos se llevan en cada compra
    * Cuántos items de esos productos se llevan en cada compra
    * ¿Cómo varía la frecuencia de compra a lo largo del tiempo?

* **Análisis transaccional temporal**

    * ¿Existen clientes con comportamiento de compra repetitivo? ¿Quienes?
    * ¿Cuántas compras hicieron en el periodo analizado? ¿En qué rango se mueve?
    * ¿cuál es la frecuencia de compra en el periodo analizado?
    * Cuál es el tiempo desde la última compra
    * Qué antiguedad tienen y cómo se distribuye

* **Sobre los items**
    * ¿Cuál es la cantidad total de items disponibles para recomendación?
    * ¿Cuál es el ratio de items por cliente?
    * ¿Cuál es la cantidad de items promedio que cada cliente compra?
    * ¿Cuál es la proporción de items que se repiten al menos una vez del total de items comprados por usuario? ¿Cómo cambia a medida que bueno la cantidad de repeticiones?
    


### Data Wrangling y modelado 


### Evaluación de modelos


### Inferencia