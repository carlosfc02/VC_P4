import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_FILE_PATH = "resultados.csv"

GROUND_TRUTH_MAP = {
    1:'3324HZC',
    295:'3713LKC',
    665:'3324HMS',
    575:'0848MCG',
    776:'1623LKZ',
    736:'6713HCR',
    802:'2914FGN',
    852:'9786LCF',
    851:'0053JFM',
    938:'0216HZK',
    1030:'0285FKZ',
    1123:'3685KMW',
    1201:'9059LYY',
    1288:'8536LKJ',
    1369:'5695LTR',
    1464:'1770JYG',
    1595:'7339LGG',
    1592:'8098JXM',
    1694:'GC5273CM',
    1733:'5143LCY',
    1682:'1965KBP',
    1763:'9733KFW'
}


def limpiar_numero_excel(valor):
    """
    Limpia los números que Python guardó como '="0,1234"'
    para que Pandas pueda leerlos como 0.1234
    """
    if isinstance(valor, str):
        valor_limpio = valor.strip('="').replace(',', '.')
        try:
            return float(valor_limpio)
        except ValueError:
            return np.nan
    elif isinstance(valor, (int, float)):
        return float(valor)
    return np.nan


def generar_graficas(metricas, nombres_archivos):

    
    for i, metrica in enumerate(metricas):
        plt.figure(figsize=(8, 6)) # 
        
        # Crear gráfico de barras
        bars = plt.bar(metrica['etiquetas'], metrica['datos'], color=['#007ACC', '#FFC300'])
        
        # Añadir etiquetas de datos
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, 
                     f'{yval:.4f}', ha='center', va='bottom')

        plt.title(metrica['titulo'], fontsize=16)
        plt.ylabel(metrica['ylabel'], fontsize=12)
        plt.xlabel('Modelo OCR', fontsize=12)
        
        if "Tasa de Acierto" in metrica['titulo']:
            plt.ylim(0, 1.0) 
        
        elif "Tiempo Promedio" in metrica['titulo']:
            max_val = max(metrica['datos'])
            if max_val > 0:
                plt.ylim(0, max_val * 1.1)
            else:
                plt.ylim(0, 1) 
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Guardar la figura
        nombre_archivo = nombres_archivos[i]
        plt.savefig(nombre_archivo)
        print(f"Gráfica generada y guardada como: {nombre_archivo}")
        plt.close() 


print(f"Cargando y analizando {CSV_FILE_PATH}...")

# --- 3. Carga y Limpieza del CSV ---
columnas_numericas = [
    'confianza_obj', 
    'confianza_matricula', 
    'tiempo_easyocr', 
    'tiempo_smolvlm'
]
converters = {col: limpiar_numero_excel for col in columnas_numericas}

try:
    df = pd.read_csv(CSV_FILE_PATH, converters=converters)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{CSV_FILE_PATH}'.")
    exit()
except Exception as e:
    print(f"Error al leer el CSV: {e}")
    exit()

# --- 4. Preparación de Datos ---
df['ground_truth'] = df['identificador_tracking'].map(GROUND_TRUTH_MAP)
df_validado = df.dropna(subset=['ground_truth']).copy()

if df_validado.empty:
    print("El 'GROUND_TRUTH_MAP' no coincide con ningún 'identificador_tracking' en el CSV.")
    exit()

df_validado['texto_easyocr'] = df_validado['texto_easyocr'].astype(str).str.strip().str.upper()
df_validado['texto_smolvlm'] = df_validado['texto_smolvlm'].astype(str).str.strip().str.upper()
df_validado['ground_truth'] = df_validado['ground_truth'].astype(str).str.strip().str.upper()

df_validado['acierto_easyocr'] = (df_validado['texto_easyocr'] == df_validado['ground_truth']).astype(int)
df_validado['acierto_smolvlm'] = (df_validado['texto_smolvlm'] == df_validado['ground_truth']).astype(int)


# --- 5. Cálculo de Métricas ---

print("\n" + "="*50)
print("     INFORME DE COMPARATIVA DE OCR (COMPLETO)")
print("="*50 + "\n")

# --- MÉTRICA A: TASA DE ACIERTO (POR FOTOGRAMA) ---
print("--- MÉTRICA A: Tasa de Acierto (por Fotograma) ---")
total_fotogramas_validos = len(df_validado)
total_aciertos_easyocr = df_validado['acierto_easyocr'].sum()
total_aciertos_smolvlm = df_validado['acierto_smolvlm'].sum()

if total_fotogramas_validos > 0:
    tasa_acierto_easyocr_fotograma = total_aciertos_easyocr / total_fotogramas_validos
    tasa_acierto_smolvlm_fotograma = total_aciertos_smolvlm / total_fotogramas_validos
else:
    tasa_acierto_easyocr_fotograma = 0.0
    tasa_acierto_smolvlm_fotograma = 0.0

print(f"Total de fotogramas válidos analizados: {total_fotogramas_validos}")
print(f"Aciertos EasyOCR: {total_aciertos_easyocr} / {total_fotogramas_validos}  (Tasa de acierto: {tasa_acierto_easyocr_fotograma:.2%})")
print(f"Aciertos SmolVLM: {total_aciertos_smolvlm} / {total_fotogramas_validos}  (Tasa de acierto: {tasa_acierto_smolvlm_fotograma:.2%})")
print("\n")

print("--- MÉTRICA B: Tasa de Acierto (por Matrícula Única) ---")
aciertos_por_id = df_validado.groupby('identificador_tracking')[['acierto_easyocr', 'acierto_smolvlm']].sum()
unicos_easyocr = (aciertos_por_id['acierto_easyocr'] > 0).sum()
unicos_smolvlm = (aciertos_por_id['acierto_smolvlm'] > 0).sum()
total_matriculas_unicas_anotadas = len(GROUND_TRUTH_MAP)

# Calculamos los porcentajes (0.0 a 1.0)
if total_matriculas_unicas_anotadas > 0:
    tasa_acierto_easyocr_unica = unicos_easyocr / total_matriculas_unicas_anotadas
    tasa_acierto_smolvlm_unica = unicos_smolvlm / total_matriculas_unicas_anotadas
else:
    tasa_acierto_easyocr_unica = 0.0
    tasa_acierto_smolvlm_unica = 0.0

print(f"Total de matrículas únicas anotadas: {total_matriculas_unicas_anotadas}")
print(f"Aciertos EasyOCR: {unicos_easyocr} / {total_matriculas_unicas_anotadas}  (Tasa de acierto: {tasa_acierto_easyocr_unica:.2%})")
print(f"Aciertos SmolVLM: {unicos_smolvlm} / {total_matriculas_unicas_anotadas}  (Tasa de acierto: {tasa_acierto_smolvlm_unica:.2%})")
print("\n")

print("--- MÉTRICA C: Tiempos de Inferencia (Velocidad) ---")
df_tiempos = df_validado[
    (df_validado['tiempo_easyocr'] > 0) | (df_validado['tiempo_smolvlm'] > 0)
]
avg_time_easyocr = df_tiempos['tiempo_easyocr'].mean()
avg_time_smolvlm = df_tiempos['tiempo_smolvlm'].mean()

if pd.notna(avg_time_easyocr) and pd.notna(avg_time_smolvlm) and avg_time_easyocr > 0:
    ratio_velocidad = avg_time_smolvlm / avg_time_easyocr
    print(f"Tiempo promedio EasyOCR: {avg_time_easyocr:.4f} seg/matrícula")
    print(f"Tiempo promedio SmolVLM: {avg_time_smolvlm:.4f} seg/matrícula")
    print(f"-> SmolVLM es {ratio_velocidad:.1f} veces más lento que EasyOCR.")
else:
    print("No se pudieron calcular los tiempos promedio (datos insuficientes o tiempo de EasyOCR fue 0).")
    
print("\n")

# --- 6. GENERACIÓN DE GRÁFICAS ---
try:
    # Definimos los datos para los gráficos
    etiquetas = ['EasyOCR', 'SmolVLM']
    
    metricas_para_graficar = [
        {
            'titulo': 'Tasa de Acierto (por Fotograma)',
            'ylabel': 'Tasa de Acierto (0.0 - 1.0)',
            'datos': [tasa_acierto_easyocr_fotograma, tasa_acierto_smolvlm_fotograma],
            'etiquetas': etiquetas
        },
        {
            'titulo': 'Tasa de Acierto (por Matrícula Única)',
            'ylabel': 'Tasa de Acierto (0.0 - 1.0)',
            'datos': [tasa_acierto_easyocr_unica, tasa_acierto_smolvlm_unica],
            'etiquetas': etiquetas
        },
        {
            'titulo': 'Tiempo Promedio de Inferencia',
            'ylabel': 'Tiempo (segundos)',
            'datos': [avg_time_easyocr, avg_time_smolvlm],
            'etiquetas': etiquetas
        }
    ]
    
    nombres_archivos = [
        'comparativa_acierto_fotograma.png',
        'comparativa_acierto_unica.png',
        'comparativa_tiempos.png'
    ]

    generar_graficas(metricas_para_graficar, nombres_archivos)

except Exception as e:
    print(f"Error al generar las gráficas: {e}")

