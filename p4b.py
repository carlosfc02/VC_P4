### INICIO DEL CÓDIGO ###
import cv2
import csv
import easyocr
from ultralytics import YOLO
import time
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image

# --- 1. CONFIGURACIÓN INICIAL ---

# --- Carga tus modelos ---
model_general = YOLO('yolo11n.pt') 
model_placas = YOLO('runs/detect/train6/weights/best.pt') 

# --- Inicializa EasyOCR ---
print("Cargando EasyOCR...")
reader = easyocr.Reader(['es'], gpu=True) 
print("EasyOCR cargado.")

# --- Carga de SmolVLM (Versión 256M Instruct) ---
print("Cargando SmolVLM (256M Instruct)...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct" 
device = "cuda" 

# Configuración 4-bit para bajo uso de memoria
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

try:
    processor_vlm = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    model_vlm = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
    ).to(device)

    print("SmolVLM cargado.")

except Exception as e:
    print(f"Error al cargar SmolVLM: {e}")
    exit() 

# --- Prompt para el VLM (Formato "Instruct") ---
vlm_prompt = "USER: <image>\nWhat is the text on the license plate?\nASSISTANT:"

# --- Configuración de Vídeo ---
#VIDEO_IN_PATH = "C0142.MP4"
#VIDEO_OUT_PATH = "video_resultado.mp4"
VIDEO_IN_PATH = "videoCarsMad.mp4"
VIDEO_OUT_PATH = "videoCarsMad_resultado.mp4"
CSV_OUT_PATH = "resultados_carsMad.csv"

# Abre el vídeo de entrada
cap = cv2.VideoCapture(VIDEO_IN_PATH)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el vídeo {VIDEO_IN_PATH}")
    exit()

# Obtiene propiedades del vídeo para el escritor de salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Crea el escritor de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
out_video = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))

# --- Configuración del CSV ---
csv_file = open(CSV_OUT_PATH, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
# Escribe la cabecera del CSV (¡Correcta para guardar todo!)
csv_writer.writerow([
    "fotograma", "tipo_objeto", "confianza_obj", "identificador_tracking", 
    "x1", "y1", "x2", "y2", 
    "confianza_matricula", "mx1", "my1", "mx2", "my2", 
    "texto_easyocr", "tiempo_easyocr", "texto_smolvlm", "tiempo_smolvlm"
])

# --- Variables para conteo ---
tracked_people_ids = set()
tracked_vehicle_ids = set()
frame_count = 0

print("Procesando vídeo (esto será lento con SmolVLM)...")

# --- 2. BUCLE PRINCIPAL DE PROCESAMIENTO ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Fin del vídeo

    frame_count += 1
    
    if frame_count % 100 == 0:
        print(f"Procesando fotograma {frame_count}...")

    results_general = model_general.track(
        frame, 
        persist=True, 
        classes=[0, 2, 5, 7], 
        verbose=False 
    )

    if results_general[0].boxes is None or results_general[0].boxes.id is None:
        out_video.write(frame) 
        continue 

    boxes = results_general[0].boxes.xyxy.cpu().numpy().astype(int)
    track_ids = results_general[0].boxes.id.cpu().int()
    confidences = results_general[0].boxes.conf.cpu().float()
    class_ids = results_general[0].boxes.cls.cpu().int()

    for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
        x1, y1, x2, y2 = box
        track_id = int(track_id)
        class_name = model_general.names[int(cls_id)] 
        
        # --- Inicializa variables para el CSV ---
        plate_conf = 0.0
        mx1, my1, mx2, my2 = 0, 0, 0, 0
        
        plate_text_easyocr = ""
        time_easyocr = 0.0
        plate_text_smolvlm = ""
        time_smolvlm = 0.0

        if class_name == "person":
            tracked_people_ids.add(track_id)
        else: 
            tracked_vehicle_ids.add(track_id)

            try:
                vehicle_crop = frame[y1:y2, x1:x2]
                results_placa = model_placas.predict(vehicle_crop, verbose=False)
                
                if len(results_placa[0].boxes) > 0:
                    best_plate = results_placa[0].boxes[0]
                    plate_conf = float(best_plate.conf.cpu()[0])
                    
                    ### --- MODIFICACIÓN CLAVE: FILTRO DE CONFIANZA --- ###
                    # Si la confianza de la matrícula es baja (ej. 0.31), 
                    # ignórala y no ejecutes los OCRs.
                    if plate_conf > 0.5:
                        
                        # --- INICIO DEL BLOQUE MOVIDO ---
                        m_box = best_plate.xyxy.cpu().numpy().astype(int)[0]
                        
                        mx1, my1 = int(m_box[0]) + x1, int(m_box[1]) + y1
                        mx2, my2 = int(m_box[2]) + x1, int(m_box[3]) + y1
                        
                        plate_crop = frame[my1:my2, mx1:mx2]
                        
                        # --- Ejecuta EasyOCR ---
                        start_time = time.perf_counter()
                        ocr_result_easy = reader.readtext(plate_crop)
                        end_time = time.perf_counter()
                        time_easyocr = end_time - start_time
                        
                        if ocr_result_easy:
                            plate_text_easyocr = " ".join([res[1] for res in ocr_result_easy])
                            plate_text_easyocr = "".join(filter(str.isalnum, plate_text_easyocr)).upper()

                        # --- Ejecuta SmolVLM ---
                        start_time = time.perf_counter()
                        try:
                            pil_image = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                            
                            inputs = processor_vlm(text=vlm_prompt, images=pil_image, return_tensors="pt").to(device)

                            generation_output = model_vlm.generate(
                                **inputs,
                                max_new_tokens=50, 
                                do_sample=False
                            )
                            
                            decoded_text = processor_vlm.decode(generation_output[0], skip_special_tokens=True)
                            
                            answer_part = decoded_text.split("ASSISTANT:")
                            if len(answer_part) > 1:
                                plate_text_smolvlm = answer_part[1].strip()
                                plate_text_smolvlm = "".join(filter(str.isalnum, plate_text_smolvlm)).upper()
                            else:
                                plate_text_smolvlm = "PARSE_ERROR"

                        except Exception as e:
                            print(f"Error en SmolVLM: {e}")
                            plate_text_smolvlm = "ERROR_VLM"
                        
                        end_time = time.perf_counter()
                        time_smolvlm = end_time - start_time

                        # --- Dibuja la caja de la matrícula ---
                        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2) # Rojo
                        
                        # Dibuja el de EasyOCR arriba (Verde)
                        cv2.putText(frame, f"E: {plate_text_easyocr}", (mx1, my1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 

                        # Dibuja el de SmolVLM abajo (Rojo)
                        cv2.putText(frame, f"S: {plate_text_smolvlm}", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # --- FIN DEL BLOQUE MOVIDO ---

            except Exception as e:
                print(f"Error procesando matrícula: {e}") 

        # --- Dibuja la caja del objeto principal (persona/vehículo) ---
        color = (0, 255, 0) if class_name == "person" else (255, 0, 0) 
        label = f"ID:{track_id} {class_name}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Tarea 6: Escribir en CSV (¡Correcto!) ---
        # Esto se ejecuta en CADA iteración, guardando los datos 
        # (vacíos si no se detectó matrícula con confianza)
        csv_writer.writerow([
            frame_count, class_name, float(conf), track_id, 
            int(x1), int(y1), int(x2), int(y2), 
            float(plate_conf), int(mx1), int(my1), int(mx2), int(my2), 
            plate_text_easyocr, time_easyocr, plate_text_smolvlm, time_smolvlm
        ])

    # --- Tarea 5: Volcar a disco ---
    total_people = len(tracked_people_ids)
    total_vehicles = len(tracked_vehicle_ids)
    cv2.putText(frame, f"Personas: {total_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehiculos: {total_vehicles}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out_video.write(frame)

# --- 3. LIMPIEZA ---
print("\nProcesamiento completado.")
print(f"Total personas únicas detectadas: {len(tracked_people_ids)}")
print(f"Total vehículos únicos detectados: {len(tracked_vehicle_ids)}")
print(f"Vídeo de resultados guardado en: {VIDEO_OUT_PATH}")
print(f"Log CSV guardado en: {CSV_OUT_PATH}")

cap.release()
out_video.release()
csv_file.close()


