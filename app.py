# API unificada: Tuya IoT + EDA/IQR + XGBoost (GridSearchCV & Data Drift)
# Versión: FINAL PRODUCCIÓN - Tabla Única, IA Cara Sucia, Fix Constraints, Endpoints IA y Fix Preprocesamiento

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import hashlib
import hmac
import time
import requests
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os
import psycopg2
import psycopg2.extras
import json
import pandas as pd
import numpy as np
import joblib
import logging
import sys
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

# ==========================================
# CONFIGURACIÓN DE LOGS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ==========================================
# CONFIG: Dispositivos, DB y Modelo
# ==========================================
ID_CARA_SUCIA = os.getenv("TUYA_DEVICE_ID", "bf9b2ec293a9f9b528lkdl")
ID_NAHUIZALCO = "bfbb9424274a58f7c805lh"
ID_JUAYUA     = "bfc04053ebf458efd9dil7"
CLIENT_ID = os.getenv("TUYA_CLIENT_ID", "dhd4knqghttrtrx3n5vu")
ACCESS_SECRET = os.getenv("TUYA_ACCESS_SECRET", "d51e817b7fec4b6091b51a2cc3c323d5")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require")

SENSORS_MAP = {ID_CARA_SUCIA: "Cara Sucia", ID_NAHUIZALCO: "Nahuizalco", ID_JUAYUA: "Juayúa"}

if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
RUTA_MODELO = 'data/cerebro_xgboost_carasucia.joblib'
SENSORES_IA = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']
LIMITE_FATIGA = 5  
CANTIDAD_MINIMA_ENTRENAMIENTO = 100 # Umbral de arranque

# ==========================================
# TUYA AUTH & REQUESTS
# ==========================================
current_token = None
token_expires_at = None
token_lock = threading.Lock()

def get_tuya_token():
    timestamp = str(int(time.time() * 1000))
    url_path = "/v1.0/token?grant_type=1"
    content_sha256 = hashlib.sha256("".encode()).hexdigest()
    string_to_sign = f"GET\n{content_sha256}\n\n{url_path}"
    str_to_sign = CLIENT_ID + timestamp + string_to_sign
    signature = hmac.new(ACCESS_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).hexdigest().upper()
    url = f"https://openapi.tuyaeu.com{url_path}"
    headers = {"client_id": CLIENT_ID, "sign": signature, "t": timestamp, "sign_method": "HMAC-SHA256"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get("success"):
            return {"token": data["result"]["access_token"], "expires_in": data["result"].get("expire_time", 7200)}
        return {"error": f"Error Tuya Token: {data}"}
    except Exception as e:
        return {"error": str(e)}

def ensure_valid_token():
    global current_token, token_expires_at
    with token_lock:
        now = datetime.now(timezone.utc)
        if not current_token or not token_expires_at or now >= (token_expires_at - timedelta(minutes=5)):
            token_result = get_tuya_token()
            if "error" in token_result: return token_result
            current_token = token_result["token"]
            token_expires_at = now + timedelta(seconds=token_result["expires_in"])
            logger.info(f"✅ Token renovado. Expira: {token_expires_at}")
        return {"token": current_token}

def calculate_tuya_signature(access_token, method, url_path, body=""):
    timestamp = str(int(time.time() * 1000))
    content_sha256 = hashlib.sha256(body.encode()).hexdigest()
    string_to_sign = f"{method}\n{content_sha256}\n\n{url_path}"
    str_to_sign = CLIENT_ID + access_token + timestamp + "" + string_to_sign
    signature = hmac.new(ACCESS_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).hexdigest().upper()
    return {"sign_method": "HMAC-SHA256", "client_id": CLIENT_ID, "t": timestamp, "access_token": access_token, "sign": signature, "Content-Type": "application/json"}

def get_tuya_data(device_id):
    token_result = ensure_valid_token()
    if "error" in token_result: return token_result
    url_path = f"/v1.0/devices/{device_id}/status"
    headers = calculate_tuya_signature(token_result["token"], "GET", url_path)
    try:
        response = requests.get(f"https://openapi.tuyaeu.com{url_path}", headers=headers, timeout=10)
        data = response.json()
        if 'result' in data and data['result']:
            data['result'] = [item for item in data['result'] if item.get('code') != 'alarm_volume']
        return data
    except Exception as e:
        return {"error": str(e), "success": False}

# ==========================================
# EDA E ICCA (CORREGIDO PARA IGNORAR STRINGS)
# ==========================================
def clasificar_calidad_aire(pm25, pm10, co2):
    if pm25 is None or pm10 is None or co2 is None: return "Desconocido"
    if pm25 > 65 or pm10 > 150 or co2 > 1000: return "Crítico"
    elif pm25 <= 15 and pm10 <= 45 and co2 <= 600: return "Óptimo"
    else: return "Moderado"

def aplicar_eda_y_preprocesamiento(df):
    # 1. Asegurarnos estrictamente de que los sensores sean numéricos
    for col in SENSORES_IA:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 2. Interpolar y rellenar nulos SOLO en las columnas numéricas
    if set(SENSORES_IA).issubset(df.columns):
        df[SENSORES_IA] = df[SENSORES_IA].interpolate(method='linear', limit_direction='both')
        df[SENSORES_IA] = df[SENSORES_IA].fillna(df[SENSORES_IA].mean())
    
    # 3. Calcular Outliers con Rango Intercuartílico (IQR)
    for col in SENSORES_IA:
        if col in df.columns and len(df) >= CANTIDAD_MINIMA_ENTRENAMIENTO:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[f'is_outlier_{col}'] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), True, False)
            
    return df

# ==========================================
# BASE DE DATOS Y GUARDADO (FIX CONSTRAINTS)
# ==========================================
def db_connect(): return psycopg2.connect(DATABASE_URL)

def create_tables_if_not_exist():
    sql = """
    CREATE TABLE IF NOT EXISTS registro_sensores (
        id SERIAL PRIMARY KEY, 
        device_id TEXT NOT NULL, 
        recorded_at TIMESTAMP NOT NULL,
        air_quality_index TEXT, 
        temp_current DOUBLE PRECISION, 
        humidity_value DOUBLE PRECISION,
        co2_value DOUBLE PRECISION, 
        ch2o_value DOUBLE PRECISION, 
        pm25_value DOUBLE PRECISION,
        pm1 DOUBLE PRECISION, 
        pm10 DOUBLE PRECISION, 
        battery_percentage DOUBLE PRECISION,
        charge_state BOOLEAN, 
        raw JSONB
    );

    DO $$ 
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_device_time') THEN
            ALTER TABLE registro_sensores ADD CONSTRAINT uq_device_time UNIQUE(device_id, recorded_at);
        END IF;
    END $$;

    CREATE TABLE IF NOT EXISTS predicciones_log (
        id SERIAL PRIMARY KEY,
        device_id TEXT NOT NULL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        prediccion_icca TEXT,
        valor_real_pm25 DOUBLE PRECISION,
        valor_predicho_pm25 DOUBLE PRECISION,
        error_absoluto DOUBLE PRECISION,
        acertado BOOLEAN,
        reentrenamiento_activado BOOLEAN DEFAULT FALSE
    );
    """
    conn = None
    try:
        conn = db_connect(); cur = conn.cursor()
        cur.execute(sql); conn.commit(); cur.close()
        logger.info("✅ Estructura de BD inicializada correctamente.")
    except Exception as e: logger.error(f"⚠️ Error DB Setup: {e}")
    finally:
        if conn: conn.close()

CODE_TO_COLUMN = {
    "air_quality_index": "air_quality_index", "temp_current": "temp_current",
    "humidity_value": "humidity_value", "co2_value": "co2_value",
    "ch2o_value": "ch2o_value", "pm25_value": "pm25_value",
    "pm1": "pm1", "pm10": "pm10", "battery_percentage": "battery_percentage",
    "charge_state": "charge_state"
}

def save_full_reading(device_id, full_data):
    conn = None
    try:
        conn = db_connect(); cur = conn.cursor()
        
        # MODO PRODUCCIÓN: Guardado cada hora en punto (00)
        recorded_at_now = datetime.now(ZoneInfo("America/El_Salvador"))
        naive_dt = recorded_at_now.replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
        
        raw_json = json.dumps(full_data, default=str)

        cols = {col: None for col in CODE_TO_COLUMN.values()}
        for it in (full_data.get("result") or []):
            code = it.get("code")
            if code in CODE_TO_COLUMN:
                val = it.get("value")
                if code == "charge_state":
                    cols[CODE_TO_COLUMN[code]] = bool(val) if not isinstance(val, str) else val.lower() == "true"
                else:
                    try: cols[CODE_TO_COLUMN[code]] = float(val) if val is not None and str(val).strip() != "" else None
                    except: cols[CODE_TO_COLUMN[code]] = None

        categoria_icca = clasificar_calidad_aire(cols["pm25_value"], cols["pm10"], cols["co2_value"])
        cols["air_quality_index"] = categoria_icca

        cur.execute("""
            INSERT INTO registro_sensores (device_id, recorded_at, air_quality_index, temp_current, humidity_value, co2_value, 
            ch2o_value, pm25_value, pm1, pm10, battery_percentage, charge_state, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (device_id, recorded_at) DO UPDATE SET
                air_quality_index = EXCLUDED.air_quality_index,
                temp_current = EXCLUDED.temp_current,
                humidity_value = EXCLUDED.humidity_value,
                co2_value = EXCLUDED.co2_value,
                pm25_value = EXCLUDED.pm25_value,
                pm10 = EXCLUDED.pm10,
                raw = EXCLUDED.raw
            RETURNING id;
        """, (device_id, naive_dt, cols["air_quality_index"], cols["temp_current"], cols["humidity_value"], cols["co2_value"],
              cols["ch2o_value"], cols["pm25_value"], cols["pm1"], cols["pm10"], cols["battery_percentage"], cols["charge_state"], raw_json))
        m_id = cur.fetchone()[0]

        conn.commit(); cur.close()
        
        return {"success": True, "metric_id": m_id, "icca": categoria_icca, "recorded_at": str(naive_dt)}
    except Exception as e:
        if conn: conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        if conn: conn.close()

# ==========================================
# MOTOR DE IA (XGBOOST AISLADO PARA CARA SUCIA)
# ==========================================
def cargar_y_entrenar():
    try:
        conn = db_connect()
        query = f"SELECT * FROM registro_sensores WHERE device_id = '{ID_CARA_SUCIA}' ORDER BY recorded_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty or len(df) < CANTIDAD_MINIMA_ENTRENAMIENTO: 
            logger.warning(f"⏳ IA (Cara Sucia): Esperando acumular {CANTIDAD_MINIMA_ENTRENAMIENTO} registros. Actualmente hay: {len(df)}")
            return None, None

        logger.info(f"🧠 IA: Ejecutando GridSearchCV sobre {len(df)} datos de Cara Sucia...")
        
        df = aplicar_eda_y_preprocesamiento(df)

        for s in SENSORES_IA:
            df[f'{s}_pasado'] = df[s].shift(1)
            df[f'{s}_futuro'] = df[s].shift(-1)
        
        df_train = df.dropna()
        columnas_X = SENSORES_IA + [f'{s}_pasado' for s in SENSORES_IA]
        columnas_Y = [f'{s}_futuro' for s in SENSORES_IA]
        
        param_grid = {
            'estimator__n_estimators': [50, 100, 150],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__max_depth': [3, 5, 7]
        }
        
        base_xgb = MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1))
        grid_search = GridSearchCV(base_xgb, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(df_train[columnas_X], df_train[columnas_Y])
        
        modelo = grid_search.best_estimator_ 
        
        logger.info(f"🎯 Mejores Hiperparámetros Encontrados: {grid_search.best_params_}")
        joblib.dump(modelo, RUTA_MODELO)
        logger.info("✅ IA: Modelo de Cara Sucia optimizado y guardado en disco.")
        return modelo, columnas_X
    except Exception as e:
        logger.error(f"❌ Error entrenamiento: {e}")
        return None, None

def monitor_ia_xgboost():
    modelo, columnas_X = None, None
    prediccion_actual = None
    last_id = 0
    errores_consecutivos = 0

    while True:
        try:
            if modelo is None:
                if os.path.exists(RUTA_MODELO):
                    modelo = joblib.load(RUTA_MODELO)
                    columnas_X = SENSORES_IA + [f'{s}_pasado' for s in SENSORES_IA]
                else:
                    modelo, columnas_X = cargar_y_entrenar()
            
            if not modelo:
                time.sleep(120)
                continue

            conn = db_connect()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if last_id == 0:
                cursor.execute("SELECT id FROM registro_sensores WHERE device_id = %s ORDER BY id DESC LIMIT 1;", (ID_CARA_SUCIA,))
                res = cursor.fetchone()
                last_id = res['id'] if res else 0

            cursor.execute("SELECT * FROM registro_sensores WHERE id > %s AND device_id = %s ORDER BY id ASC;", (last_id, ID_CARA_SUCIA))
            nuevos = cursor.fetchall()

            if nuevos:
                for registro in nuevos:
                    if prediccion_actual is not None:
                        v_real_pm25 = float(registro['pm25_value'] or 0)
                        v_pred_pm25 = float(prediccion_actual[3]) 
                        error = abs(v_real_pm25 - v_pred_pm25)
                        acertado = bool(error <= (abs(v_real_pm25) * 0.05))
                        prediccion_icca = clasificar_calidad_aire(prediccion_actual[3], prediccion_actual[4], prediccion_actual[2])

                        cursor.execute("""
                            INSERT INTO predicciones_log (device_id, prediccion_icca, valor_real_pm25, valor_predicho_pm25, error_absoluto, acertado) 
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (registro['device_id'], prediccion_icca, v_real_pm25, v_pred_pm25, error, acertado))
                        conn.commit()
                        
                        logger.info(f"🔮 PREDICCIÓN EVALUADA (Cara Sucia) -> Real PM2.5: {v_real_pm25:.2f} | Predicho: {v_pred_pm25:.2f} | ¿Acertó?: {acertado}")

                        errores_consecutivos = errores_consecutivos + 1 if not acertado else 0
                        if errores_consecutivos >= LIMITE_FATIGA:
                            logger.warning("🔄 Data Drift detectado en Cara Sucia. Reejecutando GridSearchCV y Reentrenando...")
                            cursor.execute("UPDATE predicciones_log SET reentrenamiento_activado = TRUE WHERE id = (SELECT max(id) FROM predicciones_log)")
                            conn.commit()
                            modelo, columnas_X = cargar_y_entrenar()
                            errores_consecutivos = 0

                    cursor.execute("SELECT * FROM registro_sensores WHERE id < %s AND device_id = %s ORDER BY id DESC LIMIT 1", (registro['id'], ID_CARA_SUCIA))
                    anterior = cursor.fetchone()
                    if anterior:
                        inputs = [float(registro[s] or 0) for s in SENSORES_IA] + [float(anterior[s] or 0) for s in SENSORES_IA]
                        df_X = pd.DataFrame([inputs], columns=columnas_X)
                        prediccion_actual = modelo.predict(df_X)[0]
                    last_id = registro['id']
            
            cursor.close(); conn.close()
            time.sleep(30)
        except Exception as e:
            time.sleep(60)

def periodic_save_job():
    while True:
        try:
            data = get_tuya_data(ID_CARA_SUCIA)
            if "error" not in data:
                res = save_full_reading(ID_CARA_SUCIA, data)
                hora_guardada = res.get('recorded_at') if isinstance(res, dict) else 'Desconocida'
                logger.info(f"⏱️ Muestreo Guardado BD Única (Hora: {hora_guardada})")
        except Exception as e: 
            logger.error(f"Job Error: {e}")
        time.sleep(10 * 60)

# ==========================================
# ENDPOINTS (Flask API - Adaptados a Tabla Única)
# ==========================================
@app.route('/')
def api_info():
    return jsonify({
        "name": "Tuya Sensors API Unified (Tabla Única, Producción & IA Cara Sucia)",
        "version": "10.0 FINAL",
        "min_train_records": CANTIDAD_MINIMA_ENTRENAMIENTO,
        "endpoints": {
            "/api/metrics/all": "Datos de TODOS los sensores para gráficas del Dashboard",
            "/api/ia/hiperparametros": "Ver hiperparámetros óptimos de XGBoost",
            "/api/ia/precision": "Porcentaje de precisión de la IA en tiempo real",
            "/api/metrics": "Métricas históricas (soporta start_date y end_date)",
            "/api/sensors/realtime": "Estado en tiempo real de todos los dispositivos",
            "/api/sensors/formatted": "Obtiene datos formateados de sensores (BD Caché)",
            "/api/health": "Estado de la API y token",
            "/api/save-now": "Fuerza un guardado inmediato en BD",
            "/api/latest-metrics": "Último registro columnar por ID",
            "/api/snapshots": "Último raw por dispositivo general",
            "/api/historial": "Log de predicciones recientes"
        }
    })

@app.route('/api/metrics/all', methods=['GET'])
def get_all_metrics():
    limit = request.args.get('limit', type=int, default=50)
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        datos_completos = {}
        for dev_id, nombre_parque in SENSORS_MAP.items():
            query = """
                SELECT recorded_at, temp_current, humidity_value, co2_value, 
                       pm25_value, pm10, air_quality_index 
                FROM registro_sensores 
                WHERE device_id = %s 
                ORDER BY recorded_at DESC LIMIT %s
            """
            cur.execute(query, (dev_id, limit))
            rows = cur.fetchall()
            
            if rows:
                df = pd.DataFrame([dict(row) for row in rows])
                df = df.sort_values(by='recorded_at', ascending=True)
                df['recorded_at'] = pd.to_datetime(df['recorded_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                datos_completos[nombre_parque] = [{k: (v if pd.notna(v) else None) for k, v in row.items()} for row in df.to_dict(orient='records')]
            else:
                datos_completos[nombre_parque] = []
                
        cur.close(); conn.close()
        return jsonify({"success": True, "data": datos_completos})
    except Exception as e: 
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ia/hiperparametros', methods=['GET'])
def get_hiperparametros_ia():
    try:
        if not os.path.exists(RUTA_MODELO):
            return jsonify({"success": False, "status": "IA no entrenada"}), 404
        modelo = joblib.load(RUTA_MODELO)
        estimator = modelo.estimators_[0] if hasattr(modelo, 'estimators_') else modelo
        params = estimator.get_params()
        return jsonify({
            "success": True,
            "algoritmo": "XGBoost (MultiOutputRegressor)",
            "optimizador": "GridSearchCV",
            "hiperparametros_optimos": {
                "n_estimators": params.get('n_estimators'),
                "learning_rate": params.get('learning_rate'),
                "max_depth": params.get('max_depth')
            }
        })
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ia/precision', methods=['GET'])
def get_precision_ia():
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = """
            SELECT 
                COUNT(*) as total_evaluaciones,
                SUM(CASE WHEN acertado = TRUE THEN 1 ELSE 0 END) as aciertos,
                SUM(CASE WHEN acertado = FALSE THEN 1 ELSE 0 END) as fallos,
                AVG(error_absoluto) as error_promedio_pm25
            FROM predicciones_log 
            WHERE device_id = %s;
        """
        cur.execute(query, (ID_CARA_SUCIA,))
        res = cur.fetchone()
        cur.close(); conn.close()
        
        total = res['total_evaluaciones'] or 0
        aciertos = res['aciertos'] or 0
        fallos = res['fallos'] or 0
        error_prom = float(res['error_promedio_pm25']) if res['error_promedio_pm25'] is not None else 0.0
        porcentaje_precision = round((aciertos / total) * 100, 2) if total > 0 else 0.0

        return jsonify({
            "success": True,
            "dispositivo": "Cara Sucia",
            "resumen": {
                "total_predicciones_evaluadas": total,
                "aciertos": aciertos,
                "fallos": fallos,
                "porcentaje_precision": f"{porcentaje_precision}%",
                "error_absoluto_promedio_pm25": round(error_prom, 3)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sensors/realtime', methods=['GET'])
def get_all_realtime():
    results = []
    for dev_id, name in SENSORS_MAP.items():
        data = get_tuya_data(dev_id)
        results.append({"name": name, "device_id": dev_id, "success": "error" not in data, "data": data.get("result", []), "error": data.get("error", None)})
    return jsonify({"timestamp": int(time.time()), "devices": results})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    limit = request.args.get('limit', type=int)
    device_id = request.args.get('device_id', ID_CARA_SUCIA)
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = "SELECT recorded_at, temp_current, humidity_value, co2_value, ch2o_value, pm25_value, pm1, pm10, battery_percentage, air_quality_index FROM registro_sensores WHERE device_id = %s ORDER BY recorded_at DESC"
        params = [device_id]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if rows:
            df = pd.DataFrame([dict(row) for row in rows])
            df = df.sort_values(by='recorded_at', ascending=True)
            df['recorded_at'] = pd.to_datetime(df['recorded_at']).dt.strftime('%Y-%m-%dT%H:%M:%S')
            data = [{k: (v if pd.notna(v) else None) for k, v in row.items()} for row in df.to_dict(orient='records')]
        else: data = []
        return jsonify({"data": data, "total_records": len(data)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/sensors', methods=['GET'])
def get_sensors():
    device_id = request.args.get("device_id", ID_CARA_SUCIA)
    return jsonify(get_tuya_data(device_id))

@app.route('/api/sensors/formatted', methods=['GET'])
def get_sensors_formatted():
    device_id = request.args.get("device_id", ID_CARA_SUCIA)
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT air_quality_index, temp_current, humidity_value, co2_value, 
                   ch2o_value, pm25_value, pm1, pm10, battery_percentage, charge_state, recorded_at
            FROM registro_sensores 
            WHERE device_id = %s 
            ORDER BY recorded_at DESC LIMIT 1;
        """, (device_id,))
        row = cur.fetchone()
        cur.close(); conn.close()

        if row:
            sensor_names = {
                'air_quality_index': 'Calidad del Aire', 'temp_current': 'Temperatura',
                'humidity_value': 'Humedad', 'co2_value': 'CO₂', 'ch2o_value': 'Formaldehído',
                'pm25_value': 'PM2.5', 'pm1': 'PM1.0', 'pm10': 'PM10',
                'battery_percentage': 'Batería', 'charge_state': 'Estado de Carga'
            }
            formatted_sensors = []
            for code, name in sensor_names.items():
                if code in row and row[code] is not None:
                    formatted_sensors.append({"code": code, "name": name, "value": row[code]})
            
            return jsonify({
                "success": True, 
                "source": "database_cache",
                "timestamp": str(row["recorded_at"]), 
                "sensors": formatted_sensors
            })
    except Exception as e:
        logger.error(f"Error leyendo BD rápida: {e}")

    data = get_tuya_data(device_id)
    if 'error' in data: return jsonify(data)
    
    formatted_data = {"success": True, "source": "tuya_live", "timestamp": int(time.time()), "sensors": []}
    sensor_names = {
        'air_quality_index': 'Calidad del Aire', 'temp_current': 'Temperatura',
        'humidity_value': 'Humedad', 'co2_value': 'CO₂', 'ch2o_value': 'Formaldehído',
        'pm25_value': 'PM2.5', 'pm1': 'PM1.0', 'pm10': 'PM10',
        'battery_percentage': 'Batería', 'charge_state': 'Estado de Carga'
    }
    if 'result' in data and data['result']:
        for item in data['result']:
            sensor = {"code": item.get('code'), "name": sensor_names.get(item.get('code'), item.get('code', '').title()), "value": item.get('value')}
            formatted_data["sensors"].append(sensor)
    return jsonify(formatted_data)

@app.route('/api/token', methods=['GET'])
def get_token_info():
    if not current_token or not token_expires_at: return jsonify({"status": "no_token"})
    now = datetime.now(timezone.utc)
    is_valid = now < token_expires_at
    time_remaining = (token_expires_at - now).total_seconds() if is_valid else 0
    return jsonify({"status": "active" if is_valid else "expired", "expires_at": token_expires_at.isoformat(), "time_remaining_seconds": int(time_remaining)})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": int(time.time()), "service": "API Unificada (Single Table & IA Cara Sucia)"})

@app.route('/api/save-now', methods=['POST', 'GET'])
def save_now():
    device_id = request.args.get("device_id", ID_CARA_SUCIA)
    data = get_tuya_data(device_id)
    if "error" in data: return jsonify({"success": False, "error": data.get("error")}), 500
    result = save_full_reading(device_id, data)
    return jsonify(result)

@app.route('/api/latest-metrics', methods=['GET'])
def latest_metrics():
    device_id = request.args.get("device_id", ID_CARA_SUCIA)
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM registro_sensores WHERE device_id = %s ORDER BY recorded_at DESC LIMIT 1;", (device_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if row and row.get("recorded_at"): row["recorded_at"] = row["recorded_at"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": True, "metrics": row})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/snapshots', methods=['GET'])
def snapshots():
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT DISTINCT ON (device_id) device_id, recorded_at as last_recorded_at, raw FROM registro_sensores ORDER BY device_id, recorded_at DESC;")
        rows = cur.fetchall()
        cur.close(); conn.close()
        for r in rows:
            if r.get("last_recorded_at"): r["last_recorded_at"] = r["last_recorded_at"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": True, "snapshots": rows})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/descargar-cerebro', methods=['GET'])
def descargar_cerebro():
    try:
        if os.path.exists(RUTA_MODELO): return send_file(RUTA_MODELO, as_attachment=True, download_name='cerebro_xgboost_carasucia.joblib')
        return jsonify({"success": False, "error": "El modelo aún no ha alcanzado los 100 registros para entrenar."}), 404
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/estadisticas')
def stats():
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT SUM(CASE WHEN acertado THEN 1 ELSE 0 END) as aciertos, COUNT(*) as total FROM predicciones_log WHERE device_id = %s", (ID_CARA_SUCIA,))
        res = cur.fetchone()
        cur.close(); conn.close()
        t = res['total'] or 0
        a = res['aciertos'] or 0
        return jsonify({"aciertos": int(a), "fallos": int(t - a), "total": int(t)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/historial')
def historial():
    try:
        conn = db_connect(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM predicciones_log WHERE device_id = %s ORDER BY id DESC LIMIT 50", (ID_CARA_SUCIA,))
        logs = cur.fetchall()
        cur.close(); conn.close()
        return jsonify(logs)
    except Exception as e: return jsonify({"error": str(e)}), 500

# ==========================================
# INICIO AUTOMÁTICO
# ==========================================
create_tables_if_not_exist()

if not os.environ.get("WERKZEUG_RUN_MAIN") == "true": 
    threading.Thread(target=periodic_save_job, daemon=True).start()
    threading.Thread(target=monitor_ia_xgboost, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)