from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

readings = []
counter = 0

@app.route('/iop', methods=['POST'])
def get_iop_data():
    global counter, readings

    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON data"}), 400

        data = request.get_json(force=True)
        piezo_value = data.get('piezo')
        fsr_value = data.get('fsr')
        iop_value = data.get('iop')

        if all(val is not None for val in [piezo_value, fsr_value, iop_value]):
            print(f"✅ Received: Piezo={piezo_value}, FSR={fsr_value}, IOP={iop_value}")
            readings.append([piezo_value, fsr_value, iop_value])
            counter += 1

            if counter >= 10:
                file_name = 'sensor_data.csv'
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Piezo', 'FSR', 'IOP'])
                    writer.writerows(readings)
                print("📁 Data written to sensor_data.csv")
                readings = []
                counter = 0

            return jsonify({"status": "success", "message": "Data received"}), 200
        else:
            print("❌ Invalid JSON keys or values:", data)
            return jsonify({"error": "Missing or invalid data fields"}), 400

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/iop', methods=['GET'])
def block_get():
    return jsonify({"error": "GET not allowed. Use POST with JSON."}), 405

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
