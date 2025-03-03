from flask import Flask, request
import csv
import os

app = Flask(__name__)

readings = []
counter = 0

@app.route('/iop', methods=['POST'])
def get_iop_data():
    global counter, readings
    
    try:
        data = request.get_json() 
        piezo_value = data.get('piezo', None)
        fsr_value = data.get('fsr', None)
        iop_value = data.get('iop', None)
        if piezo_value is not None and fsr_value is not None and iop_value is not None:
            print(f"Received data: Piezo: {piezo_value}, FSR: {fsr_value}, IOP: {iop_value}")
            
            readings.append([piezo_value, fsr_value, iop_value])
            counter += 1
            if counter == 10:
                file_name = 'sensor_data.csv'
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Piezo', 'FSR', 'IOP'])
                    writer.writerows(readings)
                readings = []
                counter = 0
            return "Data received and saved successfully", 200
        else:
            return "Invalid data format", 400
    except Exception as e:
        print(f"Error: {e}")
        return f"Error processing request: {e}", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
