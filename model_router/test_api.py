import requests
import sys
import os
import json

def test_audio_endpoint(audio_path):
    url = "http://localhost:5000/api/process"
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found -> {audio_path}")
        return
        
    print(f"🚀 Simulating Website Frontend Upload...")
    print(f"-> Sending '{os.path.basename(audio_path)}' to Model Router API ({url})")
    
    with open(audio_path, 'rb') as f:
        # Simulate the exact FormData payload that the website sends
        files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
        data = {'type': 'file'}
        
        try:
            response = requests.post(url, files=files, data=data)
            
            print(f"\n[SERVER STATUS CODE: {response.status_code}]")
            print("--- AI RESPONSE PAYLOAD ---")
            
            try:
                # Pretty print the JSON
                parsed_json = response.json()
                print(json.dumps(parsed_json, indent=4))
            except ValueError:
                # If the server crashed and returned HTML instead of JSON
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("\n❌ CRITICAL ERROR: Could not connect to localhost:5000!")
            print("-> Did you forget to start the Flask app using 'python app.py' in a separate terminal?")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_audio_file>")
        sys.exit(1)
        
    test_audio_endpoint(sys.argv[1])
